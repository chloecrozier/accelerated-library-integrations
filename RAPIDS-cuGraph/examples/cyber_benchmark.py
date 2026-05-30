"""
CPU vs GPU benchmark for the lateral-movement cyber workflow.

This times the *analytics engine* only, not data loading. The CSV is read and
the integer-keyed edgelist is built once, up front, OUTSIDE all timers, so the
numbers reflect cuGraph vs NetworkX rather than cuDF vs pandas I/O.

Two scenarios are reported per platform, matching the demo pipeline:
  * Algorithm-only : PageRank (directed) + BFS (undirected) on a pre-built graph.
  * End-to-end     : graph build from the in-memory edgelist + the algorithms.

GPU timing correctness:
  * One warm-up run is discarded (CUDA context init / JIT).
  * cp.cuda.runtime.deviceSynchronize() is called before stopping every GPU
    timer, because cuGraph kernels launch asynchronously.

Results are written to ../cyber_benchmark_results.md.

Run with the RAPIDS venv:
    /home/ubuntu/.venv/bin/python examples/cyber_benchmark.py
"""

from pathlib import Path
import platform
import statistics
import sys
import time

try:
    import cudf
    import cugraph
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "This benchmark needs RAPIDS cuDF/cuGraph/cuPy on a CUDA machine. "
        "Run it with /home/ubuntu/.venv/bin/python."
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "NetworkX is the CPU baseline. Install it into the venv:\n"
        "    /home/ubuntu/.venv/bin/python -m pip install networkx"
    ) from exc


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
DATASET = PROJECT / "dataset" / "lanl_auth_sample.csv"
RESULTS_MD = PROJECT / "cyber_benchmark_results.md"

GPU_REPEATS = 7          # GPU work is cheap, average over more runs
CPU_REPEATS = 3          # NetworkX PageRank is slow, fewer runs
PAGERANK_MAX_ITER = 200  # give NetworkX room to converge


def median_ms(times):
    return statistics.median(times) * 1000.0


def gpu_sync():
    cp.cuda.runtime.deviceSynchronize()


# ---------------------------------------------------------------------------
# Shared prep: load once, build the integer edgelist once. NOT timed.
# ---------------------------------------------------------------------------
def prepare():
    if not DATASET.exists():
        sys.exit(
            f"Missing dataset at {DATASET}\n"
            "Generate it with dataset/prepare_lanl.py first."
        )

    df = cudf.read_csv(str(DATASET))

    hosts = (
        cudf.concat([df["src_computer"], df["dst_computer"]])
        .drop_duplicates()
        .reset_index(drop=True)
        .to_frame(name="host")
    )
    hosts["id"] = cudf.Series(range(len(hosts)), dtype="int32")

    edges = (
        df[["src_computer", "dst_computer", "is_redteam"]]
        .merge(
            hosts.rename(columns={"host": "src_computer", "id": "src"}),
            on="src_computer",
            how="left",
        )
        .merge(
            hosts.rename(columns={"host": "dst_computer", "id": "dst"}),
            on="dst_computer",
            how="left",
        )
    )

    # Seed = source host with the most red-team events (same as the demo).
    red_edges = edges[edges["is_redteam"] == 1]
    seed_id = int(
        red_edges.groupby("src")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .iloc[0]["src"]
    )

    edges_gpu = edges[["src", "dst"]]
    # CPU baseline needs the same edgelist in host memory (pandas). Converting
    # here, outside the timers, keeps the comparison engine-vs-engine.
    edges_cpu = edges_gpu.to_pandas()

    stats = {
        "events": len(df),
        "edges": len(edges_gpu),
        "nodes": len(hosts),
        "redteam": int(df["is_redteam"].sum()),
        "seed_id": seed_id,
    }
    return edges_gpu, edges_cpu, stats


# ---------------------------------------------------------------------------
# GPU (cuGraph)
# ---------------------------------------------------------------------------
def gpu_build(edges_gpu):
    directed = cugraph.Graph(directed=True)
    directed.from_cudf_edgelist(edges_gpu, source="src", destination="dst")
    undirected = cugraph.Graph(directed=False)
    undirected.from_cudf_edgelist(edges_gpu, source="src", destination="dst")
    gpu_sync()
    return directed, undirected


def gpu_algo(directed, undirected, seed_id):
    pr = cugraph.pagerank(directed)
    bfs = cugraph.bfs(undirected, start=seed_id)
    gpu_sync()
    return pr, bfs


# ---------------------------------------------------------------------------
# CPU (NetworkX)
# ---------------------------------------------------------------------------
def cpu_build(edges_cpu):
    directed = nx.from_pandas_edgelist(
        edges_cpu, "src", "dst", create_using=nx.DiGraph
    )
    undirected = nx.from_pandas_edgelist(
        edges_cpu, "src", "dst", create_using=nx.Graph
    )
    return directed, undirected


def cpu_algo(directed, undirected, seed_id):
    try:
        pr = nx.pagerank(directed, max_iter=PAGERANK_MAX_ITER)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(directed, max_iter=PAGERANK_MAX_ITER * 5, tol=1e-4)
    # cuGraph BFS returns distance-from-seed for all reachable vertices;
    # single_source_shortest_path_length is the unweighted-BFS equivalent.
    bfs = nx.single_source_shortest_path_length(undirected, seed_id)
    return pr, bfs


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
def time_build_and_algo(build_fn, algo_fn, args, seed_id, repeats, warmup):
    """Return (build_ms, algo_ms) medians. End-to-end = build + algo."""
    if warmup:
        d, u = build_fn(args)
        algo_fn(d, u, seed_id)

    build_times, algo_times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        d, u = build_fn(args)
        build_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        algo_fn(d, u, seed_id)
        algo_times.append(time.perf_counter() - t0)

    return median_ms(build_times), median_ms(algo_times)


def main():
    edges_gpu, edges_cpu, stats = prepare()
    seed_id = stats["seed_id"]

    print(
        f"Graph: {stats['nodes']:,} hosts, {stats['edges']:,} edges "
        f"({stats['events']:,} auth events, {stats['redteam']} red-team).\n"
    )

    print("Benchmarking GPU (cuGraph)...")
    gpu_build_ms, gpu_algo_ms = time_build_and_algo(
        gpu_build, gpu_algo, edges_gpu, seed_id,
        repeats=GPU_REPEATS, warmup=True,
    )

    print("Benchmarking CPU (NetworkX)... (PageRank is slow, be patient)")
    cpu_build_ms, cpu_algo_ms = time_build_and_algo(
        cpu_build, cpu_algo, edges_cpu, seed_id,
        repeats=CPU_REPEATS, warmup=False,
    )

    gpu_e2e = gpu_build_ms + gpu_algo_ms
    cpu_e2e = cpu_build_ms + cpu_algo_ms

    def speedup(cpu, gpu):
        return cpu / gpu if gpu > 0 else float("inf")

    # Environment
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    rows = [
        ("Graph build (directed + undirected)", cpu_build_ms, gpu_build_ms),
        ("PageRank + BFS (algorithm-only)", cpu_algo_ms, gpu_algo_ms),
        ("End-to-end (build + algorithm)", cpu_e2e, gpu_e2e),
    ]

    print("\nResults")
    print("-" * 72)
    print(f"{'Stage':<40}{'CPU ms':>10}{'GPU ms':>10}{'Speedup':>11}")
    for name, c, g in rows:
        print(f"{name:<40}{c:>10.1f}{g:>10.1f}{speedup(c, g):>10.1f}x")

    write_markdown(stats, gpu_name, rows, speedup)
    print(f"\nWrote {RESULTS_MD}")


def write_markdown(stats, gpu_name, rows, speedup):
    cugraph_v = getattr(cugraph, "__version__", "n/a")
    cudf_v = getattr(cudf, "__version__", "n/a")
    nx_v = getattr(nx, "__version__", "n/a")

    lines = []
    lines.append("# Cyber lateral-movement benchmark: CPU vs GPU\n")
    lines.append(
        "Lateral-movement / blast-radius pipeline "
        "(PageRank + BFS) from `cyber_lateral_movement.py`, timed CPU "
        "(NetworkX) vs GPU (cuGraph).\n"
    )
    lines.append("## Method\n")
    lines.append(
        "- Data is loaded and the integer-keyed edgelist is built **once, "
        "outside all timers**, so this measures the graph engine "
        "(cuGraph vs NetworkX), not cuDF vs pandas I/O.\n"
        "- GPU: one warm-up run discarded (CUDA init/JIT); "
        "`deviceSynchronize()` before stopping every GPU timer "
        f"(asynchronous kernels); median of {GPU_REPEATS} runs.\n"
        f"- CPU: median of {CPU_REPEATS} runs.\n"
        "- Both sides build a directed graph (PageRank) and an undirected "
        "graph (BFS) from the same edgelist, and run the same two algorithms "
        "from the same seed host.\n"
    )
    lines.append("## Graph\n")
    lines.append(
        f"- Hosts (vertices): {stats['nodes']:,}\n"
        f"- Edges: {stats['edges']:,}\n"
        f"- Auth events: {stats['events']:,} "
        f"({stats['redteam']} red-team)\n"
    )
    lines.append("## Environment\n")
    lines.append(
        f"- GPU: {gpu_name}\n"
        f"- CPU: {platform.processor() or platform.machine()}\n"
        f"- cuGraph {cugraph_v}, cuDF {cudf_v}, NetworkX {nx_v}, "
        f"Python {platform.python_version()}\n"
    )
    lines.append("## Results\n")
    lines.append("| Stage | CPU (NetworkX) ms | GPU (cuGraph) ms | Speedup |")
    lines.append("|---|---:|---:|---:|")
    for name, c, g in rows:
        lines.append(f"| {name} | {c:,.1f} | {g:,.1f} | {speedup(c, g):.1f}x |")
    lines.append("")
    lines.append(
        "> The two rows you asked for: **algorithm-only** isolates the "
        "PageRank+BFS compute; **end-to-end** adds graph construction from the "
        "in-memory edgelist. Graph build is broken out as its own row.\n"
    )

    RESULTS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
