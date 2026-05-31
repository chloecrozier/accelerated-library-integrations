"""
Lateral-movement / blast-radius analysis with cuGraph.

Vertical: enterprise cybersecurity (SOC, threat hunting, incident response).

Data: a slice of the Los Alamos National Lab "Comprehensive, Multi-Source
Cyber-Security Events" dataset (CC0). Each row is one Windows authentication
event between a source computer and a destination computer. A small number
of rows are flagged with is_redteam=1: these are real, labeled red-team
compromise events from a 58-day enterprise exercise.

Workflow this script demonstrates:
1. Load the auth events into a cuDF DataFrame on the GPU.
2. Build a directed host->host graph with cuGraph.
3. Pick a known-compromised host (from the red-team ground truth).
4. Run BFS from that host -> blast radius (which hosts could the attacker
   pivot to within N hops?).
5. Run PageRank on the directed graph -> identify high-value pivot hubs
   (domain controllers, jump boxes).
6. Combine BFS proximity + PageRank into a risk score.
7. Compare the top-K risk-ranked hosts against the actual red-team
   destinations to see how many real attacker landing points the workflow
   surfaced.

Run on a Brev GPU instance or DGX Spark with RAPIDS installed. Prep the
dataset first with dataset/prepare_lanl.py if dataset/lanl_auth_sample.csv
does not exist.
"""

from pathlib import Path
import sys


try:
    import cudf
    import cugraph
except ImportError as exc:
    raise SystemExit(
        "This example needs RAPIDS cuDF/cuGraph on a CUDA-capable Linux or "
        "WSL2 machine. Run it on Brev or DGX Spark, not local macOS."
    ) from exc


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "lanl_auth_sample.csv"


def main():
    if not DATASET.exists():
        sys.exit(
            f"Missing dataset at {DATASET}\n"
            "Generate it once by running dataset/prepare_lanl.py against the "
            "raw LANL auth.txt.gz and redteam.txt.gz files."
        )

    # 1. Load the CSV into a GPU DataFrame.
    df = cudf.read_csv(str(DATASET))
    print(f"Loaded {len(df):,} authentication events")
    print(f"  red-team events:  {int(df['is_redteam'].sum()):,}")
    print(f"  unique src hosts: {df['src_computer'].nunique():,}")
    print(f"  unique dst hosts: {df['dst_computer'].nunique():,}")
    print()

    # 2. cuGraph needs integer vertex ids, so map each computer name to an int.
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

    # 3. Build the directed graph (for PageRank) and undirected (for BFS).
    directed_graph = cugraph.Graph(directed=True)
    directed_graph.from_cudf_edgelist(edges, source="src", destination="dst")

    undirected_graph = cugraph.Graph(directed=False)
    undirected_graph.from_cudf_edgelist(edges, source="src", destination="dst")

    # 4. Pick a compromised seed host: the source computer that initiated
    #    the most red-team events. In a real SOC this would come from the
    #    EDR or SIEM that triggered the alert.
    red_edges = edges[edges["is_redteam"] == 1]
    seed_id = int(
        red_edges.groupby("src")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .iloc[0]["src"]
    )
    seed_name = hosts[hosts["id"] == seed_id]["host"].iloc[0]
    print(f"Known-compromised seed host: {seed_name} (vertex {seed_id})")
    print()

    # 5. BFS from the seed = blast radius.
    bfs = cugraph.bfs(undirected_graph, start=seed_id)
    reachable = bfs[bfs["distance"] >= 0]
    print(f"Hosts reachable from seed: {len(reachable):,}")

    # 6. PageRank = pivot-hub centrality on the directed auth graph.
    pagerank = cugraph.pagerank(directed_graph)

    # 7. Risk score = PageRank weight + BFS proximity to the seed.
    risk = pagerank.merge(
        bfs[["vertex", "distance"]], on="vertex", how="left"
    )
    risk["distance"] = risk["distance"].fillna(-1).astype("int32")
    # proximity: closer to the seed -> higher. Unreachable hosts (distance < 0)
    # score 0. Vectorized so it stays on the GPU (no per-row Python UDF).
    risk["proximity"] = (1.0 / (risk["distance"].clip(lower=0) + 1)).where(
        risk["distance"] >= 0, 0.0
    )
    risk["risk_score"] = risk["pagerank"] * 100.0 + risk["proximity"] * 10.0
    risk = risk.merge(
        hosts.rename(columns={"id": "vertex", "host": "host_name"}),
        on="vertex",
        how="left",
    )

    top = (
        risk.sort_values("risk_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    print()
    print("Top 20 hosts by risk score (PageRank + BFS proximity)")
    print("-" * 55)
    print(
        top[["host_name", "pagerank", "distance", "risk_score"]]
        .to_pandas()
        .to_string(index=False)
    )

    # 8. Ground-truth check: how many of the real red-team destination hosts
    #    did our risk ranking surface in the top 20?
    real_red_dst = set(red_edges["dst"].to_pandas().unique().tolist())
    top_vertices = set(top["vertex"].to_pandas().tolist())
    hits = real_red_dst & top_vertices
    print()
    print(
        f"Red-team destinations recovered in top 20: "
        f"{len(hits)}/{len(real_red_dst)}"
    )
    print()
    print(
        "Workflow fit: a SOC analyst feeds the compromised host into the "
        "pipeline, cuGraph ranks every other host by attacker pivot risk, "
        "and the top of the list becomes the isolation / investigation queue."
    )


if __name__ == "__main__":
    main()
