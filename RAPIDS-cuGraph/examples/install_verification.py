"""End-to-end smoke test for a RAPIDS cuGraph install.

Confirms cuDF/cuGraph import, a CUDA device is visible, and simple graph
algorithms can run on GPU-backed data. This is the first script to run after
creating the environment.
"""

import sys


try:
    import cudf
    import cugraph
    import cupy as cp
except ImportError as exc:
    sys.exit(f"FAIL: RAPIDS cuDF/cuGraph/CuPy not importable -> {exc}")


print(f"cuDF version:    {cudf.__version__}")
print(f"cuGraph version: {getattr(cugraph, '__version__', 'unknown')}")
print(f"CuPy version:    {cp.__version__}")

if cp.cuda.runtime.getDeviceCount() == 0:
    sys.exit("FAIL: no CUDA devices detected")

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
runtime = cp.cuda.runtime.runtimeGetVersion()

print(f"CUDA runtime:    {runtime // 1000}.{(runtime % 1000) // 10}")
print(f"GPU 0:           {name} (compute {props['major']}.{props['minor']})")

edges = cudf.DataFrame(
    {
        "src": [0, 1, 2, 2, 3, 4],
        "dst": [1, 2, 0, 3, 4, 2],
        "weight": [1.0, 1.0, 1.0, 0.5, 0.5, 0.25],
    }
)

graph = cugraph.Graph(directed=True)
graph.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight")

pagerank = cugraph.pagerank(graph)
bfs = cugraph.bfs(graph, start=0)
cp.cuda.runtime.deviceSynchronize()

if len(pagerank) != 5:
    sys.exit(f"FAIL: expected PageRank for 5 vertices, got {len(pagerank)}")

if int((bfs["distance"] >= 0).sum()) < 5:
    sys.exit("FAIL: BFS did not reach all expected vertices")

print("PASS: cuGraph is installed and GPU graph algorithms completed.")
