# Cyber lateral-movement benchmark: CPU vs GPU

Lateral-movement / blast-radius pipeline (PageRank + BFS) from `cyber_lateral_movement.py`, timed CPU (NetworkX) vs GPU (cuGraph).

## Method

- Data is loaded and the integer-keyed edgelist is built **once, outside all timers**, so this measures the graph engine (cuGraph vs NetworkX), not cuDF vs pandas I/O.
- GPU: one warm-up run discarded (CUDA init/JIT); `deviceSynchronize()` before stopping every GPU timer (asynchronous kernels); median of 7 runs.
- CPU: median of 3 runs.
- Both sides build a directed graph (PageRank) and an undirected graph (BFS) from the same edgelist, and run the same two algorithms from the same seed host.

## Graph

- Hosts (vertices): 11,775
- Edges: 341,216
- Auth events: 341,216 (316 red-team)

## Environment

- GPU: NVIDIA L4
- CPU: x86_64
- cuGraph 26.04.000, cuDF 26.04.000, NetworkX 3.6.1, Python 3.12.13

## Results

| Stage | CPU (NetworkX) ms | GPU (cuGraph) ms | Speedup |
|---|---:|---:|---:|
| Graph build (directed + undirected) | 800.2 | 14.8 | 53.9x |
| PageRank + BFS (algorithm-only) | 104.0 | 12.2 | 8.5x |
| End-to-end (build + algorithm) | 904.2 | 27.1 | 33.4x |

> The two rows you asked for: **algorithm-only** isolates the PageRank+BFS compute; **end-to-end** adds graph construction from the in-memory edgelist. Graph build is broken out as its own row.
