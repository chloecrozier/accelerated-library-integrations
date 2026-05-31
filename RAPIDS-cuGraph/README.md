# RAPIDS cuGraph

cuGraph is the graph analytics library in RAPIDS. It lets Python data science workflows build graphs from GPU-resident data and run graph algorithms on NVIDIA GPUs.

- **vs NetworkX:** cuGraph is built for larger graph workloads on the GPU. `nx-cugraph` can accelerate supported NetworkX algorithms with little or no source-code change.
- **vs graph databases:** cuGraph is the analytics engine. A graph database handles storage, transactions, APIs, and operational graph queries; cuGraph handles high-throughput algorithms such as PageRank, centrality, components, traversal, similarity, and community detection.

## Purpose & Prerequisites

cuGraph is useful when relationships matter as much as rows: transactions between accounts, links between web pages, connections between users, dependencies between services, or molecular and biological interaction networks.

It integrates naturally with the RAPIDS stack:

1. Load and transform edge lists with cuDF.
2. Build a graph from cuDF, pandas, CuPy, or SciPy data.
3. Run GPU-accelerated graph algorithms with cuGraph.
4. Join graph features back into a cuDF or cuML workflow.

Required:
- Linux with `glibc >= 2.28`, or Windows 11 via WSL2
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA 12.2-12.9 with driver 535+, or CUDA 13.0-13.1 with driver 580+
- Python 3.11, 3.12, 3.13, or 3.14 for RAPIDS 26.04
- RAPIDS cuDF and cuGraph installed

This will not run with cuGraph on local macOS because Apple Silicon GPUs do not support CUDA. Use a CUDA-capable Linux, WSL2, Brev, DGX Spark, or another NVIDIA GPU environment.

## Installation & Basic Functionality

Use the RAPIDS install selector for the exact command that matches the GPU machine:

https://docs.rapids.ai/install/

Example conda install for CUDA 13:

```bash
conda create -n rapids-cugraph -c rapidsai -c conda-forge -c nvidia \
    cugraph=26.04 python=3.13 cuda-version=13.1
conda activate rapids-cugraph
```

Example pip install:

```bash
python -m pip install cugraph-cu13 --extra-index-url=https://pypi.nvidia.com
# or, for CUDA 12:
python -m pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
```

Optional NetworkX backend:

```bash
python -m pip install nx-cugraph-cu13 --extra-index-url=https://pypi.nvidia.com
# or nx-cugraph-cu12 for CUDA 12
```

Verify the GPU and cuGraph install:

```bash
nvidia-smi
python examples/install_verification.py
```

Expected result: the script imports cuDF and cuGraph, detects GPU 0, creates a
small directed graph on the GPU, runs PageRank and breadth-first search, then
prints a `PASS` message.

Run the basic functionality demo:

```bash
python examples/basic_uses.py
```

This script starts with a cuDF edge list, builds cuGraph graph objects, and
demonstrates PageRank, BFS, and weakly connected components. The full setup and
runbook is in [`examples/SETUP_BASIC_USES.md`](./examples/SETUP_BASIC_USES.md).

## Relevant Use Case

**First Bowl of Soup: Financial fraud ring detection**

Financial services teams often need to find coordinated behavior across many accounts, devices, merchants, and transactions. A single transaction row may look normal, but the relationship graph can reveal shared devices, circular money movement, high-risk hubs, or tightly connected communities.

The workflow is:

1. Load transactions, accounts, devices, and merchant records with cuDF.
2. Build an edge list where vertices are entities and edges represent payments, logins, shared identifiers, or transfers.
3. Use cuGraph algorithms:
   - PageRank or HITS to find influential accounts.
   - Louvain or Leiden to find suspicious communities.
   - BFS or SSSP to trace paths from known bad actors.
   - Weakly or strongly connected components to group related entities.
   - Jaccard or overlap similarity to find accounts with similar neighborhoods.
4. Join graph-derived features back to tabular data.
5. Feed those features into cuML, a rules engine, or an analyst case-management workflow.

[`examples/relevant_uses.py`](./examples/relevant_uses.py) demonstrates this
workflow with a synthetic financial-services graph. Vertices represent
accounts, devices, merchants, IP addresses, and phone numbers. Edges represent
transactions and shared identifiers. The script uses cuGraph to find connected
components, trace the neighborhood around a known-bad account with BFS, rank
entities with PageRank, and produce a small investigation queue that could be
fed into analyst review or a fraud model.

Run it with:

```bash
python examples/relevant_uses.py
```

This connects to AIPS:

- **Accelerate:** run graph algorithms on large edge lists using NVIDIA GPUs.
- **Integrate:** keep the workflow in Python with cuDF, cuGraph, NetworkX, and cuML.
- **Promote:** show a clear customer story around relationship analytics.
- **Sell:** explain why GPU graph analytics matters when fraud patterns are hidden in connections.

## Files

- `examples/SETUP_BASIC_USES.md` - setup and runbook for all cuGraph examples.
- `examples/install_verification.py` - install smoke test for cuDF, cuGraph,
  CUDA, PageRank, and BFS.
- `examples/basic_uses.py` - small cuDF edge list demo covering PageRank, BFS,
  and connected components.
- `examples/relevant_uses.py` - First Bowl of Soup financial fraud ring
  workflow.

## Helpful Links

- [Official Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [Installation Guide](https://docs.rapids.ai/api/cugraph/stable/installation/getting_cugraph/)
- [RAPIDS Install Selector](https://docs.rapids.ai/install/)
- [Supported Algorithms](https://docs.rapids.ai/api/cugraph/stable/graph_support/algorithms/)
- [GitHub Repository](https://github.com/rapidsai/cugraph)
- [RAPIDS cuGraph Page](https://rapids.ai/cugraph/)
- [NVIDIA Developer Page](https://developer.nvidia.com/rapids)

## Contributor

Gayatri Kondabathini
