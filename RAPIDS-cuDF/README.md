# RAPIDS cuDF

[cuDF](https://github.com/rapidsai/cudf) is the GPU DataFrame library in the RAPIDS SDK. It exposes a pandas-style API backed by Apache Arrow's columnar memory format and executes operations on NVIDIA GPUs.

- **vs pandas:** pandas-compatible API. With `cudf.pandas`, existing pandas code runs on the GPU with no source changes; unsupported operations fall back to CPU.
- **vs Dask-cuDF:** cuDF is single-GPU and eager. Dask-cuDF wraps cuDF to partition workloads across multiple GPUs or nodes.

## Purpose & Prerequisites

cuDF targets tabular data workflows where pandas is the throughput bottleneck and an NVIDIA GPU is available.

Required:
- Linux with `glibc >= 2.28` (Ubuntu 20.04+) or Windows 11 via WSL2
- CUDA 12 with driver `>= 525.60.13`, or CUDA 13 with driver `>= 580.65.06`
- NVIDIA GPU, compute capability 7.0+ (Volta or newer)

Recommended: NVMe SSD, ~2:1 system RAM to total VRAM, NVLink for 2+ GPU configurations.

## Installation & Basic Functionality

Choose the path that matches your environment. The [RAPIDS install guide](https://docs.rapids.ai/install/#system-req) has the full compatibility matrix.

**pip** (CUDA 12 / 13 wheels)
```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12  # or cudf-cu13
```

**conda** (CUDA 12)
```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
    cudf=25.10 python=3.12 cuda-version=12.8 && conda activate rapids
```

**Docker** — prebuilt [RAPIDS images](https://docs.rapids.ai/install/#docker).

**From source** — [cuDF build setup](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#setting-up-your-build-environment).

### Try it

The [`examples/`](./examples) folder is intended to be run in order:

1. [`SETUP.md`](./examples/SETUP.md) — conda env and instructions for running the remaining files.
2. [`install_verification.py`](./examples/install_verification.py) — end-to-end smoke test covering driver, CUDA runtime, and GPU compute. Run first to confirm a healthy install.
3. [`basic_uses.ipynb`](./examples/basic_uses.ipynb) — DataFrame operations, joins, pandas interop, `cudf.pandas`, and Dask-cuDF.

## Relevant Use Case

**GPU-native data analytics — IBM × NVIDIA (Velox + cuDF)**

[At GTC 2026, IBM and NVIDIA announced an integration of cuDF with the Velox query execution engine](https://newsroom.ibm.com/2026-03-16-ibm-and-nvidia-announce-expanded-collaboration-at-gtc-2026-to-advance-ai-for-the-enterprise), enabling GPU-native query execution in platforms such as Presto and Apache Spark. The collaboration reports up to 30x price-performance in GPU-accelerated analytics.

[Reported benchmarks](https://developer.nvidia.com/blog/accelerating-large-scale-data-analytics-with-gpu-native-velox-and-nvidia-cudf/):
- Presto on GPU: TPC-H SF1000 in 99.9s on a GH200 Grace Hopper Superchip vs. 1,246s on an AMD 5965X CPU.
- Spark in hybrid mode: compute-heavy stages such as TPC-DS Q95 (SF100) execute on GPU while remaining stages run on CPU.

The integration requires no changes to existing Presto or Spark queries; cuDF executes beneath the query engine via Velox. [`examples/relevant_uses.py`](./examples/relevant_uses.py) demonstrates the same pattern at small scale by downloading public NYC TLC taxi data and timing an identical load → filter → groupby → sort pipeline across pandas, cuDF, and Dask-cuDF.

### Notes

- **GB10 / Grace Blackwell** is an SoC with unified memory between the ARM CPU and Blackwell GPU (no PCIe transfers between them).
- **Dask-cuDF on a single GPU** adds scheduler overhead vs plain cuDF. It pays off when scaling across multiple GPUs / nodes (e.g. a DGX Spark cluster of GB10s).

## Helpful Links

- [Official Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [Installation Guide](https://docs.rapids.ai/install/#system-req)
- [GitHub Repository](https://github.com/rapidsai/cudf)
- [NVIDIA Developer Page](https://developer.nvidia.com/rapids)

## Contributor

Chloe Crozier
