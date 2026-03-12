# cuDF (RAPIDS)

## Purpose

cuDF is a GPU-accelerated DataFrame library in the RAPIDS ecosystem. It provides a pandas-like API for loading, filtering, joining, and aggregating tabular data entirely on the GPU -- often achieving significant speedups over CPU-based pandas with minimal code changes.

## Basic Use Cases

- Loading and exploring CSV/Parquet datasets on the GPU
- Filtering, grouping, and aggregating large tabular data
- Drop-in acceleration of existing pandas workflows via `cudf.pandas`

See [`examples/basic_cudf_demo.py`](examples/basic_cudf_demo.py) for a walkthrough of core operations (filtering, group-by, joins, sorting) on a 1M-row synthetic dataset.

## Documentation Gap

NVIDIA publishes a single-node [CUDA-X Data Science playbook](https://build.nvidia.com/spark/cuda-x-data-science) for DGX Spark, but it only covers an 8 GB strings dataset with `cudf.pandas`. There is no guidance on pushing cuDF further on the Spark or on cloud GPU instances like Brev.

## Advanced Use Case

### 1. Large-dataset ETL on a single DGX Spark

The Spark's 128 GB unified memory pool lets cuDF hold DataFrames far larger than what fits on a typical 24-48 GB discrete GPU. Test cuDF with a 50-100 GB dataset (e.g., NYC taxi trip data) and benchmark read, transform, and write throughput compared to pandas on the same hardware and to cuDF on a cloud GPU with separate host/device memory.

### 2. Distributed cuDF across stacked Sparks with Dask

Two DGX Sparks connected via 200 Gb/s RoCE form a 256 GB GPU cluster. [`examples/distributed_stacked_sparks.py`](examples/distributed_stacked_sparks.py) demonstrates this with a 50M-row dataset partitioned across both nodes, covering:

- **Distributed group-by with shuffle** -- data for each key must be co-located on the same worker before aggregation; Dask hash-partitions and transfers across the RoCE link.
- **Broadcast join** -- a small lookup table is replicated to every worker so the large table stays in place. This is a coscheduling-friendly pattern: avoid shuffling the big side by broadcasting the small side.
- **Shuffle join (large x large)** -- both sides are large, so Dask must hash-partition and redistribute across workers. The 200 Gb/s interconnect between Sparks is what makes this practical at scale.
- **RMM memory pools** -- pre-allocating GPU memory with `--rmm-pool-size` avoids millions of small CUDA allocations that degrade performance.

See the script's docstring for full setup instructions (scheduler, workers, networking).

## Helpful Links

- [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)
- [cuDF GitHub Repository](https://github.com/rapidsai/cudf)
- [RAPIDS Installation Guide](https://docs.rapids.ai/install)
- [cudf.pandas Accelerator](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/)
- [Dask-CUDA Best Practices](https://docs.rapids.ai/api/dask-cuda/stable/examples/best-practices/)
- [DGX Spark Stacking Guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [CUDA-X Data Science Playbook (DGX Spark)](https://build.nvidia.com/spark/cuda-x-data-science)
- [RAPIDS on NVIDIA Developer](https://developer.nvidia.com/rapids)

## Contributor

@chloecrozier