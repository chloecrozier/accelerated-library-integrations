# Math-cuBLAS Examples

This folder contains scripts to verify cuBLAS installation and benchmark GPU-accelerated matrix multiplication.

## Scripts

- `verify_cuBLAS.py`
  - Loads `libcublas.so` and initializes a cuBLAS handle.
  - Useful to confirm that the CUDA runtime and cuBLAS library are available.

- `cuBLAS_single_benchmark.py`
  - Runs a single CPU `np.dot` matrix multiplication and a single GPU cuBLAS `cublasSgemm` operation.
  - Prints the runtime for CPU and GPU and compares speedup.

- `cuBLAS_multi_benchmark.py`
  - Runs repeated CPU and GPU matrix multiply benchmarks.
  - Reports average, min, and max timing for CPU and GPU runs.

## How to run

```bash
conda activate cublas_env
python examples/verify_cuBLAS.py
python examples/cuBLAS_single_benchmark.pygit add Math-cuBLAS
python examples/cuBLAS_multi_benchmark.py
```
