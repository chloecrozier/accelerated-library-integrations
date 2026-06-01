# cuBLAS Project

## Purpose & Prerequisites

**cuBLAS** is NVIDIA’s GPU-accelerated implementation of the BLAS (Basic Linear Algebra Subprograms) library.

It is designed to accelerate dense linear algebra operations on NVIDIA GPUs, including:

- Vector and matrix operations
- Matrix multiplication (GEMM)
- Linear algebra building blocks used by ML, AI, and HPC applications

Prerequisites:

- NVIDIA GPU with CUDA support
- CUDA Toolkit or `cudatoolkit` installed
- Python 3.8+
- `numpy`
- `numba`

---

## Installation & Basic Functionality

### 1. Create a Conda Environment

```bash
conda create -n cublas_env python=3.11 -y
conda activate cublas_env
```

### 2. Install Python Dependencies and CUDA Runtime

```bash
pip install numpy numba
conda install -c conda-forge cudatoolkit=11.8.0 -y
```

> Note: If you already have a system CUDA installation, you may skip the `cudatoolkit` install step.

### 3. Verify CUDA Availability

```bash
nvidia-smi
nvcc --version
```

### 4. Run the Verification Script

The `examples/verify_cuBLAS.py` script confirms that `libcublas.so` can be loaded and that a cuBLAS handle can be created.

```bash
conda activate cublas_env
python examples/verify_cuBLAS.py
```

### 5. Run the cuBLAS Example Demos

The `examples/verify_cuBLAS.py` script confirms that the cuBLAS library can be loaded and initialized.

```bash
conda activate cublas_env
python examples/verify_cuBLAS.py
```

The `examples/cuBLAS_single_benchmark.py` script demonstrates a single GPU-accelerated matrix multiplication workflow and compares it to a CPU matmul.

```bash
conda activate cublas_env
python examples/cuBLAS_single_benchmark.py
```

The `examples/cuBLAS_multi_benchmark.py` script demonstrates a repeated benchmark of GPU cuBLAS GEMM performance versus CPU performance.

```bash
conda activate cublas_env
python examples/cuBLAS_multi_benchmark.py
```

---

## Screenshots

Screenshots of the installation and benchmark results are available in `Screenshots/`.

---

## Relevant Use Case — HPC / Scientific Simulation

### First Bowl of Soup: GPU-accelerated dense linear algebra for scientific simulation

In scientific simulation workflows, large systems of equations and dense matrix operations are common. cuBLAS is often used as the computational engine for:

1. Preprocessing simulation data into dense matrices
2. Performing high-throughput matrix-matrix multiplications and linear algebra kernels on the GPU
3. Feeding accelerated results into a simulation loop or machine learning model

Example workflow:

- **Data prep:** generate or load matrices that represent system states, physical coefficients, or neural network weights.
- **Compute:** use cuBLAS GEMM operations (`cublasSgemm` / `cublasDgemm`) to accelerate dense linear algebra on NVIDIA GPUs.
- **Postprocess:** transfer results back to the host for analysis, visualization, or integration into a larger HPC pipeline.

This workflow is relevant for:

- Computational fluid dynamics
- Structural engineering simulation
- Finite element analysis
- Deep learning training loops where dense linear algebra is a bottleneck

---

## Helpful Links

- NVIDIA cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- CUDA Python: https://nvidia.github.io/cuda-python/
- NVIDIA Developer Portal: https://developer.nvidia.com/

---

## Contributor

Harika Mekala