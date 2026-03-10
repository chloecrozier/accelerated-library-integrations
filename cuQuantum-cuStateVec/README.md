# cuStateVec (cuQuantum SDK)

## Purpose

cuStateVec is the state-vector simulation library within the NVIDIA cuQuantum SDK. It provides GPU-accelerated primitives for quantum circuit simulation -- applying gate matrices, measuring qubits, sampling bit strings, and computing expectation values.

## Basic Use Cases

- Simulating quantum circuits with full state-vector representation on a single GPU
- Preparing and measuring entangled states (e.g., Bell states, GHZ states)
- Benchmarking qubit capacity across different GPU memory sizes

## Documentation Gap

Running cuStateVec on an NVIDIA DGX Spark -- no existing tutorials cover the Spark's unified memory architecture or its desktop-scale thermal constraints.

## Advanced Use Case

cuStateVec on a DGX Spark: exploring how the Spark's 128 GB unified memory pool affects maximum qubit count compared to discrete data-center GPUs.

## Helpful Links

- [cuStateVec Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html)
- [cuStateVec Python Bindings](https://docs.nvidia.com/cuda/cuquantum/latest/python/bindings/custatevec.html)
- [cuQuantum GitHub Repository](https://github.com/NVIDIA/cuQuantum)
- [cuQuantum Python on PyPI](https://pypi.org/project/cuquantum-python/)
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

## Contributor

@chloecrozier