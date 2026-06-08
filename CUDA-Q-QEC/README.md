# CUDA-Q QEC

[CUDA-Q QEC](https://nvidia.github.io/cudaqx/components/qec/introduction.html)
is the quantum error-correction library in CUDA-QX. It provides Python and C++
tools for error-correcting codes, stabilizer measurements, detector error
models, and syndrome decoders.

This project uses CUDA-Q QEC for one workflow: estimating logical error rates.

1. Choose a code.
2. Add physical errors or circuit-level noise.
3. Measure or compute a syndrome.
4. Decode the syndrome.
5. Compare logical errors before and after decoding.

## Purpose & Prerequisites

Physical qubits are noisy. Quantum error correction protects a logical qubit by
encoding it across multiple physical qubits, measuring stabilizers, and using a
classical decoder to infer a correction.

CUDA-Q QEC is useful for teams that want to:

- test QEC codes before fault-tolerant hardware is available
- compare decoders and noise assumptions
- estimate logical error rates
- study GPU-accelerated syndrome decoding

This project is intended to run on a Brev L4 instance. The required packages
are listed in [`requirements.txt`](./requirements.txt).

## Installation & Basic Functionality

Use [`examples/SETUP.md`](./examples/SETUP.md) for setup and run commands. The
basic flow is:

1. Check the GPU with `nvidia-smi`.
2. Install the Python requirements.
3. Run `examples/install_verification.py`.
4. Run the demos and save outputs in [`results/`](./results).

Included examples:

- `examples/install_verification.py` checks imports, Steane code access, and a
  tiny `single_error_lut` decode.
- `examples/hello_syndrome.py` sweeps physical bit-flip probability for the
  Steane code and reports raw vs decoded logical error rates.
- `examples/surface_memory.py` runs one surface-code memory experiment with
  circuit-level noise.
- `examples/surface_sweep.py` creates surface-code logical error-rate plots.
- `examples/decoder_benchmark.py` measures decoder throughput on one fixed
  surface-code workload.
- `examples/plot_results.py` creates summary plots from CSV outputs.

## Relevant Use Case

**Logical error-rate estimation for quantum hardware and software teams**

A quantum hardware company, quantum software platform, HPC center, or research
lab may want to evaluate whether a code and decoder combination can protect a
logical qubit well enough for future workloads. CUDA-Q QEC fits into that
workflow by connecting noise modeling, syndrome sampling, decoding, and logical
error-rate measurement.

This connects to AIPS:

- **Accelerate:** benchmark GPU-capable QEC decoders for high-throughput
  syndrome decoding.
- **Integrate:** add QEC experiments to an existing CUDA-Q Python or C++
  workflow.
- **Promote:** show logical error-rate plots and decoder-throughput results
  from Brev L4.
- **Sell:** explain why NVIDIA GPUs matter before large-scale fault-tolerant
  QPUs are widely available.

## Helpful Links

CUDA-Q QEC:

- [CUDA-Q QEC Documentation](https://nvidia.github.io/cudaqx/components/qec/introduction.html)
- [CUDA-QX Installation Guide](https://nvidia.github.io/cudaqx/quickstart/installation.html)
- [CUDA-Q QEC Python API](https://nvidia.github.io/cudaqx/api/qec/python_api.html)
- [Code-Capacity QEC Example](https://nvidia.github.io/cudaqx/examples_rst/qec/code_capacity_noise.html)
- [Circuit-Level QEC Example](https://nvidia.github.io/cudaqx/examples_rst/qec/circuit_level_noise.html)
- [CUDA-Q QEC 0.6 Real-Time Decoding Blog](https://nvidia.github.io/cuda-quantum/blogs/blog/2026/04/14/cudaq-qec-0.6/)
- [CUDA-QX GitHub Repository](https://github.com/NVIDIA/cudaqx)

QEC background:

- [IBM Quantum Learning: Foundations of Quantum Error Correction](https://quantum.cloud.ibm.com/learning/courses/foundations-of-quantum-error-correction)
- [Error Correction Zoo: Steane Code](https://errorcorrectionzoo.org/c/steane)
- [Error Correction Zoo: Surface Code](https://errorcorrectionzoo.org/c/surface)

## Contributor

Adrian Mittal
