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

The presentation flow is:

1. **Hello QEC:** the Steane demo shows the smallest complete QEC loop.
2. **Realistic QEC:** the circuit-level surface-code memory sweep adds repeated
   syndrome rounds, circuit-level noise, and a GPU-capable decoder.
3. **Distance scaling:** the surface-code code-capacity sweep isolates direct
   data errors to make the effect of increasing code distance easier to see.
4. **Why GPUs matter:** the CPU/GPU syndrome benchmark and decoder
   time/accuracy benchmark show throughput and accuracy tradeoffs for repeated
   QEC work.

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
3. Run `examples/run_project.py`.
4. Use the generated CSVs, plots, and summary in [`results/`](./results).

Included examples:

- `examples/run_project.py` runs the standard project workflow with default
  Brev L4 settings, including the Steane demo, QLDPC surface-code sweeps, GPU
  throughput checks, decoder benchmark, plots, and summary.
- `examples/install_verification.py` checks imports, Steane code access, and a
  tiny `single_error_lut` decode.
- `examples/hello_syndrome.py` sweeps physical bit-flip probability for the
  Steane code and reports raw vs decoded logical error rates.
- `examples/surface_memory.py` contains the one-point surface-code memory
  helper used by the circuit-level sweep.
- `examples/surface_sweep.py` creates the QLDPC circuit-level surface-code
  memory sweep CSV and plot. It defaults to `rounds=d`, low physical error
  rates, 10,000 shots, and distances 3, 5, and 7.
- `examples/surface_code_capacity.py` creates the cleaner QLDPC surface-code
  code-capacity CSV and plot by applying direct data errors instead of full
  circuit-level noise.
- `examples/cpu_gpu_benchmark.py` compares CPU NumPy and GPU CuPy throughput
  for the same batched QEC syndrome calculation.
- `examples/decoder_benchmark.py` compares LUT and QLDPC BP=0 across distances
  3, 5, 7, 9, and 11 by default.
- `examples/plot_results.py` creates presentation plots and
  `results/SUMMARY.md` from CSV outputs.

## Relevant Use Case

**Logical error-rate estimation for quantum hardware and software teams**

A quantum hardware company, quantum software platform, HPC center, or research
lab may want to evaluate whether a code and decoder combination can protect a
logical qubit well enough for future workloads. CUDA-Q QEC fits into that
workflow by connecting noise modeling, syndrome sampling, decoding, and logical
error-rate measurement.

This connects to AIPS:

- **Accelerate:** compare CPU and GPU-style throughput plus QEC decoder
  time/accuracy tradeoffs on Brev L4.
- **Integrate:** add QEC experiments to an existing CUDA-Q Python or C++
  workflow.
- **Promote:** show logical error-rate plots and decoder-throughput results
  from Brev L4.
- **Sell:** explain why NVIDIA GPUs matter before large-scale fault-tolerant
  QPUs are widely available.

The two surface-code sweeps answer different questions. The circuit-level
memory sweep is the more realistic integration demo. The code-capacity sweep is
the cleaner distance-scaling accuracy demo. Neither should be presented as a
perfect reproduction of published threshold plots because decoder choice, noise
model, shot count, and the CUDA-Q QEC version all affect the trend.

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
