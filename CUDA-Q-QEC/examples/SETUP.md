# CUDA-Q QEC Setup

Run these commands from the repo root on a Brev L4 instance.

## 1. Environment

```bash
nvidia-smi
cd CUDA-Q-QEC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If pip installation fails because of CUDA-QX compatibility, use the official
CUDA-QX container from the installation guide.

## 2. Standard Run

```bash
python examples/run_project.py
```

This runs:

- install verification
- Steane code-capacity demo for the basic QEC loop
- CPU vs GPU syndrome-throughput benchmark
- QLDPC circuit-level surface-code memory sweep with `rounds=d`
- QLDPC surface-code code-capacity sweep
- LUT and BP decoder benchmark for distances 3, 5, 7, 9, and 11
- result plotting and summary generation

The main outputs are written to `results/`.

If the QLDPC decoder is unavailable in the current environment, run:

```bash
python examples/run_project.py --skip-qldpc
```

If CuPy is unavailable and you only want the QEC demos, run:

```bash
python examples/run_project.py --skip-cpu-gpu
```

If the code-capacity sweep takes too long, run:

```bash
python examples/run_project.py --skip-code-capacity
```

## 3. Surface Sweeps

The standard run includes two QLDPC surface-code sweeps:

- `surface_sweep.py`: circuit-level memory sweep for the realistic CUDA-Q QEC
  integration demo.
- `surface_code_capacity.py`: code-capacity sweep for the cleaner distance
  scaling accuracy graph.

The circuit-level sweep uses distances 3, 5, and 7; `rounds=d`; lower physical
error rates; and 10,000 shots.

To rerun the circuit-level surface sweep manually:

```bash
python examples/surface_sweep.py
```

To rerun the code-capacity surface sweep manually:

```bash
python examples/surface_code_capacity.py
```

To include distances 9 and 11 in the standard workflow:

```bash
python examples/run_project.py --full-surface-sweep
```

Use more shots when you need smoother low-error-rate curves:

```bash
python examples/surface_sweep.py --shots 100000
python examples/surface_code_capacity.py --shots 200000
```

The code-capacity sweep has a better chance of showing the expected distance
trend because it applies data errors directly. The circuit-level sweep is more
realistic but may still show larger distances getting worse if extra noisy
circuit locations dominate.

## 4. Individual Commands

These are useful if you only want one artifact:

```bash
python examples/install_verification.py
python examples/hello_syndrome.py
python examples/cpu_gpu_benchmark.py
python examples/surface_sweep.py
python examples/surface_code_capacity.py
python examples/decoder_benchmark.py
python examples/plot_results.py
```

The decoder benchmark defaults to 2,000 shots per distance. It compares
`single_error_lut` and QLDPC `BP=0` over distances 3, 5, 7, 9, and 11.
Use `--distance 7` only if you want one focused decoder comparison.
