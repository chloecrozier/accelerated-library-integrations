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
- surface-code memory demo for the realistic QEC workflow
- CPU vs GPU syndrome-throughput benchmark
- LUT and QLDPC surface-code sweeps with `rounds=d`
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

## 3. Surface Sweeps

The standard run already includes both LUT and QLDPC surface sweeps. The default
sweep uses distances 3, 5, and 7; `rounds=d`; lower physical error rates; and
10,000 shots. This gives the graph a better chance of showing the expected
low-error QEC trend.

To rerun the LUT sweep manually:

```bash
python examples/surface_sweep.py
```

To rerun the same sweep with the GPU-capable QLDPC decoder:

```bash
python examples/surface_sweep.py \
  --decoder nv-qldpc-decoder
```

To include distances 9 and 11 in the standard workflow:

```bash
python examples/run_project.py --full-surface-sweep
```

Use more shots when you need smoother low-error-rate curves:

```bash
python examples/surface_sweep.py \
  --decoder nv-qldpc-decoder \
  --shots 100000
```

The surface sweep is still not guaranteed to match published threshold plots.
Decoder choice, noise model, shot count, and CUDA-Q QEC version can all change
whether larger distance visibly lowers the logical error rate.

## 4. Individual Commands

These are useful if you only want one artifact:

```bash
python examples/install_verification.py
python examples/hello_syndrome.py
python examples/surface_memory.py
python examples/cpu_gpu_benchmark.py
python examples/decoder_benchmark.py
python examples/plot_results.py
```

The decoder benchmark defaults to 2,000 shots per distance. It compares
`single_error_lut` and QLDPC `BP=0` over distances 3, 5, 7, 9, and 11.
Use `--distance 7` only if you want one focused decoder comparison.
