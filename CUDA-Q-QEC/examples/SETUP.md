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

Optional cleanup before regenerating artifacts:

```bash
find results -type f ! -name "ARTIFACTS.md" -delete
```

```bash
python examples/run_project.py
```

This runs:

- install verification
- Steane code-capacity demo for the basic QEC loop
- CPU vs GPU syndrome-throughput benchmark
- QLDPC circuit-level surface-code memory sweep with `rounds=d`
- LUT and BP decoder benchmark for distances 3, 5, 7, 9, and 11
- result plotting and summary generation

The main outputs are written to `results/`.

Expected presentation outputs:

- `steane_logical_error_rates.png`
- `cpu_gpu_speedup.png`
- `surface_sweep_qldpc_brev_l4.png`
- `decoder_time_accuracy.png`
- `SUMMARY.md`

If the QLDPC decoder is unavailable in the current environment, run:

```bash
python examples/run_project.py --skip-qldpc
```

If CuPy is unavailable and you only want the QEC demos, run:

```bash
python examples/run_project.py --skip-cpu-gpu
```

## 3. Surface Sweep

The circuit-level sweep uses distances 3, 5, and 7; `rounds=d`; lower physical
error rates; and 10,000 shots. Its plot shows raw logical error rates and
decoded logical error rates, similar to the Steane logical error-rate graph.

To rerun the surface sweep manually:

```bash
python examples/surface_sweep.py
```

To include distances 9 and 11 in the standard workflow:

```bash
python examples/run_project.py --full-surface-sweep
```

Use more shots when you need smoother low-error-rate curves:

```bash
python examples/surface_sweep.py --shots 100000
```

The surface sweep is more realistic than a direct data-error experiment, but it
may still show larger distances getting worse if extra noisy circuit locations
dominate.

## 4. Individual Commands

These are useful if you only want one artifact:

```bash
python examples/install_verification.py
python examples/hello_syndrome.py
python examples/cpu_gpu_benchmark.py
python examples/surface_sweep.py
python examples/decoder_benchmark.py
python examples/plot_results.py
```

The decoder benchmark defaults to 2,000 shots per distance. It compares
`single_error_lut` and QLDPC `BP=0` over distances 3, 5, 7, 9, and 11.
Use `--distance 7` only if you want one focused decoder comparison.
