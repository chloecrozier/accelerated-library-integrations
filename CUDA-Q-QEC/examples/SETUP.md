# CUDA-Q QEC Setup

Run these commands from the repo root on a Brev L4 instance. The scripts use
Brev L4 output filenames by default.

## 1. Check The GPU

```bash
nvidia-smi
```

This should show the NVIDIA GPU, driver, and CUDA version visible to the host.

## 2. Create The Environment

```bash
cd CUDA-Q-QEC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If pip installation fails because of CUDA-QX compatibility, use the CUDA-QX
container instead:

```bash
docker pull ghcr.io/nvidia/cudaqx
docker run --gpus all -it \
  -v "$PWD:/workspace/accelerated-library-integrations" \
  ghcr.io/nvidia/cudaqx
cd /workspace/accelerated-library-integrations/CUDA-Q-QEC
python -m pip install -r requirements.txt
```

## 3. Verify The Install

```bash
python examples/install_verification.py
```

Expected result: the script imports CUDA-Q QEC, loads the Steane code, decodes
an all-zero syndrome, and prints `PASS`.

## 4. Run The Demos

Run the examples:

```bash
python examples/hello_syndrome.py --shots 1000

python examples/surface_memory.py \
  --shots 1000 \
  --distance 3

python examples/decoder_benchmark.py \
  --decoder single_error_lut \
  --shots 10000

python examples/decoder_benchmark.py \
  --decoder nv-qldpc-decoder \
  --shots 10000

python examples/decoder_benchmark.py \
  --decoder nv-qldpc-decoder \
  --distance 7 \
  --shots 10000 \
  --bp-batch-size 10000
```

These commands write Brev L4-labeled CSV files in `results/`.

If `nv-qldpc-decoder` is unavailable, use `single_error_lut` to verify the
workflow, then switch to the CUDA-QX container for the final benchmark.
The QLDPC comparison is an accuracy-throughput tradeoff: LUT is a tiny teaching
decoder, while QLDPC is the GPU-capable belief-propagation decoder for larger
workloads.

## 5. Make Surface-Code Sweep Plots

Small teaching graph with the lookup-table decoder:

```bash
python examples/surface_sweep.py \
  --decoder single_error_lut \
  --distances 3 5 \
  --shots 1000
```

Larger GPU decoder graph after the small run works:

```bash
python examples/surface_sweep.py \
  --decoder nv-qldpc-decoder \
  --distances 3 5 7 \
  --shots 100000 \
  --batch-size 10000
```

Use more shots when you need to estimate smaller logical error rates.

## 6. Generate Summary Plots

```bash
python examples/plot_results.py --results-dir results --output-dir results
```

This creates summary plots from captured Steane and decoder benchmark CSV files.
