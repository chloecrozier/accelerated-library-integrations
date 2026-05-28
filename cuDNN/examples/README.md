# Setup & Run

Runbook for using the cuDNN examples on a Brev GPU instance or DGX Spark system.
Do not commit internal hostnames, IP addresses, passwords, API keys, tokens, or
screenshots containing private infrastructure details.

## 1. Verify the GPU

```bash
nvidia-smi
```

Confirm that a CUDA-capable NVIDIA GPU is visible before running the examples.

## 2. Clone the repo

```bash
git clone <repo-url>
cd accelerated-library-integrations/cuDNN
```

## 3. Create or activate a PyTorch environment

From the `cuDNN/` folder, create a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements.txt
```

This installs the CUDA 12.8 PyTorch wheels plus the small notebook/charting
dependencies. If the machine needs a different CUDA wheel, use the PyTorch
install selector to choose the command that matches the target machine's CUDA
runtime:

https://pytorch.org/get-started/locally/

If the machine already has a CUDA-enabled PyTorch environment, activate it and
skip to the verification step.

## 4. Verify cuDNN through PyTorch

```bash
python3 examples/install_verification.py
```

Expected result: PyTorch imports, CUDA is available, cuDNN is available, one
small GPU convolution runs, and the script ends with a `PASS` message.

## 5. Run the benchmark

On Brev L4:

```bash
python3 examples/benchmark_cudnn.py --platform brev_l4 --output results/brev_l4.csv
```

On DGX Spark:

```bash
python3 examples/benchmark_cudnn.py --platform dgx_spark --output results/dgx_spark.csv
```

Use fewer iterations for a quick smoke test:

```bash
python3 examples/benchmark_cudnn.py --platform quick_test --iters 10 --output results/quick_test.csv
```

The benchmark prints a table and writes CSV data with latency, throughput, and
speedup versus the CPU baseline.

## 6. Run the notebook

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 examples/cudnn_resnet18_benchmark.ipynb
```

On your local machine, forward the port if the notebook is running remotely:

```bash
ssh -N -L 8888:localhost:8888 <username>@<remote-host>
```

Then open `http://localhost:8888/` locally and paste the token from the remote
Jupyter output.

## 7. Optional Nsight Systems capture

The PyTorch benchmark is the required path. If Nsight Systems is available, use
the single-inference profiler target to produce a clean visual comparison:

```bash
nsys profile --force-overwrite=true --trace=cuda,cudnn,nvtx \
    --sample=none --cpuctxsw=none \
    -o screenshots/brev_l4_single_inference \
    python3 examples/profile_inference_for_nsight.py \
        --platform brev_l4 --workload resnet18 --warmup 1 --runs 1 \
        --output results/brev_l4_single_inference.csv
```

Open the `.nsys-rep` in Nsight Systems and look for the labeled NVTX ranges:

- `MEASURE | resnet18 | 01 CPU | inference 1`
- `MEASURE | resnet18 | 02 GPU, cuDNN disabled | inference 1`
- `MEASURE | resnet18 | 03 GPU, cuDNN enabled | inference 1`

Save screenshots of the timeline under `screenshots/`.
