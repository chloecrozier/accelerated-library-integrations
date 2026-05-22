# Setup & Run

Runbook for using the cuML examples on a Brev GPU instance or DGX Spark system.
Do not commit internal hostnames, IP addresses, passwords, API keys, or tokens
to this repository.

## 1. Verify the GPU

```bash
nvidia-smi
```

Confirm that a CUDA-capable NVIDIA GPU is visible before installing or running
RAPIDS packages.

## 2. Clone the repo

```bash
gh auth login
git clone <repo-url>
cd accelerated-library-integrations/RAPIDS-cuML
```

## 3. Create or activate a RAPIDS environment

If the machine already has a RAPIDS environment, activate it and skip to the
verification step.

For a fresh environment, use the RAPIDS install selector for the exact command
that matches your CUDA and Python versions:

https://docs.rapids.ai/install/

Example conda environment for RAPIDS 26.04:

```bash
conda create -n rapids-cuml -c rapidsai -c conda-forge -c nvidia \
    cudf=26.04 cuml=26.04 python=3.13 cuda-version=12.9
conda activate rapids-cuml
```

Install JupyterLab if you want notebook access:

```bash
conda install -c conda-forge jupyterlab
```

## 4. Verify the install

```bash
python examples/install_verification.py
```

Expected result: versions print, GPU 0 is detected, and the script ends with a
`PASS` message.

## 5. Run the wine quality workflow

```bash
python examples/wine_quality_cuml.py
```

This loads the bundled wine quality CSV with cuDF, separates features from the
known `quality` label, trains one cuML Random Forest classifier, and prints
test-set accuracy.

## 6. Optional JupyterLab over SSH

On the remote machine:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

On your local machine:

```bash
ssh -N -L 8888:localhost:8888 <username>@<remote-host>
```

Then open `http://localhost:8888/` locally and paste the token from the remote
Jupyter output.
