# Setup & Run

Step-by-step walkthrough for setting up cuGraph on an NVIDIA GPU and running
the examples in this folder. These scripts are meant for Linux, WSL2, Brev,
DGX Spark, or another CUDA-capable NVIDIA GPU environment.

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
cd accelerated-library-integrations/RAPIDS-cuGraph
```

## 3. Create or activate a RAPIDS environment

If the machine already has a RAPIDS environment with cuDF and cuGraph, activate
it and skip to the verification step.

For a fresh environment, use the RAPIDS install selector for the exact command
that matches your CUDA and Python versions:

https://docs.rapids.ai/install/

Example conda environment for RAPIDS 26.04 and CUDA 13:

```bash
conda create -n rapids-cugraph -c rapidsai -c conda-forge -c nvidia \
    cugraph=26.04 python=3.13 cuda-version=13.1
conda activate rapids-cugraph
```

Example pip install for CUDA 13:

```bash
python -m pip install cugraph-cu13 --extra-index-url=https://pypi.nvidia.com
```

Use `cugraph-cu12` instead if the environment is on CUDA 12.

## 4. Verify the cuGraph install

```bash
python examples/install_verification.py
```

Expected result: versions print, GPU 0 is detected, PageRank and BFS run on a
small graph, and the script ends with a `PASS` message.

## 5. Run the basic cuGraph demo

```bash
python examples/basic_uses.py
```

This creates a small cuDF edge list, builds cuGraph graph objects, then runs
PageRank, breadth-first search, and connected components.

## 6. Run the First Bowl of Soup demo

```bash
python examples/relevant_uses.py
```

This demonstrates a financial fraud ring workflow. It creates a synthetic
entity graph of accounts, devices, merchants, IP addresses, and phone numbers;
then it uses cuGraph to find connected components, trace the neighborhood
around a known-bad account, rank influential entities, and produce a small
investigation queue.

For a slightly larger synthetic graph:

```bash
python examples/relevant_uses.py --benign-components 100
```

The script does not download external data, so it is safe to run as a quick
demo after install verification.
