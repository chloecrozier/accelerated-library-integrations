# Setup & Run

Step-by-step walkthrough for setting up cuVS on an NVIDIA GPU and running the molecular search examples in this folder.

## 1. Verify the GPU

```bash
nvidia-smi
```

You should see an NVIDIA GPU and a driver new enough for the CUDA version you plan to use.

## 2. Clone the repo

```bash
gh auth login
git clone <repo-url>
cd accelerated-library-integrations/RAPIDS-cuVS/examples
```

## 3. Create the environment

Using conda:

```bash
conda create -n rapids-cuvs -c rapidsai -c conda-forge \
    cuvs python=3.12 cuda-version=12.9
conda activate rapids-cuvs
conda install -c conda-forge pytorch transformers pandas scikit-learn pyarrow jupyterlab
pip install smirk
```

Using pip with CUDA 12 wheels:

```bash
python -m venv .venv
source .venv/bin/activate
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cupy-cuda12x torch transformers smirk pandas scikit-learn pyarrow jupyterlab
```

`smirk` can require Rust to build on some systems. Install Rust from https://www.rust-lang.org/tools/install if pip cannot find a wheel for your platform.

## 4. Verify the cuVS install

```bash
python install_verification.py
```

This test does not download embedding models. It only checks that cuVS, CuPy, CUDA, and a small nearest-neighbor search work.

## 5. Run the basic-use notebook

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 basic_uses.ipynb
```

On your local machine, forward the port if you are working on a remote GPU host:

```bash
ssh -N -L 8888:localhost:8888 <username>@<remote-host>
```

Then open `http://localhost:8888/` locally and paste the token from Jupyter.

## 6. Run the molecular search benchmark

First run a small smoke test. This downloads the 1M+ molecule GuacaMol corpus and the embedding model if they are not already cached, but only embeds the requested subset:

```bash
python relevant_uses.py --limit 256 --queries 16 --batch-size 16 --k 5
```

Then run the default benchmark, which embeds the first 50,000 corpus rows:

```bash
python relevant_uses.py
```

For a heavier run:

```bash
python relevant_uses.py --limit 100000 --queries 512 --batch-size 64 --k 10
```

To embed and index the full GuacaMol corpus, set `--limit 0`. This is intentionally much heavier because it processes more than a million molecules:

```bash
python relevant_uses.py --limit 0 --queries 1024 --batch-size 128 --k 10
```

The script prints dataset statistics, embedding time, CPU exact search time, cuVS exact search time, IVF-Flat build/search time, recall@k, and example molecular neighbors.
