# Running these examples

Set up a conda env with cuDF, then run the files in this folder.

## 1. Create + activate the env

```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
    cudf=25.10 python=3.12 cuda-version=12.8
conda activate rapids
```

## 2. Run the files

```bash
# sanity check the install
python install_verification.py

# notebook
conda install -c conda-forge jupyterlab
jupyter lab basic_uses.ipynb

# pandas vs cuDF benchmark on real data
python relevant_uses.py

```