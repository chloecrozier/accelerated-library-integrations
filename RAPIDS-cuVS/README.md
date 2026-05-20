# RAPIDS cuVS

[cuVS](https://github.com/rapidsai/cuvs) is the RAPIDS library for GPU-accelerated vector search and clustering. It provides exact and approximate nearest-neighbor indexes that run on NVIDIA GPUs and can be used directly from Python, C, C++, and Rust.

- **vs CPU nearest neighbors:** cuVS moves high-dimensional search to the GPU, where large batches of similarity queries can run with much higher throughput.
- **vs vector databases:** cuVS is the accelerated search library underneath an application or database. It provides the indexing and query kernels, while a database adds persistence, metadata filtering, APIs, and operations.

## Purpose & Prerequisites

cuVS targets workflows where a dataset has already been converted into vectors and nearest-neighbor search is the bottleneck. Common examples include retrieval systems, recommendation, clustering, molecular similarity search, and k-NN graph construction.

Required:
- Linux, or Windows 11 via WSL2 with GPU passthrough
- NVIDIA GPU with a compatible driver and CUDA runtime
- Python 3.10+ for the Python examples
- Enough GPU memory to hold the embedding matrix and index

For the molecular examples, embedding generation is preprocessing. cuVS is the accelerated library demonstrated here.

## Installation & Basic Functionality

Choose the path that matches your CUDA version. The [cuVS install guide](https://docs.rapids.ai/api/cuvs/stable/build/) has the current compatibility details.

**pip** (CUDA 12 / 13 wheels)
```bash
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com
# or
pip install cuvs-cu13 --extra-index-url=https://pypi.nvidia.com
```

Install the molecular example dependencies:
```bash
pip install cupy-cuda12x torch transformers smirk pandas scikit-learn pyarrow
```

Use the CuPy wheel that matches your CUDA version; `cupy-cuda12x` is the CUDA 12 example.

**conda** (CUDA 12)
```bash
conda create -n rapids-cuvs -c rapidsai -c conda-forge \
    cuvs python=3.12 cuda-version=12.9
conda activate rapids-cuvs
conda install -c conda-forge pytorch transformers pandas scikit-learn pyarrow
pip install smirk
```

**Docker** - use RAPIDS images when you want the CUDA, driver, and Python stack pinned together.

**From source** - see the [cuVS build documentation](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source).

### Try It

The [`examples/`](./examples) folder is intended to be run in order:

1. [`README.md`](./examples/README.md) - setup steps and run commands.
2. [`install_verification.py`](./examples/install_verification.py) - lightweight cuVS smoke test with no model downloads.
3. [`basic_uses.ipynb`](./examples/basic_uses.ipynb) - toy molecular embeddings searched with cuVS brute force and IVF-Flat.
4. [`relevant_uses.py`](./examples/relevant_uses.py) - GuacaMol similarity-search benchmark on a 1M+ molecule corpus.

## Relevant Use Case

**Molecular analog search for cheminformatics**

Drug-discovery and materials-screening teams often start with a library of candidate molecules and ask: "Which known compounds are closest to this query molecule?" The end-to-end workflow is:

1. **Represent molecules** - generate one dense vector per SMILES string.
2. **Index vectors** - move the embedding matrix to GPU memory and build a cuVS index.
3. **Search** - run exact brute-force search for ground truth, or IVF-Flat for faster approximate search.
4. **Inspect hits** - return neighbor molecules and metadata such as source rows or assay IDs.

[`examples/relevant_uses.py`](./examples/relevant_uses.py) demonstrates this on [GuacaMol All SMILES](https://figshare.com/articles/dataset/GuacaMol_All_SMILES/7322252), a ChEMBL-derived molecular design corpus with over 1M molecules. It downloads the SMILES corpus, generates molecular embeddings for the requested subset, then compares:

- CPU exact cosine search
- cuVS brute-force cosine search
- cuVS IVF-Flat approximate cosine search

The script reports corpus size, timings, speedups, approximate-search recall@k against exact cuVS results, and sample nearest-neighbor hits. Use `--limit 0` to embed and index the full corpus when the machine has enough time and memory.

The default embedding model is [`mist-models/mist-28M-ti624ev1`](https://huggingface.co/mist-models/mist-28M-ti624ev1), used to produce input vectors for cuVS.

## Helpful Links

- [Official Documentation](https://docs.rapids.ai/api/cuvs/stable/)
- [Installation Guide](https://docs.rapids.ai/api/cuvs/stable/build/)
- [GitHub Repository](https://github.com/rapidsai/cuvs)
- [RAPIDS cuVS Page](https://rapids.ai/cuvs)
- [GuacaMol All SMILES Dataset](https://figshare.com/articles/dataset/GuacaMol_All_SMILES/7322252)
- [Default Embedding Model](https://huggingface.co/mist-models/mist-28M-ti624ev1)

## Contributor

TBD
