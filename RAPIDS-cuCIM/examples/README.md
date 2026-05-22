# cuCIM Examples — Setup & Run

Step-by-step walkthrough for setting up cuCIM on an NVIDIA GPU and running every example in this folder. The same code was exercised on **Brev (NVIDIA L4, x86_64, CUDA 12)** and **DGX Spark (GB10, aarch64, CUDA 13)**.

Two install paths are documented below — pick whichever fits your environment:

- **Path A (Docker)** — single command, same image runs identically on Brev L4 (`amd64`) and DGX Spark (`arm64`). Recommended when the NVIDIA Container Toolkit is configured and your user is in the `docker` group.
- **Path B (Conda on host)** — closer to the `RAPIDS-cuDF/` reference, no container required. This is the path we used on DGX Spark, where `jcasalmontse` is not in the `docker` group.

## Files

| File | What it does |
|---|---|
| `install_verification.py` | Smoke test: cuCIM + CuPy import, CUDA device visible, one GPU image op. |
| `hello_cucim.py` | Synthetic 1024x1024 noisy image with one bright "lesion": Gaussian -> threshold -> labels. |
| `relevant_uses.py` | Mini digital-pathology workflow: N synthetic 512x512 tiles -> denoise -> threshold -> morphology -> labels, reports throughput. |
| `hello_cucim_blobs.py` | Synthetic 512x512 `binary_blobs` field + noise -> same pipeline. |
| `hello_cucim_mitosis.py` | Real microscopy sample `skimage.data.human_mitosis` (fluorescence-style) -> Gaussian -> **Otsu** -> nuclei count. |
| `hello_cucim_ihc.py` | Real IHC sample `skimage.data.immunohistochemistry` -> **`rgb2hed`** hematoxylin channel -> Otsu -> nuclei count. |
| `Dockerfile` | Reproducible container built on `nvcr.io/nvidia/rapidsai/base:26.04-cuda12-py3.13` (multi-arch: `amd64` + `arm64`). |

## 1. Verify the GPU

```bash
nvidia-smi
```

Output should show driver version (`>= 525` for CUDA 12, `>= 580` for CUDA 13) and at least one NVIDIA GPU. The two runs captured for this project saw `Driver Version: 550.107.02` (L4, CUDA 12.4) and `Driver Version: 580.95.05` (GB10, CUDA 13.0).

## 2. Clone the repo

```bash
gh auth login                 # use SSH auth
git clone <repo-url>
cd accelerated-library-integrations/RAPIDS-cuCIM/examples
```

## 3. Install — pick one path

### Path A — Docker (recommended for Brev L4; also works on any DGX Spark user in the `docker` group)

Build the multi-arch container image from the included `Dockerfile`:

```bash
docker build -t cucim-hello .
```

The RAPIDS base image is published for both `linux/amd64` and `linux/arm64`, so the *same* `Dockerfile` builds on Brev (L4) and on DGX Spark (GB10) without modification.

### Path B — Conda on host (used on DGX Spark in this project)

Create a fresh conda env and install cuCIM from the RAPIDS channel. On DGX Spark we used `--prefix ~/cucim-env` because `/opt/miniforge3/envs/` is root-owned; on a personal box, drop `--prefix` and use `-n cucim` instead.

```bash
# Same RAPIDS release (26.04) as the Dockerfile; CUDA 13 to match the DGX Spark driver.
# pooch is required for skimage.data.human_mitosis (downloads mitosis.tif on first use).
mamba create -y --prefix ~/cucim-env \
    -c rapidsai -c conda-forge -c nvidia \
    cucim=26.04 python=3.13 cuda-version=13.0 \
    cupy matplotlib scikit-image numpy pillow pooch

conda activate ~/cucim-env
```

> On Brev L4 (driver supports up to CUDA 12.4) substitute `cuda-version=12.8` instead. The cuCIM package version stays at 26.04 either way.

## 4. Run the install verification smoke test

A healthy run prints `PASS: cuCIM is installed and the GPU image workflow is working.`

**Docker:**

```bash
docker run --rm --gpus all cucim-hello
```

**Conda:**

```bash
python install_verification.py
```

Captured DGX Spark output: [`../screenshots/dgx_spark/00_install_verification.out`](../screenshots/dgx_spark/00_install_verification.out)

## 5. Run the cuCIM hello-world

Builds a 1024×1024 noisy synthetic image with one bright circular region, runs Gaussian blur → threshold → connected-component labeling on the GPU, and reports the detected region count (expected: `1`).

**Docker:**

```bash
docker run --rm --gpus all cucim-hello python hello_cucim.py
```

**Conda:**

```bash
python hello_cucim.py
```

Expected output:

```text
Hello cuCIM
Input type: <class 'cupy.ndarray'>
GPU device: NVIDIA ...
Image shape: (1024, 1024)
Detected regions: 1
```

Captured DGX Spark output: [`../screenshots/dgx_spark/01_hello_cucim.out`](../screenshots/dgx_spark/01_hello_cucim.out) — pipeline visualization: [`../screenshots/dgx_spark/hello_cucim_pipeline.png`](../screenshots/dgx_spark/hello_cucim_pipeline.png).

## 6. Run the First Bowl of Soup workflow demo

Generates synthetic pathology-like tiles and runs the four-stage preprocessing pipeline (Gaussian denoise → threshold → `remove_small_objects` cleanup → connected-component label) over every tile. Reports total elapsed time and throughput in tiles/second.

**Docker:**

```bash
docker run --rm --gpus all cucim-hello python relevant_uses.py             # default: 64 tiles, 512×512
docker run --rm --gpus all cucim-hello python relevant_uses.py --tiles 128 --size 512
```

**Conda:**

```bash
python relevant_uses.py
python relevant_uses.py --tiles 128 --size 512
```

Captured DGX Spark outputs: [64-tile run](../screenshots/dgx_spark/02_relevant_uses_64.out) (112.75 tiles/sec) and [128-tile run](../screenshots/dgx_spark/03_relevant_uses_128.out) (679.86 tiles/sec). Per-tile pipeline visualization: [`../screenshots/dgx_spark/relevant_uses_tiles.png`](../screenshots/dgx_spark/relevant_uses_tiles.png).

## 7. Run the synthetic-blobs demo

`binary_blobs` produces an irregular random-walk binary field; we add Gaussian noise on the GPU and run the same Gaussian → threshold → cleanup → label pipeline. This stresses cuCIM on geometry that is *not* a perfect circle, so the region count is non-trivial.

**Docker:**

```bash
docker run --rm --gpus all cucim-hello python hello_cucim_blobs.py
```

**Conda:**

```bash
python hello_cucim_blobs.py
```

## 8. Run the real microscopy (mitosis) demo

`skimage.data.human_mitosis` is a 512×512 fluorescence-style grayscale crop of human cells with several in mitosis. Nuclei are *bright* against a dark background, so we use standard Otsu thresholding (`blurred > otsu`). Output reports the Otsu cutoff and the detected nucleus count.

> First run downloads `mitosis.tif` (~250 KB) into `~/.cache/scikit-image/`. The conda env must include `pooch` for this fetch — install with `mamba install -n cucim -c conda-forge pooch` if it's missing.

**Docker:**

```bash
docker run --rm --gpus all cucim-hello python hello_cucim_mitosis.py
```

**Conda:**

```bash
python hello_cucim_mitosis.py
```

## 9. Run the real IHC (DAB) demo

`skimage.data.immunohistochemistry` is a 512×512 RGB image of human prostate tissue, DAB stained with hematoxylin counterstain. We run `cucim.skimage.color.rgb2hed` on the GPU to **separate the H / E / DAB stains**, take the hematoxylin channel (nuclei), then run Otsu + morphology + connected components.

This is the realistic "first slide of digital pathology" pipeline: color decomposition, automatic threshold, nucleus count.

**Docker:**

```bash
docker run --rm --gpus all cucim-hello python hello_cucim_ihc.py
```

**Conda:**

```bash
python hello_cucim_ihc.py
```

## Captured runs

Verbatim sanitized run captures from both platforms are already in `../screenshots/`:

- **Brev L4 (Docker)** — [`../screenshots/brev_l4_results.md`](../screenshots/brev_l4_results.md) with composite pipeline PNGs at `../screenshots/hello_cucim_pipeline.png` and `../screenshots/relevant_uses_tiles.png`, plus per-stage subdirs `../screenshots/hello_cucim/` and `../screenshots/relevant_uses/`.
- **DGX Spark (Conda)** — [`../screenshots/dgx_spark_results.md`](../screenshots/dgx_spark_results.md) with all 7 demos' `.out` files (`00_*.out` … `06_*.out`), 7 composite pipeline PNGs, the 8×8 all-tiles grid, and 5 per-stage subdirs all under `../screenshots/dgx_spark/`.

Captures were sanitized at write time — no IP addresses, hostnames, MAC addresses, tokens, passwords, or internal URLs appear in any committed `.out` or PNG file.
