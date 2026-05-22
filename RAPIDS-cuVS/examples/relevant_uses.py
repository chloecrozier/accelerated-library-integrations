"""Molecular embedding search with CPU exact search and cuVS indexes.

Downloads the public GuacaMol all-SMILES corpus, embeds SMILES strings,
and compares CPU exact cosine search with cuVS brute-force and IVF-Flat
nearest-neighbor search. See ../README.md for context.
"""

import argparse
import gc
import io
import math
import os
import logging
import warnings
import sys
import tempfile
import time
import urllib.request
import zipfile

GUACAMOL_URL = "https://ndownloader.figshare.com/files/13612745"
MIN_EXPECTED_MOLECULES = 1_000_000

logging.basicConfig(level=logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        help="maximum corpus rows to embed; use 0 for full corpus (default: 1_000_000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=128,
        help="number of indexed molecules to use as queries (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="embedding batch size (default: 64)"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="nearest neighbors per query (default: 10)"
    )
    parser.add_argument(
        "--embedding-model",
        default="mist-models/mist-28M-ti624ev1",
        help="Hugging Face model used to generate molecule embeddings",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="tokenizer max sequence length (default: 512)",
    )
    parser.add_argument(
        "--n-lists",
        type=int,
        default=None,
        help="IVF-Flat inverted lists; defaults to sqrt(n), capped at 1024",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=20,
        help="IVF-Flat lists to probe per query (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed for selecting query molecules (default: 7)",
    )
    return parser.parse_args()


def require_positive(name, value):
    if value < 1:
        sys.exit(f"FAIL: --{name} must be >= 1")


def require_nonnegative(name, value):
    if value < 0:
        sys.exit(f"FAIL: --{name} must be >= 0")


def load_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import torch
        import cupy as cp
        import smirk  # noqa: F401 - imported to fail early when tokenizer support is missing
        from transformers import AutoModel, AutoTokenizer
        from cuvs.neighbors import brute_force, ivf_flat
    except ImportError as e:
        sys.exit(f"FAIL: missing dependency -> {e}")

    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "cp": cp,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "brute_force": brute_force,
        "ivf_flat": ivf_flat,
    }


def download_guacamol_dataset():
    tmp = tempfile.NamedTemporaryFile(suffix=".download", delete=False)
    tmp.close()

    def reporthook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        downloaded_mb = downloaded / 1e6
        if total_size > 0:
            total_mb = total_size / 1e6
            percent = min(100.0, (downloaded / total_size) * 100.0)
            print(
                f"  downloaded {downloaded_mb:,.2f}/{total_mb:,.2f} MB ({percent:5.1f}%)",
                end="\r",
                flush=True,
            )
        else:
            print(f"  downloaded {downloaded_mb:,.2f} MB", end="\r", flush=True)

    try:
        print(f"Downloading GuacaMol all-SMILES corpus from {GUACAMOL_URL} ...")
        urllib.request.urlretrieve(GUACAMOL_URL, tmp.name, reporthook)
        print()
        size_mb = os.path.getsize(tmp.name) / 1e6
        print(f"  {size_mb:.2f} MB on disk")
        return tmp.name
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


def iter_smiles_lines(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            candidates = [
                name
                for name in zf.namelist()
                if not name.endswith("/")
                and name.lower().endswith((".smi", ".smiles", ".txt", ".csv"))
            ]
            if not candidates:
                candidates = [name for name in zf.namelist() if not name.endswith("/")]
            if not candidates:
                sys.exit(
                    "FAIL: GuacaMol download did not contain a readable molecule file"
                )
            with zf.open(candidates[0]) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    yield line
        return

    with open(path, encoding="utf-8") as f:
        yield from f


def parse_smiles_line(line):
    stripped = line.strip()
    if not stripped:
        return None
    first_field = stripped.split(",", 1)[0]
    token = first_field.split()[0].strip("\"'")
    if token.lower() in {"smiles", "canonical_smiles"}:
        return None
    return token


def prepare_molecules(path, limit, deps):
    pd = deps["pd"]

    records = []
    total_rows = 0
    max_records = None if limit == 0 else limit

    for line in iter_smiles_lines(path):
        smiles = parse_smiles_line(line)
        if smiles is None:
            continue

        total_rows += 1
        if max_records is not None and total_rows > max_records:
            continue

        records.append(
            {
                "source_row": total_rows - 1,
                "smiles": smiles,
            }
        )

    if total_rows < MIN_EXPECTED_MOLECULES:
        sys.exit(
            f"FAIL: expected a corpus with >={MIN_EXPECTED_MOLECULES:,} molecules, "
            f"but found {total_rows:,}"
        )
    if not records:
        sys.exit("FAIL: no molecules found after reading the corpus")

    out = pd.DataFrame(records)
    return out, total_rows


def embed_smiles(smiles, model_name, batch_size, max_length, deps):
    np = deps["np"]
    torch = deps["torch"]
    AutoModel = deps["AutoModel"]
    AutoTokenizer = deps["AutoTokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Embedding device: {device}")

    batches = []
    total = len(smiles)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = smiles[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = (
                outputs.last_hidden_state[:, 0, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            batches.append(emb)
            done = min(start + batch_size, total)
            print(f"  embedded {done:>6,}/{total:,} molecules", end="\r")
    print()

    embeddings = np.concatenate(batches, axis=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return embeddings, elapsed_ms


def l2_normalize(x, np):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def cpu_cosine_search(dataset, queries, k, np):
    data = l2_normalize(dataset, np)
    q = l2_normalize(queries, np)
    all_distances = []
    all_neighbors = []

    for start in range(0, q.shape[0], 32):
        batch = q[start : start + 32]
        sims = batch @ data.T
        candidate_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(batch.shape[0])[:, None]
        candidate_sims = sims[rows, candidate_idx]
        order = np.argsort(-candidate_sims, axis=1)
        neighbors = candidate_idx[rows, order]
        distances = 1.0 - candidate_sims[rows, order]
        all_neighbors.append(neighbors.astype(np.int64))
        all_distances.append(distances.astype(np.float32))

    return np.vstack(all_distances), np.vstack(all_neighbors)


def recall_at_k(exact_neighbors, approx_neighbors):
    hits = 0
    total = exact_neighbors.shape[0] * exact_neighbors.shape[1]
    for exact, approx in zip(exact_neighbors, approx_neighbors):
        hits += len(set(exact.tolist()) & set(approx.tolist()))
    return hits / total if total else 0.0


def auto_n_lists(n_samples):
    return max(2, min(1024, int(math.sqrt(n_samples))))


def speedup_ratio(baseline_ms, candidate_ms):
    return baseline_ms / max(candidate_ms, 1e-9)


def main():
    args = parse_args()
    require_nonnegative("limit", args.limit)
    for name in ("queries", "batch-size", "k", "max-length", "n-probes"):
        attr = name.replace("-", "_")
        require_positive(name, getattr(args, attr))
    if args.n_lists is not None:
        require_positive("n-lists", args.n_lists)

    deps = load_dependencies()
    np = deps["np"]
    cp = deps["cp"]
    brute_force = deps["brute_force"]
    ivf_flat = deps["ivf_flat"]

    if cp.cuda.runtime.getDeviceCount() == 0:
        sys.exit("FAIL: no CUDA devices detected")

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = (
        props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    )

    dataset_path = download_guacamol_dataset()
    try:
        records, total_rows = prepare_molecules(dataset_path, args.limit, deps)
    finally:
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

    if len(records) < 2:
        sys.exit("FAIL: need at least two molecules for nearest-neighbor search")

    embeddings, embed_ms = embed_smiles(
        records["smiles"].tolist(),
        args.embedding_model,
        args.batch_size,
        args.max_length,
        deps,
    )

    n_samples, dim = embeddings.shape
    k = min(args.k, n_samples)
    n_queries = min(args.queries, n_samples)
    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(n_samples, size=n_queries, replace=False)
    queries = embeddings[query_ids]

    n_lists = args.n_lists if args.n_lists is not None else auto_n_lists(n_samples)
    n_lists = min(n_lists, n_samples)
    n_probes = min(args.n_probes, n_lists)

    print("=" * 72)
    print(f"GPU:        {gpu_name}")
    limit_label = "full corpus" if args.limit == 0 else f"first {args.limit:,} rows"
    print(f"Dataset:    GuacaMol all SMILES ({total_rows:,} molecules total)")
    print(f"Embedded:   {n_samples:,} molecules from {limit_label}")
    print(f"Embedding:  {dim} dimensions, {embed_ms:,.1f} ms")
    print(f"Queries:    {n_queries:,}, k={k}")
    print(f"IVF-Flat:   n_lists={n_lists}, n_probes={n_probes}")
    print("=" * 72)
    print()

    # Warm up CUDA before measuring cuVS work.
    cp.asarray([[0.0]], dtype=cp.float32).sum()
    cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    cpu_distances, cpu_neighbors = cpu_cosine_search(embeddings, queries, k, np)
    cpu_ms = (time.perf_counter() - t0) * 1000

    dataset_gpu = cp.asarray(embeddings, dtype=cp.float32)
    queries_gpu = cp.asarray(queries, dtype=cp.float32)

    t0 = time.perf_counter()
    exact_index = brute_force.build(dataset_gpu, metric="cosine")
    cp.cuda.runtime.deviceSynchronize()
    exact_build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    exact_dist_gpu, exact_neighbors_gpu = brute_force.search(
        exact_index, queries_gpu, k
    )
    cp.cuda.runtime.deviceSynchronize()
    exact_search_ms = (time.perf_counter() - t0) * 1000
    exact_distances = cp.asarray(exact_dist_gpu).get()
    exact_neighbors = cp.asarray(exact_neighbors_gpu).get()

    build_params = ivf_flat.IndexParams(
        n_lists=n_lists,
        metric="cosine",
        kmeans_trainset_fraction=1.0,
    )
    search_params = ivf_flat.SearchParams(n_probes=n_probes)

    t0 = time.perf_counter()
    approx_index = ivf_flat.build(build_params, dataset_gpu)
    cp.cuda.runtime.deviceSynchronize()
    approx_build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    approx_dist_gpu, approx_neighbors_gpu = ivf_flat.search(
        search_params, approx_index, queries_gpu, k
    )
    cp.cuda.runtime.deviceSynchronize()
    approx_search_ms = (time.perf_counter() - t0) * 1000
    approx_distances = cp.asarray(approx_dist_gpu).get()
    approx_neighbors = cp.asarray(approx_neighbors_gpu).get()

    cpu_recall = recall_at_k(exact_neighbors, cpu_neighbors)
    approx_recall = recall_at_k(exact_neighbors, approx_neighbors)

    print("[exact search]")
    print("-" * 40)
    print(f"  CPU cosine        {cpu_ms:>10.1f} ms")
    print(f"  cuVS build        {exact_build_ms:>10.1f} ms")
    print(
        f"  cuVS search       {exact_search_ms:>10.1f} ms  ({speedup_ratio(cpu_ms, exact_search_ms):>5.1f}x vs CPU)"
    )
    print(f"  CPU recall@{k:<2}      {cpu_recall:>10.3f} vs cuVS exact")
    print()

    print("[approximate search]")
    print("-" * 40)
    print(f"  IVF-Flat build    {approx_build_ms:>10.1f} ms")
    print(
        f"  IVF-Flat search   {approx_search_ms:>10.1f} ms  ({speedup_ratio(cpu_ms, approx_search_ms):>5.1f}x vs CPU)"
    )
    print(f"  IVF recall@{k:<2}      {approx_recall:>10.3f} vs cuVS exact")
    print()

    # Drop GPU-backed objects before interpreter shutdown so their destructors
    # run while imported modules are still alive.
    del approx_index, approx_dist_gpu, approx_neighbors_gpu
    del exact_index, exact_dist_gpu, exact_neighbors_gpu
    del dataset_gpu, queries_gpu
    gc.collect()


if __name__ == "__main__":
    main()
