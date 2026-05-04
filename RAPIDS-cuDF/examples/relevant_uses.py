"""pandas vs cuDF vs Dask-cuDF benchmark on real public data.

Downloads NYC TLC Yellow Taxi (~50MB), runs the same load -> filter ->
groupby -> sort pipeline on each backend, prints per-stage timings in ms,
then deletes the file. See ../README.md for the bigger industry story
(IBM x NVIDIA / Velox + cuDF).
"""

import os
import sys
import tempfile
import time
import urllib.request

URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2024-01.parquet"
)


def benchmark(lib, path, sync, lazy=False):
    """Same pipeline against pandas / cuDF / dask-cuDF. Returns ms per step."""
    times = {}

    if lazy:
        import dask

        def go(x):
            # synchronous scheduler so persist() blocks until done
            with dask.config.set(scheduler="synchronous"):
                return x.persist()
    else:
        def go(x):
            return x

    # warm up the parquet reader -- cold start dominates the first read
    go(lib.read_parquet(path))
    sync()

    t0 = time.perf_counter()
    df = go(lib.read_parquet(path))
    sync()
    times["read"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    over10 = go(df[df["fare_amount"] > 10])
    sync()
    times["filter"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = go(over10.groupby("passenger_count").agg({
        "trip_distance": ["mean", "max"],
        "fare_amount":   ["mean", "sum"],
        "tip_amount":    "sum",
    }))
    sync()
    times["groupby"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = go(over10.sort_values("total_amount", ascending=False).head(100))
    sync()
    times["sort"] = (time.perf_counter() - t0) * 1000

    return times


def main():
    try:
        import pandas as pd
        import cudf
        import cupy as cp
    except ImportError as e:
        sys.exit(f"FAIL: missing dep -> {e}")

    try:
        import dask_cudf
        have_dask = True
    except ImportError:
        have_dask = False

    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    bar = "=" * 60
    sub = "-" * 40

    print(bar)
    print(f"GPU:     {name}")
    print(f"Dataset: NYC TLC Yellow Taxi 2024-01 (~3M rows)")
    print(bar)
    print()
    print("cuDF in the wild — see ../README.md for the IBM x NVIDIA / Velox story.")
    print()

    # tempfile lives outside the repo so it can't sneak into a commit
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    path = tmp.name

    try:
        print(f"Downloading {URL.split('/')[-1]} ...")
        urllib.request.urlretrieve(URL, path)
        print(f"  {os.path.getsize(path) / 1e6:.1f} MB on disk\n")

        # one-time GPU warmup so CUDA init isn't billed to anything
        cudf.Series([1, 2, 3]).sum()
        cp.cuda.runtime.deviceSynchronize()

        gpu_sync = cp.cuda.runtime.deviceSynchronize
        results = {
            "pandas":    benchmark(pd, path, sync=lambda: None),
            "cuDF":      benchmark(cudf, path, sync=gpu_sync),
        }
        if have_dask:
            results["dask-cuDF"] = benchmark(dask_cudf, path, sync=gpu_sync, lazy=True)

        for step in results["pandas"]:
            print(f"[{step}]")
            print(sub)
            base = results["pandas"][step]
            for lib_name, t in results.items():
                ms = t[step]
                if lib_name == "pandas":
                    print(f"  {lib_name:<10} {ms:>8.1f} ms")
                else:
                    speedup = base / ms if ms > 0 else float("inf")
                    print(f"  {lib_name:<10} {ms:>8.1f} ms  ({speedup:>5.1f}x vs pandas)")
            print()

        print(bar)
        print("Notes")
        print(sub)
        print("- GB10 / Grace Blackwell is an SoC with unified memory between")
        print("  the ARM CPU and Blackwell GPU (no PCIe transfers between them).")
        print("- Dask-cuDF on a single GPU adds scheduler overhead vs plain cuDF.")
        print("  It pays off when scaling across multiple GPUs / nodes (e.g. a")
        print("  DGX Spark cluster of GB10s).")
        if not have_dask:
            print("- dask-cuDF isn't installed; that comparison was skipped.")
            print("  install: conda install -c rapidsai -c conda-forge -c nvidia dask-cudf")
        print(bar)
    finally:
        if os.path.exists(path):
            os.remove(path)
            print(f"\nCleaned up {path}")


if __name__ == "__main__":
    main()
