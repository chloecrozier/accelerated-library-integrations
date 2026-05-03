"""Tiny pandas vs cuDF benchmark on real public data.

Downloads NYC TLC Yellow Taxi (~50MB), runs the same load -> filter ->
groupby pipeline on CPU and GPU, prints per-step timings in ms, then
deletes the file. See ../README.md for the bigger industry story
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


def benchmark(lib, path, sync):
    """Same load -> filter -> groupby pipeline against pandas or cuDF."""
    times = {}

    t0 = time.perf_counter()
    df = lib.read_parquet(path)
    sync()
    times["read"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    over10 = df[df["fare_amount"] > 10]
    sync()
    times["filter"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = over10.groupby("passenger_count")["trip_distance"].mean()
    sync()
    times["groupby"] = (time.perf_counter() - t0) * 1000

    return times


def main():
    try:
        import pandas as pd
        import cudf
        import cupy as cp
    except ImportError as e:
        sys.exit(f"FAIL: missing dep -> {e}")

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n")

    print("cuDF in the wild")
    print("-" * 40)
    print("Where it shows up today:")
    print("  - Financial services: ETL + risk on tick data")
    print("  - Telco / ad tech:    clickstream + log analytics")
    print("  - Retail:             SKU-level demand + recommender prep")
    print("  - Data platforms:     Presto / Spark via Velox + cuDF")
    print("                        (IBM x NVIDIA, GTC 2026 — see ../README.md)")
    print()

    # tempfile lives outside the repo so it can't sneak into a commit
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    path = tmp.name

    try:
        print(f"Downloading {URL.split('/')[-1]} ...")
        urllib.request.urlretrieve(URL, path)
        print(f"  {os.path.getsize(path) / 1e6:.1f} MB on disk\n")

        # first GPU call always pays for CUDA init -- burn that off here
        # so it doesn't show up in the read timing
        cudf.Series([1, 2, 3]).sum()
        cp.cuda.runtime.deviceSynchronize()

        # GPU work is async, so we have to sync before stopping the clock
        cpu = benchmark(pd, path, sync=lambda: None)
        gpu = benchmark(cudf, path, sync=cp.cuda.runtime.deviceSynchronize)

        print(f"{'step':<10} {'pandas (ms)':>12} {'cuDF (ms)':>12} {'speedup':>10}")
        print("-" * 48)
        for step in cpu:
            c, g = cpu[step], gpu[step]
            ratio = c / g if g else float("inf")
            print(f"{step:<10} {c:>12.1f} {g:>12.1f} {ratio:>9.1f}x")
    finally:
        if os.path.exists(path):
            os.remove(path)
            print(f"\nCleaned up {path}")


if __name__ == "__main__":
    main()
