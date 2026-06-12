"""Compare CPU and GPU throughput for a batched QEC syndrome calculation."""

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from surface_memory import load_dependencies, write_csv


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "results" / "cpu_gpu_syndrome_brev_l4.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="brev_l4")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def gpu_name():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else "unknown"


def cpu_syndromes(errors, h_matrix):
    return ((errors @ h_matrix.T) % 2).astype("uint8")


def time_cpu(errors, h_matrix, warmup, repeats):
    for _ in range(warmup):
        cpu_syndromes(errors, h_matrix)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        cpu_syndromes(errors, h_matrix)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def time_gpu(cp, errors, h_matrix, warmup, repeats):
    errors_gpu = cp.asarray(errors)
    h_gpu = cp.asarray(h_matrix)
    cp.cuda.runtime.deviceSynchronize()

    for _ in range(warmup):
        (errors_gpu @ h_gpu.T) % 2
    cp.cuda.runtime.deviceSynchronize()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        (errors_gpu @ h_gpu.T) % 2
        cp.cuda.runtime.deviceSynchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples, h_gpu


def main():
    args = parse_args()
    if args.shots < 1 or args.repeats < 1 or args.warmup < 0:
        sys.exit("FAIL: --shots and --repeats must be >= 1; --warmup must be >= 0")
    if args.distance < 3:
        sys.exit("FAIL: --distance must be >= 3")
    if args.p < 0 or args.p > 1:
        sys.exit("FAIL: --p must be in [0, 1]")

    try:
        import cupy as cp
    except ImportError as exc:
        sys.exit(
            "FAIL: CuPy is required for the CPU/GPU benchmark. Install with:\n"
            "    python -m pip install -r requirements.txt\n"
            f"Import error: {exc}"
        )

    np, cudaq, qec = load_dependencies()
    cudaq.set_target("stim")

    rounds = args.rounds if args.rounds is not None else args.distance
    code = qec.get_code("surface_code", distance=args.distance)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(args.p), 1)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, rounds, noise)
    h_matrix = np.ascontiguousarray(dem.detector_error_matrix, dtype=np.float32)

    rng = np.random.default_rng(1234)
    errors = (rng.random((args.shots, h_matrix.shape[1])) < args.p).astype(np.float32)

    check_rows = min(args.shots, 256)
    expected = cpu_syndromes(errors[:check_rows], h_matrix)
    actual = cp.asnumpy(((cp.asarray(errors[:check_rows]) @ cp.asarray(h_matrix).T) % 2).astype(cp.uint8))
    if not np.array_equal(expected, actual):
        sys.exit("FAIL: CPU and GPU syndrome results did not match")

    print(
        f"Timing batched syndrome calculation "
        f"(shots={args.shots}, d={args.distance}, rounds={rounds}, H={h_matrix.shape})..."
    )
    cpu_ms = time_cpu(errors, h_matrix, args.warmup, args.repeats)
    gpu_ms, _ = time_gpu(cp, errors, h_matrix, args.warmup, args.repeats)
    cpu_median = statistics.median(cpu_ms)
    gpu_median = statistics.median(gpu_ms)
    speedup = cpu_median / gpu_median if gpu_median > 0 else float("inf")

    rows = [
        {
            "benchmark": "cpu_gpu_syndrome",
            "backend": "cpu_numpy",
            "platform": args.platform,
            "gpu_name": gpu_name(),
            "code": "surface_code",
            "distance": args.distance,
            "rounds": rounds,
            "physical_error_probability": args.p,
            "shots": args.shots,
            "h_rows": h_matrix.shape[0],
            "h_cols": h_matrix.shape[1],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "median_ms": cpu_median,
            "min_ms": min(cpu_ms),
            "max_ms": max(cpu_ms),
            "syndromes_per_second": args.shots / (cpu_median / 1000),
            "speedup_vs_cpu": 1.0,
            "scope": "compute_only",
        },
        {
            "benchmark": "cpu_gpu_syndrome",
            "backend": "gpu_cupy",
            "platform": args.platform,
            "gpu_name": gpu_name(),
            "code": "surface_code",
            "distance": args.distance,
            "rounds": rounds,
            "physical_error_probability": args.p,
            "shots": args.shots,
            "h_rows": h_matrix.shape[0],
            "h_cols": h_matrix.shape[1],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "median_ms": gpu_median,
            "min_ms": min(gpu_ms),
            "max_ms": max(gpu_ms),
            "syndromes_per_second": args.shots / (gpu_median / 1000),
            "speedup_vs_cpu": speedup,
            "scope": "compute_only",
        },
    ]
    write_csv(Path(args.output), rows)

    print()
    print(f"CPU median:   {cpu_median:.3f} ms")
    print(f"GPU median:   {gpu_median:.3f} ms")
    print(f"GPU speedup:  {speedup:.2f}x")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
