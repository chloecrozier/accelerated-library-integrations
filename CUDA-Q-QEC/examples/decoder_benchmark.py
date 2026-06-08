"""Benchmark CUDA-Q QEC decoder throughput on one surface-code workload."""

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from surface_memory import (
    build_decoder,
    count_logical_errors,
    decode_all,
    load_dependencies,
    preprocess_syndromes,
    write_csv,
)


PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="brev_l4", help="label for this run, default: brev_l4")
    parser.add_argument("--decoder", default="nv-qldpc-decoder")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--bp-batch-size", type=int, default=1000)
    parser.add_argument("--output", help="CSV output path")
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
        return platform.processor() or platform.machine() or "unknown"
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else "unknown"


def synchronize_gpu():
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def time_decode(decoder, syndromes, warmup, repeats):
    for _ in range(warmup):
        decode_all(decoder, syndromes)
    synchronize_gpu()

    samples = []
    last_results = None
    for _ in range(repeats):
        synchronize_gpu()
        start = time.perf_counter()
        last_results = decode_all(decoder, syndromes)
        synchronize_gpu()
        samples.append((time.perf_counter() - start) * 1000)
    return samples, last_results


def main():
    args = parse_args()
    if args.output is None:
        decoder_label = {
            "single_error_lut": "lut",
            "nv-qldpc-decoder": "qldpc",
        }.get(args.decoder, args.decoder.replace("-", "_"))
        args.output = str(PROJECT / "results" / f"decoder_{decoder_label}_{args.platform}.csv")

    if args.shots < 1 or args.repeats < 1 or args.warmup < 0:
        sys.exit("FAIL: --shots and --repeats must be >= 1; --warmup must be >= 0")
    if args.distance < 3:
        sys.exit("FAIL: --distance must be >= 3")
    if args.p < 0 or args.p > 1:
        sys.exit("FAIL: --p must be in [0, 1]")

    np, cudaq, qec = load_dependencies()
    cudaq.set_target("stim")

    rounds = args.rounds if args.rounds is not None else args.distance
    if rounds < 1:
        sys.exit("FAIL: --rounds must be >= 1")

    code = qec.get_code("surface_code", distance=args.distance)
    logical_z = np.asarray(code.get_observables_z(), dtype=np.uint8)
    state_prep = qec.operation.prep0

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(args.p), 1)

    dem = qec.z_dem_from_memory_circuit(code, state_prep, rounds, noise)
    h_matrix = np.asarray(dem.detector_error_matrix, dtype=np.uint8)
    observables = np.asarray(dem.observables_flips_matrix, dtype=np.uint8)
    decoder = build_decoder(qec, args.decoder, h_matrix, dem, min(args.bp_batch_size, args.shots), args.max_iterations)

    print(
        f"Preparing one workload "
        f"(decoder={args.decoder}, shots={args.shots}, d={args.distance}, p={args.p})..."
    )
    syndromes, data = qec.sample_memory_circuit(code, state_prep, args.shots, rounds, noise)
    syndromes = preprocess_syndromes(np, syndromes, args.shots, rounds, h_matrix.shape[0])
    data = np.asarray(data, dtype=np.uint8)

    print(f"Timing decoder only (warmup={args.warmup}, repeats={args.repeats})...")
    samples_ms, decoded = time_decode(decoder, syndromes, args.warmup, args.repeats)
    median_ms = statistics.median(samples_ms)
    logical_without, logical_with = count_logical_errors(np, logical_z, observables, data, decoded)
    throughput = args.shots / (median_ms / 1000) if median_ms > 0 else float("inf")

    row = {
        "platform": args.platform,
        "gpu_name": gpu_name(),
        "decoder": args.decoder,
        "code": "surface_code",
        "distance": args.distance,
        "rounds": rounds,
        "physical_error_probability": args.p,
        "shots": args.shots,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "syndromes_per_second": throughput,
        "logical_errors_without_decoding": logical_without,
        "logical_errors": logical_with,
        "logical_error_rate": logical_with / args.shots,
    }
    write_csv(Path(args.output), row)

    print()
    print(f"GPU:          {row['gpu_name']}")
    print(f"Decoder:      {args.decoder}")
    print(f"Median time:  {median_ms:.3f} ms")
    print(f"Throughput:   {throughput:,.1f} syndromes/s")
    print(f"Logical errs: {logical_with}/{args.shots}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
