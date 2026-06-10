"""Benchmark decoder time and accuracy on surface-code workloads."""

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
    load_dependencies,
    preprocess_syndromes,
    write_csv,
)


PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="brev_l4")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--distance", type=int, default=None, help="run one distance only")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    parser.add_argument("--rounds", type=int, default=None, help="defaults to distance")
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--bp-batch-size", type=int, default=10000)
    parser.add_argument("--bp-methods", type=int, nargs="+", default=[0])
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


def decode_single_loop(decoder, syndromes):
    return [decoder.decode(row) for row in syndromes]


def decode_all(decoder, syndromes):
    if hasattr(decoder, "decode_batch"):
        return decoder.decode_batch(syndromes)
    return decode_single_loop(decoder, syndromes)


def decoder_path(decoder):
    return "decode_batch" if hasattr(decoder, "decode_batch") else "decode_loop"


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


def decoder_variants(bp_methods):
    variants = [("LUT", "single_error_lut", None)]
    variants += [(f"BP={bp}", "nv-qldpc-decoder", bp) for bp in bp_methods]
    return variants


def improvement(raw_errors, decoded_errors):
    if decoded_errors == 0:
        return "inf" if raw_errors else ""
    return raw_errors / decoded_errors


def make_row(args, variant, decoder_name, bp_method, path, gpu, distance, rounds, samples_ms, decoded, np, logical_z, observables, data):
    median_ms = statistics.median(samples_ms)
    logical_without, logical_with = count_logical_errors(np, logical_z, observables, data, decoded)
    return {
        "platform": args.platform,
        "gpu_name": gpu,
        "variant": variant,
        "decoder": decoder_name,
        "bp_method": bp_method if bp_method is not None else "",
        "decode_path": path,
        "code": "surface_code",
        "distance": distance,
        "rounds": rounds,
        "physical_error_probability": args.p,
        "shots": args.shots,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "syndromes_per_second": args.shots / (median_ms / 1000),
        "raw_logical_errors": logical_without,
        "decoded_logical_errors": logical_with,
        "raw_logical_error_rate": logical_without / args.shots,
        "decoded_logical_error_rate": logical_with / args.shots,
        "logical_error_reduction_vs_raw": improvement(logical_without, logical_with),
        "speedup_vs_lut": "",
        "logical_error_ratio_vs_lut": "",
    }


def fill_lut_comparisons(rows):
    for distance in sorted({row["distance"] for row in rows}):
        group = [row for row in rows if row["distance"] == distance]
        lut = next((row for row in group if row["variant"] == "LUT"), None)
        if not lut:
            continue

        lut_speed = float(lut["syndromes_per_second"])
        lut_rate = float(lut["decoded_logical_error_rate"])
        for row in group:
            row["speedup_vs_lut"] = float(row["syndromes_per_second"]) / lut_speed
            row_rate = float(row["decoded_logical_error_rate"])
            row["logical_error_ratio_vs_lut"] = row_rate / lut_rate if lut_rate else ""


def print_table(rows, gpu, args):
    print()
    print(f"GPU:      {gpu}")
    print(f"Workload: surface_code, p={args.p}, shots={args.shots}")
    print(f"BP methods: {', '.join(str(bp) for bp in args.bp_methods)}")
    print()
    print(
        f"{'d':>3} {'variant':<8} {'median ms':>10} {'decoded errs':>13} "
        f"{'decoded rate':>13} {'speed/LUT':>9} {'rate/LUT':>9}"
    )
    print("-" * 84)
    for row in rows:
        speedup = row["speedup_vs_lut"]
        speedup_text = f"{float(speedup):.2f}x" if speedup != "" else ""
        rate_ratio = row["logical_error_ratio_vs_lut"]
        rate_ratio_text = f"{float(rate_ratio):.2f}x" if rate_ratio != "" else ""
        print(
            f"{row['distance']:>3} "
            f"{row['variant']:<8} "
            f"{row['median_ms']:>10.3f} "
            f"{row['decoded_logical_errors']:>13}/{args.shots:<5} "
            f"{row['decoded_logical_error_rate']:>13.4g} "
            f"{speedup_text:>9} "
            f"{rate_ratio_text:>9}"
        )


def main():
    args = parse_args()
    distances = [args.distance] if args.distance else args.distances
    if args.output is None:
        name = f"decoder_lut_bp_d{args.distance}" if args.distance else "decoder_lut_bp_sweep"
        args.output = str(PROJECT / "results" / f"{name}_{args.platform}.csv")

    if args.shots < 1 or args.repeats < 1 or args.warmup < 0:
        sys.exit("FAIL: --shots and --repeats must be >= 1; --warmup must be >= 0")
    if any(distance < 3 for distance in distances):
        sys.exit("FAIL: distances must be >= 3")
    if args.p < 0 or args.p > 1:
        sys.exit("FAIL: --p must be in [0, 1]")

    np, cudaq, qec = load_dependencies()
    cudaq.set_target("stim")

    gpu = gpu_name()
    rows = []
    for distance in distances:
        rounds = args.rounds if args.rounds is not None else distance
        if rounds < 1:
            sys.exit("FAIL: --rounds must be >= 1")

        code = qec.get_code("surface_code", distance=distance)
        logical_z = np.asarray(code.get_observables_z(), dtype=np.uint8)
        state_prep = qec.operation.prep0

        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(args.p), 1)

        dem = qec.z_dem_from_memory_circuit(code, state_prep, rounds, noise)
        h_matrix = np.ascontiguousarray(dem.detector_error_matrix, dtype=np.uint8)
        observables = np.asarray(dem.observables_flips_matrix, dtype=np.uint8)

        print(f"Preparing d={distance}, rounds={rounds}, shots={args.shots}, p={args.p}...")
        syndromes, data = qec.sample_memory_circuit(code, state_prep, args.shots, rounds, noise)
        syndromes = preprocess_syndromes(np, syndromes, args.shots, rounds, h_matrix.shape[0])
        data = np.asarray(data, dtype=np.uint8)

        for variant, decoder_name, bp_method in decoder_variants(args.bp_methods):
            try:
                decoder = build_decoder(
                    qec,
                    decoder_name,
                    h_matrix,
                    min(args.bp_batch_size, args.shots),
                    args.max_iterations,
                    args.p,
                    bp_method,
                )
            except SystemExit as exc:
                print(f"Skipping {variant} at d={distance}: {exc}")
                continue

            path = decoder_path(decoder)
            print(f"Timing {variant} ({path}, warmup={args.warmup}, repeats={args.repeats})...")
            samples_ms, decoded = time_decode(decoder, syndromes, args.warmup, args.repeats)
            rows.append(
                make_row(
                    args,
                    variant,
                    decoder_name,
                    bp_method,
                    path,
                    gpu,
                    distance,
                    rounds,
                    samples_ms,
                    decoded,
                    np,
                    logical_z,
                    observables,
                    data,
                )
            )

    if not rows:
        sys.exit("FAIL: no decoder benchmark rows were generated")

    fill_lut_comparisons(rows)
    write_csv(Path(args.output), rows)
    print_table(rows, gpu, args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
