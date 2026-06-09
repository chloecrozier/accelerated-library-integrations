"""Surface-code memory experiment with circuit-level noise."""

import argparse
import csv
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "results" / "surface_brev_l4.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000, help="number of shots")
    parser.add_argument("--distance", type=int, default=3, help="surface-code distance")
    parser.add_argument("--rounds", type=int, default=None, help="defaults to distance")
    parser.add_argument("--p", type=float, default=0.001, help="physical error rate")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV output path")
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        import cudaq
        import cudaq_qec as qec
    except ImportError as exc:
        sys.exit(
            "FAIL: missing CUDA-Q QEC Python dependency. Install with:\n"
            "    python -m pip install -r requirements.txt\n"
            f"Import error: {exc}"
        )
    return np, cudaq, qec


def preprocess_syndromes(np, syndromes, shots, rounds, expected_len):
    arr = np.asarray(syndromes, dtype=np.uint8)
    all_rounds = arr.reshape((shots, rounds, -1))

    z_only = all_rounds[:, :, : all_rounds.shape[2] // 2].reshape((shots, -1))
    if z_only.shape[1] == expected_len:
        return z_only

    flat = all_rounds.reshape((shots, -1))
    if flat.shape[1] == expected_len:
        return flat

    raise SystemExit(
        "FAIL: sampled syndrome shape does not match detector error model. "
        f"z_only={z_only.shape}, flat={flat.shape}, expected_len={expected_len}"
    )


def build_decoder(qec, decoder_name, h_matrix, batch_size, max_iterations=50, error_rate=None):
    if decoder_name == "nv-qldpc-decoder":
        kwargs = {
            "max_iterations": int(max_iterations),
            "bp_batch_size": int(batch_size),
            "use_sparsity": True,
        }
        if error_rate is not None:
            kwargs["error_rate"] = float(error_rate)
        try:
            return qec.get_decoder(decoder_name, h_matrix, **kwargs)
        except Exception as exc:
            raise SystemExit(
                "FAIL: could not create nv-qldpc-decoder. "
                "Use the CUDA-QX container or try single_error_lut.\n"
                f"Decoder error: {exc}"
            ) from exc

    return qec.get_decoder(decoder_name, h_matrix)


def decode_all(decoder, syndromes):
    if hasattr(decoder, "decode_batch"):
        return decoder.decode_batch(syndromes)
    return [decoder.decode(row) for row in syndromes]


def count_logical_errors(np, logical_z, observables, data, decoded):
    predictions = np.asarray(
        [np.asarray(result.result, dtype=np.uint8).reshape(-1) for result in decoded],
        dtype=np.uint8,
    )
    predicted_logicals = (observables @ predictions.T) % 2
    measured_logicals = (logical_z @ data.T) % 2
    measured_logicals = measured_logicals.reshape(-1)
    predicted_logicals = predicted_logicals.reshape(-1)

    without_decoding = int(measured_logicals.sum())
    with_decoding = int(np.sum(predicted_logicals ^ measured_logicals))
    return without_decoding, with_decoding


def run_memory_point(np, cudaq, qec, distance, rounds, p, shots, decoder_name, batch_size, max_iterations=50):
    code = qec.get_code("surface_code", distance=distance)
    logical_z = np.asarray(code.get_observables_z(), dtype=np.uint8)
    state_prep = qec.operation.prep0

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

    dem = qec.z_dem_from_memory_circuit(code, state_prep, rounds, noise)
    h_matrix = np.ascontiguousarray(dem.detector_error_matrix, dtype=np.uint8)
    observables = np.asarray(dem.observables_flips_matrix, dtype=np.uint8)
    decoder = build_decoder(qec, decoder_name, h_matrix, min(batch_size, shots), max_iterations, p)

    logical_errors_without = 0
    logical_errors_with = 0
    remaining = shots

    while remaining:
        current_shots = min(batch_size, remaining)
        syndromes, data = qec.sample_memory_circuit(code, state_prep, current_shots, rounds, noise)
        syndrome_matrix = preprocess_syndromes(np, syndromes, current_shots, rounds, h_matrix.shape[0])
        data_matrix = np.asarray(data, dtype=np.uint8)
        decoded = decode_all(decoder, syndrome_matrix)
        without, with_decoding = count_logical_errors(np, logical_z, observables, data_matrix, decoded)
        logical_errors_without += without
        logical_errors_with += with_decoding
        remaining -= current_shots

    return {
        "code": "surface_code",
        "decoder": decoder_name,
        "distance": distance,
        "rounds": rounds,
        "physical_error_probability": p,
        "shots": shots,
        "logical_errors_without_decoding": logical_errors_without,
        "logical_errors_with_decoding": logical_errors_with,
        "logical_error_rate_without_decoding": logical_errors_without / shots,
        "logical_error_rate_with_decoding": logical_errors_with / shots,
    }


def write_csv(path, rows):
    if isinstance(rows, dict):
        rows = [rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.shots < 1:
        sys.exit("FAIL: --shots must be >= 1")
    if args.distance < 3:
        sys.exit("FAIL: --distance must be >= 3")
    if args.p < 0 or args.p > 1:
        sys.exit("FAIL: --p must be in [0, 1]")

    np, cudaq, qec = load_dependencies()
    cudaq.set_target("stim")

    rounds = args.rounds if args.rounds is not None else args.distance
    if rounds < 1:
        sys.exit("FAIL: --rounds must be >= 1")

    print(
        f"Sampling {args.shots} surface-code memory shots "
        f"(distance={args.distance}, rounds={rounds}, p={args.p})..."
    )
    row = run_memory_point(
        np,
        cudaq,
        qec,
        args.distance,
        rounds,
        args.p,
        args.shots,
        "single_error_lut",
        args.shots,
    )
    write_csv(Path(args.output), row)

    print()
    print("Decoder: single_error_lut")
    print(
        "Logical errors without decoding: "
        f"{row['logical_errors_without_decoding']}/{args.shots}"
    )
    print(
        "Logical errors with decoding:    "
        f"{row['logical_errors_with_decoding']}/{args.shots}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
