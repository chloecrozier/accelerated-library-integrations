"""Steane code-capacity QEC sweep: raw vs decoded logical error rates."""

import argparse
import csv
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "results" / "steane_brev_l4.csv"
DEFAULT_P_VALUES = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000, help="shots per p value")
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=DEFAULT_P_VALUES,
        help="physical error probabilities to sweep",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV output path")
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        import cudaq_qec as qec
    except ImportError as exc:
        sys.exit(
            "FAIL: missing CUDA-Q QEC Python dependency. Install with:\n"
            "    python -m pip install -r requirements.txt\n"
            f"Import error: {exc}"
        )
    return np, qec


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print()
    print(
        f"{'p':>8} {'shots':>8} {'raw errors':>12} "
        f"{'decoded errors':>15} {'raw rate':>10} {'decoded rate':>13}"
    )
    print("-" * 78)
    for row in rows:
        print(
            f"{row['physical_error_probability']:>8.4f} "
            f"{row['shots']:>8} "
            f"{row['raw_logical_errors']:>12} "
            f"{row['decoded_logical_errors']:>15} "
            f"{row['raw_logical_error_rate']:>10.4f} "
            f"{row['decoded_logical_error_rate']:>13.4f}"
        )


def main():
    args = parse_args()
    if args.shots < 1:
        sys.exit("FAIL: --shots must be >= 1")
    if not args.p_values or any(p < 0 or p > 1 for p in args.p_values):
        sys.exit("FAIL: --p-values must be probabilities in [0, 1]")

    np, qec = load_dependencies()

    steane = qec.get_code("steane")
    hz = np.asarray(steane.get_parity_z(), dtype=np.uint8)
    observable = np.asarray(steane.get_observables_z(), dtype=np.uint8)
    decoder = qec.get_decoder("single_error_lut", hz)

    rows = []
    for p in args.p_values:
        raw_logical_errors = 0
        decoded_logical_errors = 0

        for _ in range(args.shots):
            data_error = np.asarray(
                qec.generate_random_bit_flips(hz.shape[1], p),
                dtype=np.uint8,
            )
            syndrome = (hz @ data_error) % 2

            decoded = decoder.decode(syndrome)
            prediction = np.asarray(decoded.result, dtype=np.uint8).reshape(-1)

            actual_logical = (observable @ data_error) % 2
            predicted_logical = (observable @ prediction) % 2

            if bool(np.any(actual_logical)):
                raw_logical_errors += 1
            if bool(np.any(predicted_logical != actual_logical)):
                decoded_logical_errors += 1

        rows.append(
            {
                "code": "steane",
                "decoder": "single_error_lut",
                "shots": args.shots,
                "physical_error_probability": p,
                "raw_logical_errors": raw_logical_errors,
                "decoded_logical_errors": decoded_logical_errors,
                "raw_logical_error_rate": raw_logical_errors / args.shots,
                "decoded_logical_error_rate": decoded_logical_errors / args.shots,
            }
        )

    write_csv(Path(args.output), rows)
    print_table(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
