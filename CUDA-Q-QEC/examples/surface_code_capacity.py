"""Surface-code code-capacity sweep with direct data errors."""

import argparse
import sys
from pathlib import Path

from surface_memory import build_decoder, decode_all, load_dependencies, write_csv


PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", default="nv-qldpc-decoder")
    parser.add_argument("--bp-method", type=int, default=0)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005],
    )
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--output")
    parser.add_argument("--plot")
    return parser.parse_args()


def output_paths(args):
    decoder_label = {
        "single_error_lut": "lut",
        "nv-qldpc-decoder": "qldpc",
    }.get(args.decoder, args.decoder.replace("-", "_"))
    if args.output is None:
        args.output = str(PROJECT / "results" / f"surface_code_capacity_{decoder_label}_brev_l4.csv")
    if args.plot is None:
        args.plot = str(PROJECT / "results" / f"surface_code_capacity_{decoder_label}_brev_l4.png")


def print_table(rows):
    print()
    print(f"{'decoder':<18} {'d':>3} {'p':>8} {'shots':>8} {'raw errs':>10} {'decoded errs':>13} {'decoded rate':>13}")
    print("-" * 83)
    for row in rows:
        print(
            f"{row['decoder']:<18} "
            f"{row['distance']:>3} "
            f"{row['physical_error_probability']:>8.4f} "
            f"{row['shots']:>8} "
            f"{row['raw_logical_errors']:>10} "
            f"{row['decoded_logical_errors']:>13} "
            f"{row['decoded_logical_error_rate']:>13.4g}"
        )


def count_logical_errors(np, observables, data_errors, decoded):
    predictions = np.asarray(
        [np.asarray(result.result, dtype=np.uint8).reshape(-1) for result in decoded],
        dtype=np.uint8,
    )
    actual_logicals = np.atleast_2d((observables @ data_errors.T) % 2).T
    predicted_logicals = np.atleast_2d((observables @ predictions.T) % 2).T

    raw_errors = int(np.sum(np.any(actual_logicals, axis=1)))
    decoded_errors = int(np.sum(np.any(predicted_logicals != actual_logicals, axis=1)))
    return raw_errors, decoded_errors


def decode_batches(decoder, syndromes, batch_size):
    decoded = []
    for start in range(0, syndromes.shape[0], batch_size):
        decoded.extend(decode_all(decoder, syndromes[start : start + batch_size]))
    return decoded


def run_point(np, qec, distance, p, shots, decoder_name, bp_method):
    code = qec.get_code("surface_code", distance=distance)
    hz = np.asarray(code.get_parity_z(), dtype=np.uint8)
    observables = np.atleast_2d(np.asarray(code.get_observables_z(), dtype=np.uint8))

    data_errors = (np.random.random((shots, hz.shape[1])) < p).astype(np.uint8)
    syndromes = np.ascontiguousarray((data_errors @ hz.T) % 2, dtype=np.uint8)

    decoder = build_decoder(
        qec,
        decoder_name,
        np.ascontiguousarray(hz, dtype=np.uint8),
        min(10000, shots),
        error_rate=p,
        bp_method=bp_method if decoder_name == "nv-qldpc-decoder" else None,
    )
    decoded = decode_batches(decoder, syndromes, 10000)
    raw_errors, decoded_errors = count_logical_errors(np, observables, data_errors, decoded)

    return {
        "experiment": "surface_code_capacity",
        "code": "surface_code",
        "decoder": decoder_name,
        "bp_method": bp_method if decoder_name == "nv-qldpc-decoder" else "",
        "distance": distance,
        "physical_error_probability": p,
        "shots": shots,
        "raw_logical_errors": raw_errors,
        "decoded_logical_errors": decoded_errors,
        "raw_logical_error_rate": raw_errors / shots,
        "decoded_logical_error_rate": decoded_errors / shots,
    }


def plot_results(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        sys.exit(
            "FAIL: matplotlib is required for plotting. Install with:\n"
            "    python -m pip install -r requirements.txt\n"
            f"Import error: {exc}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    saw_zero = False
    for distance in sorted({row["distance"] for row in rows}):
        group = sorted(
            [row for row in rows if row["distance"] == distance],
            key=lambda row: row["physical_error_probability"],
        )
        x = [row["physical_error_probability"] for row in group]
        y = []
        for row in group:
            rate = row["decoded_logical_error_rate"]
            if rate == 0:
                saw_zero = True
                rate = 0.5 / row["shots"]
            y.append(rate)
        ax.plot(x, y, marker="o", label=f"d = {distance}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Decoded logical error rate")
    ax.set_title("Surface-Code Code-Capacity Logical Error Rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if saw_zero:
        ax.text(0.02, 0.02, "zero-count points shown at 0.5/shots", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    args = parse_args()
    output_paths(args)

    if args.shots < 1:
        sys.exit("FAIL: --shots must be >= 1")
    if any(distance < 3 for distance in args.distances):
        sys.exit("FAIL: --distances must be >= 3")
    if any(p < 0 or p > 1 for p in args.p_values):
        sys.exit("FAIL: --p-values must be probabilities in [0, 1]")

    np, _, qec = load_dependencies()

    rows = []
    for distance in args.distances:
        for p in args.p_values:
            print(f"Running code-capacity decoder={args.decoder}, d={distance}, p={p}, shots={args.shots}")
            rows.append(run_point(np, qec, distance, p, args.shots, args.decoder, args.bp_method))

    write_csv(Path(args.output), rows)
    print_table(rows)
    print(f"\nWrote {args.output}")
    plot_results(args.plot, rows)


if __name__ == "__main__":
    main()
