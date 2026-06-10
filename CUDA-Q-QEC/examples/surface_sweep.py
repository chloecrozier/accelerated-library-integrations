"""Surface-code logical error-rate sweep for threshold-style QEC plots."""

import argparse
import sys
from pathlib import Path

from surface_memory import load_dependencies, run_memory_point, write_csv


PROJECT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", default="nv-qldpc-decoder")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01],
    )
    parser.add_argument("--rounds", type=int, default=None, help="defaults to distance")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--bp-method", type=int, default=0, help="QLDPC BP method")
    parser.add_argument("--output")
    parser.add_argument("--plot")
    return parser.parse_args()


def print_table(rows):
    print()
    print(f"{'decoder':<18} {'d':>3} {'r':>3} {'p':>8} {'shots':>8} {'logical errors':>16} {'rate':>10}")
    print("-" * 79)
    for row in rows:
        print(
            f"{row['decoder']:<18} "
            f"{row['distance']:>3} "
            f"{row['rounds']:>3} "
            f"{row['physical_error_probability']:>8.4f} "
            f"{row['shots']:>8} "
            f"{row['logical_errors_with_decoding']:>16} "
            f"{row['logical_error_rate_with_decoding']:>10.4g}"
        )


def surface_title(rows):
    rounds_match_distance = all(row["rounds"] == row["distance"] for row in rows)
    if rounds_match_distance:
        return f"Circuit-Level Surface-Code Logical Error Rate, rounds=d ({rows[0]['decoder']})"
    return f"Circuit-Level Surface-Code Logical Error Rate, custom rounds ({rows[0]['decoder']})"


def plot_results(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        sys.exit(
            "FAIL: matplotlib is required for plotting. Install with:\n"
            "    python -m pip install -r requirements.txt\n"
            f"Import error: {exc}"
        )

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
            rate = row["logical_error_rate_with_decoding"]
            if rate == 0:
                saw_zero = True
                rate = 0.5 / row["shots"]
            y.append(rate)
        ax.plot(x, y, marker="o", label=f"d = {distance}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title(surface_title(rows))
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
    decoder_label = {
        "single_error_lut": "lut",
        "nv-qldpc-decoder": "qldpc",
    }.get(args.decoder, args.decoder.replace("-", "_"))
    if args.output is None:
        args.output = str(PROJECT / "results" / f"surface_sweep_{decoder_label}_brev_l4.csv")
    if args.plot is None:
        args.plot = str(PROJECT / "results" / f"surface_sweep_{decoder_label}_brev_l4.png")

    if args.shots < 1 or args.batch_size < 1:
        sys.exit("FAIL: --shots and --batch-size must be >= 1")
    if any(distance < 3 for distance in args.distances):
        sys.exit("FAIL: --distances must be >= 3")
    if args.rounds is not None and args.rounds < 1:
        sys.exit("FAIL: --rounds must be >= 1")
    if any(p < 0 or p > 1 for p in args.p_values):
        sys.exit("FAIL: --p-values must be probabilities in [0, 1]")

    np, cudaq, qec = load_dependencies()
    cudaq.set_target("stim")

    rows = []
    for distance in args.distances:
        rounds = args.rounds if args.rounds is not None else distance
        for p in args.p_values:
            print(f"Running decoder={args.decoder}, d={distance}, rounds={rounds}, p={p}, shots={args.shots}")
            rows.append(
                run_memory_point(
                    np,
                    cudaq,
                    qec,
                    distance,
                    rounds,
                    p,
                    args.shots,
                    args.decoder,
                    args.batch_size,
                    args.max_iterations,
                    args.bp_method if args.decoder == "nv-qldpc-decoder" else None,
                )
            )

    write_csv(Path(args.output), rows)
    print_table(rows)
    print(f"\nWrote {args.output}")
    plot_results(Path(args.plot), rows)


if __name__ == "__main__":
    main()
