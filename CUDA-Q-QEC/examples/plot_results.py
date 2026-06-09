"""Generate presentation plots from CUDA-Q QEC result CSVs."""

import argparse
import csv
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def read_csv_rows(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path.stem
                rows.append(row)
    return rows


def plot_steane(plt, rows, output_dir):
    grouped = {}
    for row in rows:
        if (
            row.get("code") == "steane"
            and row.get("raw_logical_error_rate")
            and row.get("decoded_logical_error_rate")
        ):
            grouped.setdefault(row["_source"], []).append(row)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["physical_error_probability"]))
        x = [float(row["physical_error_probability"]) for row in ordered]
        raw = [float(row["raw_logical_error_rate"]) for row in ordered]
        decoded = [float(row["decoded_logical_error_rate"]) for row in ordered]
        ax.plot(x, raw, marker="o", linestyle="--", label=f"{label} raw")
        ax.plot(x, decoded, marker="o", label=f"{label} decoded")

    ax.set(
        title="Steane Code Logical Error Rate",
        xlabel="Physical error probability",
        ylabel="Logical error rate",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = output_dir / "steane_logical_error_rates.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def decoder_label(row):
    decoder = row["decoder"].replace("single_error_lut", "LUT").replace("nv-qldpc-decoder", "QLDPC")
    label = f"d={row.get('distance', '?')}\n{decoder}"
    if row.get("bp_method"):
        label += f"\nbp={row['bp_method']}"
    return label


def plot_decoder(plt, rows, output_dir):
    decoder_rows = [
        row
        for row in rows
        if row.get("platform") and row.get("decoder") and row.get("syndromes_per_second")
    ]
    if not decoder_rows:
        return

    ordered = sorted(decoder_rows, key=lambda row: (int(row.get("distance", 0)), row["decoder"]))
    labels = [decoder_label(row) for row in ordered]
    values = [float(row["syndromes_per_second"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set(title="CUDA-Q QEC Decoder Throughput", ylabel="Syndromes per second")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt=lambda value: f"{value:,.0f}", padding=3)
    fig.tight_layout()

    output = output_dir / "decoder_throughput.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def plot_decoder_accuracy(plt, rows, output_dir):
    decoder_rows = [
        row
        for row in rows
        if row.get("decoder") and row.get("distance") and row.get("logical_error_rate")
    ]
    if not decoder_rows:
        return

    ordered = sorted(decoder_rows, key=lambda row: (int(row["distance"]), row["decoder"]))
    labels = [decoder_label(row) for row in ordered]
    values = [float(row["logical_error_rate"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set(title="CUDA-Q QEC Decoder Logical Error Rate", ylabel="Logical error rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt=lambda value: f"{value:.3g}", padding=3)
    fig.tight_layout()

    output = output_dir / "decoder_logical_error_rate.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(PROJECT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(results_dir)
    if not rows:
        sys.exit(f"FAIL: no CSV files found in {results_dir}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("FAIL: install matplotlib with: python -m pip install -r requirements.txt")

    outputs = [
        plot_steane(plt, rows, output_dir),
        plot_decoder(plt, rows, output_dir),
        plot_decoder_accuracy(plt, rows, output_dir),
    ]
    if not any(outputs):
        sys.exit("FAIL: no Steane or decoder benchmark rows found to plot")


if __name__ == "__main__":
    main()
