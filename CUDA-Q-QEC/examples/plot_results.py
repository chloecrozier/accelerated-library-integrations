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


def benchmark_rows(rows):
    return [
        row
        for row in rows
        if row.get("platform") and row.get("decoder") and row.get("syndromes_per_second")
    ]


def decoder_rows(rows):
    current_rows = [row for row in benchmark_rows(rows) if row.get("variant")]
    return current_rows if current_rows else benchmark_rows(rows)


def cpu_gpu_rows(rows):
    return [row for row in rows if row.get("benchmark") == "cpu_gpu_syndrome"]


def surface_sweep_rows(rows):
    return [
        row
        for row in rows
        if row.get("_source", "").startswith("surface_sweep_")
        and row.get("logical_error_rate_with_decoding")
    ]


def surface_plot_title(rows):
    if all(int(row["rounds"]) == int(row["distance"]) for row in rows):
        return "Surface-Code Logical Error Rate, rounds=d"
    return "Surface-Code Logical Error Rate, fixed/custom rounds"


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


def plot_cpu_gpu_speedup(plt, rows, output_dir):
    comparison_rows = cpu_gpu_rows(rows)
    if not comparison_rows:
        return

    ordered = sorted(comparison_rows, key=lambda row: row["backend"])
    labels = [row["backend"].replace("_", " ").upper() for row in ordered]
    values = [float(row["syndromes_per_second"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set(title="CPU vs GPU QEC Syndrome Throughput", ylabel="Syndromes per second")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt=lambda value: f"{value:,.0f}", padding=3)
    fig.tight_layout()

    output = output_dir / "cpu_gpu_speedup.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def decoder_name(row):
    if row["decoder"] == "single_error_lut":
        return "LUT"
    if row["decoder"] == "nv-qldpc-decoder":
        bp = row.get("bp_method", "")
        return f"QLDPC BP={bp}" if bp else "QLDPC"
    return row["decoder"]


def plot_surface_combined(plt, rows, output_dir):
    sweep_rows = surface_sweep_rows(rows)
    if not sweep_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "single_error_lut": "--",
        "nv-qldpc-decoder": "-",
    }
    markers = {
        "single_error_lut": "s",
        "nv-qldpc-decoder": "o",
    }

    groups = sorted(
        {(row["decoder"], int(row["distance"])) for row in sweep_rows},
        key=lambda item: (item[0], item[1]),
    )
    saw_zero = False
    for decoder, distance in groups:
        group = sorted(
            [
                row
                for row in sweep_rows
                if row["decoder"] == decoder and int(row["distance"]) == distance
            ],
            key=lambda row: float(row["physical_error_probability"]),
        )
        x = [float(row["physical_error_probability"]) for row in group]
        y = []
        for row in group:
            rate_value = float(row["logical_error_rate_with_decoding"])
            if rate_value == 0:
                saw_zero = True
                rate_value = 0.5 / float(row["shots"])
            y.append(rate_value)
        label = f"{decoder_name(group[0])}, d={distance}"
        ax.plot(x, y, linestyle=styles.get(decoder, "-"), marker=markers.get(decoder, "o"), label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title(surface_plot_title(sweep_rows))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if saw_zero:
        ax.text(0.02, 0.02, "zero-count points shown at 0.5/shots", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()

    output = output_dir / "surface_sweep_combined.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def decoder_rate(row):
    return float(row.get("logical_error_rate") or row.get("decoded_logical_error_rate"))


def decoder_errors(row):
    return row.get("logical_errors") or row.get("decoded_logical_errors")


def decoder_speedup(row):
    return row.get("speedup_vs_lut") or 1.0


def variant_order(variant):
    if variant == "LUT":
        return 0
    if variant.startswith("BP="):
        try:
            return int(variant.split("=", 1)[1]) + 1
        except ValueError:
            return 99
    return 99


def plot_decoder_time_accuracy(plt, rows, output_dir):
    selected_rows = decoder_rows(rows)
    if not selected_rows:
        return

    distances = sorted({int(row["distance"]) for row in selected_rows})
    variants = sorted({row.get("variant") or row["decoder"] for row in selected_rows}, key=variant_order)
    by_key = {
        (int(row["distance"]), row.get("variant") or row["decoder"]): row
        for row in selected_rows
    }

    x = list(range(len(distances)))
    width = min(0.22, 0.8 / max(len(variants), 1))
    offsets = [(index - (len(variants) - 1) / 2) * width for index in range(len(variants))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for index, variant in enumerate(variants):
        positions = [value + offsets[index] for value in x]
        time_values = []
        rate_values = []
        for distance in distances:
            row = by_key.get((distance, variant))
            time_values.append(float(row["median_ms"]) if row else 0)
            rate_values.append(decoder_rate(row) if row else 0)
        axes[0].bar(positions, time_values, width, label=variant)
        axes[1].bar(positions, rate_values, width, label=variant)

    axes[0].set_title("Decoder Time")
    axes[0].set_ylabel("Median decode time (ms)")
    axes[1].set_title("Decoder Accuracy")
    axes[1].set_ylabel("Decoded logical error rate")
    for ax in axes:
        ax.set_xlabel("Surface-code distance")
        ax.set_xticks(x, [str(distance) for distance in distances])
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(title="Variant")
    fig.suptitle("Surface-Code Decoder Time and Accuracy")
    fig.tight_layout()

    output = output_dir / "decoder_time_accuracy.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def rate(value):
    return f"{float(value):.4g}"


def write_summary(rows, output_dir):
    lines = ["# CUDA-Q QEC Result Summary", ""]

    steane_rows = sorted(
        [row for row in rows if row.get("code") == "steane"],
        key=lambda row: float(row["physical_error_probability"]),
    )
    if steane_rows:
        lines += [
            "## Steane Code-Capacity Demo",
            "",
            "| p | raw errors | decoded errors | raw rate | decoded rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in steane_rows:
            lines.append(
                f"| {float(row['physical_error_probability']):.4f} "
                f"| {row['raw_logical_errors']} "
                f"| {row['decoded_logical_errors']} "
                f"| {rate(row['raw_logical_error_rate'])} "
                f"| {rate(row['decoded_logical_error_rate'])} |"
            )
        lines.append("")

    surface_rows = [
        row
        for row in rows
        if row.get("_source", "").startswith("surface_sweep_")
        and row.get("code") == "surface_code"
        and row.get("logical_error_rate_with_decoding")
        and not row.get("syndromes_per_second")
    ]
    if surface_rows:
        if all(int(row["rounds"]) == int(row["distance"]) for row in surface_rows):
            surface_note = (
                "This threshold-style sweep uses rounds=d, lower physical error rates, "
                "and more shots than the quick fixed-round demo."
            )
        else:
            surface_note = (
                "This sweep uses fixed/custom syndrome rounds, so it should not be read "
                "as a clean surface-code threshold estimate."
            )
        lines += [
            "## Surface-Code Sweep",
            "",
            surface_note,
            "",
            "It still may not match published threshold plots exactly because decoder choice, "
            "noise model, and shot count all affect the trend.",
            "",
            "| decoder | distance | rounds | p | shots | without decoding | with decoding |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(surface_rows, key=lambda row: (row["decoder"], int(row["distance"]), float(row["physical_error_probability"]))):
            lines.append(
                f"| {row['decoder']} "
                f"| {row['distance']} "
                f"| {row['rounds']} "
                f"| {float(row['physical_error_probability']):.4f} "
                f"| {row['shots']} "
                f"| {rate(row['logical_error_rate_without_decoding'])} "
                f"| {rate(row['logical_error_rate_with_decoding'])} |"
            )
        lines.append("")

    comparison_rows = cpu_gpu_rows(rows)
    if comparison_rows:
        lines += [
            "## CPU vs GPU Syndrome Benchmark",
            "",
            "| backend | median ms | syndromes/s | speedup vs CPU | scope |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for row in sorted(comparison_rows, key=lambda row: row["backend"]):
            lines.append(
                f"| {row['backend']} "
                f"| {float(row['median_ms']):.3f} "
                f"| {float(row['syndromes_per_second']):,.0f} "
                f"| {float(row['speedup_vs_cpu']):.2f}x "
                f"| {row.get('scope', '')} |"
            )
        lines.append("")

    selected_decoder_rows = decoder_rows(rows)
    if selected_decoder_rows:
        lines += [
            "## Decoder Benchmark Results",
            "",
            "| variant | decoder | distance | median ms | decoded errors | decoded rate | speed vs LUT | rate vs LUT |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(selected_decoder_rows, key=lambda row: (int(row["distance"]), variant_order(row.get("variant", row["decoder"])))):
            rate_ratio = row.get("logical_error_ratio_vs_lut", "")
            rate_ratio_text = f"{float(rate_ratio):.2f}x" if rate_ratio else ""
            lines.append(
                f"| {row.get('variant', row['decoder'])} "
                f"| {row['decoder']} "
                f"| {row['distance']} "
                f"| {float(row['median_ms']):.3f} "
                f"| {decoder_errors(row)}/{row['shots']} "
                f"| {rate(decoder_rate(row))} "
                f"| {float(decoder_speedup(row)):.2f}x "
                f"| {rate_ratio_text} |"
            )
        lines.append("")

        lines += ["## Decoder Comparison Takeaway", ""]
        by_distance = {}
        for row in selected_decoder_rows:
            by_distance.setdefault(row["distance"], {})[row.get("variant", row["decoder"])] = row
        for distance, group in sorted(by_distance.items(), key=lambda item: int(item[0])):
            lut = group.get("LUT")
            if not lut:
                continue
            for variant, row in sorted(group.items(), key=lambda item: variant_order(item[0])):
                if variant == "LUT":
                    continue
                speed_ratio = float(row["syndromes_per_second"]) / float(lut["syndromes_per_second"])
                rate_ratio = decoder_rate(row) / decoder_rate(lut) if decoder_rate(lut) else 0
                lines.append(
                    f"- d={distance}, {variant}: {speed_ratio:.2f}x LUT throughput "
                    f"and {rate_ratio:.2f}x LUT logical error rate."
                )
        lines.append("")

    output = output_dir / "SUMMARY.md"
    output.write_text("\n".join(lines))
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
        plot_cpu_gpu_speedup(plt, rows, output_dir),
        plot_surface_combined(plt, rows, output_dir),
        plot_decoder_time_accuracy(plt, rows, output_dir),
        write_summary(rows, output_dir),
    ]
    if not any(outputs):
        sys.exit("FAIL: no Steane or decoder benchmark rows found to plot")


if __name__ == "__main__":
    main()
