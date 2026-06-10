"""Run the standard CUDA-Q QEC demo workflow."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def run(command):
    print()
    print("$ " + " ".join(command))
    subprocess.run([sys.executable, *command], cwd=PROJECT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-verify", action="store_true", help="skip install_verification.py")
    parser.add_argument("--skip-cpu-gpu", action="store_true", help="skip the CPU/GPU speedup benchmark")
    parser.add_argument("--skip-sweep", action="store_true", help="skip surface-code sweeps")
    parser.add_argument("--skip-qldpc", action="store_true", help="skip QLDPC benchmark and sweep")
    parser.add_argument("--full-surface-sweep", action="store_true", help="include distances 9 and 11")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_verify:
        run(["examples/install_verification.py"])
    run(["examples/hello_syndrome.py"])
    run(["examples/surface_memory.py"])
    if not args.skip_cpu_gpu:
        run(["examples/cpu_gpu_benchmark.py"])
    if not args.skip_sweep:
        sweep_command = ["examples/surface_sweep.py"]
        if args.full_surface_sweep:
            sweep_command += ["--distances", "3", "5", "7", "9", "11"]
        run(sweep_command)
        if not args.skip_qldpc:
            qldpc_sweep_command = ["examples/surface_sweep.py", "--decoder", "nv-qldpc-decoder"]
            if args.full_surface_sweep:
                qldpc_sweep_command += ["--distances", "3", "5", "7", "9", "11"]
            run(qldpc_sweep_command)
    if not args.skip_qldpc:
        run(["examples/decoder_benchmark.py"])
    run(["examples/plot_results.py"])

    print()
    print("Done. Main outputs are in results/:")
    print("- steane_logical_error_rates.png")
    print("- cpu_gpu_speedup.png")
    print("- surface_sweep_lut_brev_l4.png")
    print("- surface_sweep_qldpc_brev_l4.png")
    print("- surface_sweep_combined.png")
    print("- decoder_lut_bp_sweep_brev_l4.csv")
    print("- decoder_time_accuracy.png")
    print("- SUMMARY.md")


if __name__ == "__main__":
    main()
