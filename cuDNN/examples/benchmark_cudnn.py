"""Benchmark cuDNN-backed deep learning primitives through PyTorch.

The benchmark intentionally uses standard PyTorch modules. That is the point:
most product teams integrate cuDNN by using a CUDA-enabled framework build, not
by calling cuDNN directly.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

try:
    from torchvision.models import resnet18
except ImportError as exc:
    sys.exit(f"FAIL: torchvision is required for the ResNet-18 benchmark -> {exc}")


@dataclass(frozen=True)
class BenchmarkMode:
    label: str
    device: torch.device
    cudnn_enabled: Optional[bool]
    cudnn_benchmark: Optional[bool]


@dataclass(frozen=True)
class BenchmarkResult:
    platform_label: str
    benchmark: str
    mode: str
    device_name: str
    cudnn_version: Optional[int]
    batch_size: int
    latency_ms: float
    throughput_img_s: float
    speedup_vs_cpu: float


@contextmanager
def cudnn_settings(enabled: Optional[bool], benchmark: Optional[bool]) -> Iterable[None]:
    old_enabled = torch.backends.cudnn.enabled
    old_benchmark = torch.backends.cudnn.benchmark
    if enabled is not None:
        torch.backends.cudnn.enabled = enabled
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = old_enabled
        torch.backends.cudnn.benchmark = old_benchmark


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "CPU"


def time_callable(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(iters):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))
            return statistics.median(samples)

        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)


def build_conv2d(device: torch.device, batch_size: int) -> tuple[Callable[[], torch.Tensor], int]:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
        torch.nn.ReLU(inplace=True),
    ).to(device).eval()
    x = torch.randn(batch_size, 64, 112, 112, device=device)
    return lambda: model(x), batch_size


def build_resnet18(device: torch.device, batch_size: int) -> tuple[Callable[[], torch.Tensor], int]:
    # Random weights are enough for a kernel/latency benchmark and avoid a network
    # dependency for downloading pretrained weights.
    model = resnet18(weights=None).to(device).eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    return lambda: model(x), batch_size


def available_modes() -> list[BenchmarkMode]:
    modes = [
        BenchmarkMode("CPU baseline", torch.device("cpu"), None, None),
    ]

    if torch.cuda.is_available():
        cuda = torch.device("cuda:0")
        modes.extend(
            [
                BenchmarkMode("GPU, cuDNN disabled", cuda, False, False),
                BenchmarkMode("GPU, cuDNN enabled", cuda, True, False),
                BenchmarkMode("GPU, cuDNN enabled + autotune", cuda, True, True),
            ]
        )

    return modes


def run_one_benchmark(
    platform_label: str,
    benchmark_name: str,
    builder: Callable[[torch.device, int], tuple[Callable[[], torch.Tensor], int]],
    modes: list[BenchmarkMode],
    batch_size: int,
    warmup: int,
    iters: int,
) -> list[BenchmarkResult]:
    rows = []
    cpu_latency = None

    for mode in modes:
        with cudnn_settings(mode.cudnn_enabled, mode.cudnn_benchmark):
            fn, images_per_iter = builder(mode.device, batch_size)
            latency_ms = time_callable(fn, mode.device, warmup, iters)

        if mode.device.type == "cpu":
            cpu_latency = latency_ms

        speedup = (cpu_latency / latency_ms) if cpu_latency else 1.0
        throughput = images_per_iter * 1000 / latency_ms

        rows.append(
            BenchmarkResult(
                platform_label=platform_label,
                benchmark=benchmark_name,
                mode=mode.label,
                device_name=device_name(mode.device),
                cudnn_version=torch.backends.cudnn.version(),
                batch_size=batch_size,
                latency_ms=latency_ms,
                throughput_img_s=throughput,
                speedup_vs_cpu=speedup,
            )
        )

    return rows


def write_csv(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_table(rows: list[BenchmarkResult]) -> None:
    print()
    print(f"{'benchmark':<12} {'mode':<32} {'latency_ms':>12} {'img/s':>12} {'speedup':>10}")
    print("-" * 82)
    for row in rows:
        print(
            f"{row.benchmark:<12} {row.mode:<32} "
            f"{row.latency_ms:>12.3f} {row.throughput_img_s:>12.1f} "
            f"{row.speedup_vs_cpu:>10.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="local", help="Label for this run, e.g. brev_l4 or dgx_spark.")
    parser.add_argument("--output", default="results/local.csv", help="CSV output path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Conv2d and ResNet-18.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per benchmark mode.")
    parser.add_argument("--iters", type=int, default=30, help="Measured iterations per benchmark mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = available_modes()

    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA build:    {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        print(f"GPU 0:         {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available; only the CPU baseline will run.")

    rows = []
    for benchmark_name, builder in [
        ("conv2d", build_conv2d),
        ("resnet18", build_resnet18),
    ]:
        rows.extend(
            run_one_benchmark(
                platform_label=args.platform,
                benchmark_name=benchmark_name,
                builder=builder,
                modes=modes,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    print_table(rows)
    write_csv(Path(args.output), rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
