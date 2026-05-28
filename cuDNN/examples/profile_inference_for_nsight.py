"""Small Nsight Systems target for visually comparing inference modes.

This script is intentionally simpler than benchmark_cudnn.py. It runs one
clearly labeled inference range per mode so the Nsight timeline is easy to
explain in a slide:

1. CPU inference
2. GPU inference with cuDNN disabled
3. GPU inference with cuDNN enabled
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torchvision.models import resnet18


@dataclass(frozen=True)
class ProfileMode:
    label: str
    device: torch.device
    cudnn_enabled: Optional[bool]
    cudnn_benchmark: Optional[bool]


@dataclass(frozen=True)
class InferenceRow:
    platform_label: str
    workload: str
    mode: str
    run: int
    device_name: str
    cudnn_version: Optional[int]
    batch_size: int
    wall_latency_ms: float
    cuda_event_ms: Optional[float]


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


@contextmanager
def nvtx_range(label: str) -> Iterable[None]:
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def build_conv2d(device: torch.device, batch_size: int) -> tuple[torch.nn.Module, torch.Tensor]:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
    ).to(device).eval()
    x = torch.randn(batch_size, 64, 112, 112, device=device)
    return model, x


def build_resnet18(device: torch.device, batch_size: int) -> tuple[torch.nn.Module, torch.Tensor]:
    model = resnet18(weights=None).to(device).eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    return model, x


def build_workload(workload: str, device: torch.device, batch_size: int) -> tuple[torch.nn.Module, torch.Tensor]:
    if workload == "conv2d":
        return build_conv2d(device, batch_size)
    if workload == "resnet18":
        return build_resnet18(device, batch_size)
    raise ValueError(f"unsupported workload: {workload}")


def run_forward(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[float, Optional[float]]:
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            model(x)
            end_event.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            return wall_ms, start_event.elapsed_time(end_event)

        wall_start = time.perf_counter()
        model(x)
        return (time.perf_counter() - wall_start) * 1000, None


def run_mode(
    platform_label: str,
    workload: str,
    mode: ProfileMode,
    batch_size: int,
    warmup: int,
    runs: int,
) -> list[InferenceRow]:
    rows = []
    with cudnn_settings(mode.cudnn_enabled, mode.cudnn_benchmark):
        model, x = build_workload(workload, mode.device, batch_size)

        for idx in range(warmup):
            with nvtx_range(f"WARMUP | {workload} | {mode.label} | {idx + 1}"):
                run_forward(model, x, mode.device)

        for idx in range(runs):
            range_label = f"MEASURE | {workload} | {mode.label} | inference {idx + 1}"
            with nvtx_range(range_label):
                wall_ms, cuda_event_ms = run_forward(model, x, mode.device)

            rows.append(
                InferenceRow(
                    platform_label=platform_label,
                    workload=workload,
                    mode=mode.label,
                    run=idx + 1,
                    device_name=device_name(mode.device),
                    cudnn_version=torch.backends.cudnn.version(),
                    batch_size=batch_size,
                    wall_latency_ms=wall_ms,
                    cuda_event_ms=cuda_event_ms,
                )
            )

    return rows


def write_csv(path: Path, rows: list[InferenceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(InferenceRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="local", help="Label for this run, e.g. brev_l4.")
    parser.add_argument("--workload", choices=["conv2d", "resnet18"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default="results/nsight_single_inference.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is required for the GPU comparison modes.")

    modes = [
        ProfileMode("01 CPU", torch.device("cpu"), None, None),
        ProfileMode("02 GPU, cuDNN disabled", torch.device("cuda:0"), False, False),
        ProfileMode("03 GPU, cuDNN enabled", torch.device("cuda:0"), True, False),
    ]

    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA build:    {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU 0:         {torch.cuda.get_device_name(0)}")
    print(f"Workload:      {args.workload}, batch_size={args.batch_size}")

    rows = []
    for mode in modes:
        rows.extend(
            run_mode(
                platform_label=args.platform,
                workload=args.workload,
                mode=mode,
                batch_size=args.batch_size,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    print()
    print(f"{'mode':<28} {'wall_ms':>10} {'cuda_ms':>10}")
    print("-" * 52)
    for row in rows:
        cuda_ms = "" if row.cuda_event_ms is None else f"{row.cuda_event_ms:.3f}"
        print(f"{row.mode:<28} {row.wall_latency_ms:>10.3f} {cuda_ms:>10}")

    write_csv(Path(args.output), rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
