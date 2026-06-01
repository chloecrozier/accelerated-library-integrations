"""Lightweight CPU vs GPU FFT benchmark using PyTorch.

The GPU path uses PyTorch's CUDA FFT implementation, which dispatches to cuFFT.
Timings exclude CPU<->GPU copies by creating data directly on each device.
"""

import statistics
import time

import torch


DEFAULT_SIZES = (
    256,
    300,
    512,
    1_000,
    1_024,
    4_096,
    5_000,
    16_384,
    20_000,
    65_536,
    100_000,
    262_144,
    1_000_000,
    1_048_576,
    4_194_304,
)
REPEATS = 50
WARMUPS = 3


def time_cpu_fft(x: torch.Tensor, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        torch.fft.rfft(x)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def time_gpu_fft(x: torch.Tensor, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.fft.rfft(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark_size(size: int, repeats: int, warmups: int) -> tuple[float, float]:
    cpu_data = torch.randn(size, dtype=torch.float32, device="cpu")
    gpu_data = torch.randn(size, dtype=torch.float32, device="cuda")

    for _ in range(warmups):
        torch.fft.rfft(cpu_data)
        torch.fft.rfft(gpu_data)
    torch.cuda.synchronize()

    return time_cpu_fft(cpu_data, repeats), time_gpu_fft(gpu_data, repeats)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; GPU/cuFFT benchmark cannot run.")

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("Times are median wall clock seconds and exclude CPU<->GPU copies.")
    print()
    print(f"{'n':>12} {'CPU FFT (s)':>14} {'GPU cuFFT (s)':>15} {'speedup':>10}")
    print("-" * 56)

    for size in DEFAULT_SIZES:
        cpu_time, gpu_time = benchmark_size(size, REPEATS, WARMUPS)
        speedup = cpu_time / gpu_time if gpu_time else float("inf")
        print(f"{size:12d} {cpu_time:14.6f} {gpu_time:15.6f} {speedup:9.2f}x")


if __name__ == "__main__":
    main()
