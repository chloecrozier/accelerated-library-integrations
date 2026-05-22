"""Mini digital pathology preprocessing workflow with cuCIM.

This is not a diagnostic model. It is a small demo that shows where cuCIM fits:
image tile -> denoise/filter -> threshold -> connected components -> region count.

Pass --tiles N and --size S to scale the workload. The script reports total
elapsed preprocessing time and throughput in tiles/second on the active GPU.
"""

import argparse
import time

import cupy as cp
from cucim.skimage import filters, measure, morphology


def make_synthetic_tiles(num_tiles: int, size: int) -> cp.ndarray:
    """Create synthetic pathology-like tiles on the GPU."""
    tiles = cp.random.normal(0, 0.04, (num_tiles, size, size), dtype=cp.float32)
    yy, xx = cp.ogrid[:size, :size]

    for i in range(num_tiles):
        cy = size // 2 + int((i % 5) - 2) * 8
        cx = size // 2 + int((i % 7) - 3) * 8
        radius = size // 8
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
        tiles[i][mask] += 1.0

    return tiles


def process_tiles(num_tiles: int, size: int) -> tuple[float, int]:
    tiles = make_synthetic_tiles(num_tiles, size)

    start = time.perf_counter()
    total_regions = 0

    for i in range(num_tiles):
        blurred = filters.gaussian(tiles[i], sigma=2)
        tissue_mask = blurred > 0.35
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=32)
        labels = measure.label(tissue_mask)
        total_regions += int(labels.max().get())

    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, total_regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=64)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    elapsed, regions = process_tiles(args.tiles, args.size)

    print("cuCIM First Bowl of Soup demo")
    print(f"GPU device: {gpu_name}")
    print(f"Tiles processed: {args.tiles}")
    print(f"Tile size: {args.size} x {args.size}")
    print(f"Detected regions: {regions}")
    print(f"Elapsed preprocessing time: {elapsed:.3f} seconds")
    print(f"Throughput: {args.tiles / elapsed:.2f} tiles/second")


if __name__ == "__main__":
    main()
