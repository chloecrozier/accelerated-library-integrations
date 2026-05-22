"""End-to-end smoke test for a cuCIM install.

Confirms:
1. cuCIM imports.
2. CuPy imports.
3. A CUDA device is visible.
4. A small cuCIM image-processing operation runs on the GPU.

This script exits nonzero on failure, so it is safe to use in setup docs or CI.
"""

import sys

try:
    import cupy as cp
    import cucim
    from cucim.skimage import filters, measure
except ImportError as exc:
    sys.exit(f"FAIL: cuCIM / CuPy not importable -> {exc}")


def main() -> None:
    print(f"cuCIM version: {getattr(cucim, '__version__', 'unknown')}")
    print(f"CuPy version: {cp.__version__}")

    if cp.cuda.runtime.getDeviceCount() == 0:
        sys.exit("FAIL: no CUDA devices detected")

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    rt = cp.cuda.runtime.runtimeGetVersion()

    print(f"CUDA runtime: {rt // 1000}.{(rt % 1000) // 10}")
    print(f"GPU 0: {name} (compute {props['major']}.{props['minor']})")

    img = cp.zeros((128, 128), dtype=cp.float32)
    img[40:88, 40:88] = 1.0

    blurred = filters.gaussian(img, sigma=2)
    labels = measure.label(blurred > 0.25)
    regions = int(labels.max().get())

    assert regions == 1, f"unexpected region count: {regions}"
    print("PASS: cuCIM is installed and the GPU image workflow is working.")


if __name__ == "__main__":
    main()
