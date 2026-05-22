"""Medical-imaging hello-world for cuCIM.

Generates a noisy 2D image with one bright circular "lesion" region, then runs
a cuCIM/scikit-image-style GPU pipeline on it (Gaussian filter -> threshold ->
connected-component labeling) and reports how many regions were detected.
"""

import cupy as cp
from cucim.skimage import filters, measure


def main() -> None:
    height, width = 1024, 1024
    img = cp.random.normal(0, 0.05, (height, width), dtype=cp.float32)

    yy, xx = cp.ogrid[:height, :width]
    lesion_mask = (yy - height // 2) ** 2 + (xx - width // 2) ** 2 < 120 ** 2
    img[lesion_mask] += 1.0

    blurred = filters.gaussian(img, sigma=3)
    thresholded = blurred > 0.5
    labels = measure.label(thresholded)

    cp.cuda.Device().synchronize()

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    detected_regions = int(labels.max().get())

    print("Hello cuCIM")
    print(f"Input type: {type(img)}")
    print(f"GPU device: {gpu_name}")
    print(f"Image shape: {img.shape}")
    print(f"Detected regions: {detected_regions}")


if __name__ == "__main__":
    main()
