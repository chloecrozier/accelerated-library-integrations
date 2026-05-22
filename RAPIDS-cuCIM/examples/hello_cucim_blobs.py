"""cuCIM hello world on a synthetic binary-blob field.

Generates an irregular 512x512 random-walk blob field with scikit-image's
`binary_blobs`, transfers it to GPU, adds Gaussian noise, then runs the
standard cuCIM pipeline:

    Gaussian denoise -> threshold (> 0.5) -> remove_small_objects -> label

Reports the GPU device and the number of connected components detected.
"""

import cupy as cp
import numpy as np
from cucim.skimage import filters, measure, morphology
from skimage.data import binary_blobs


def main() -> None:
    blobs_cpu = binary_blobs(
        length=512,
        blob_size_fraction=0.08,
        n_dim=2,
        volume_fraction=0.25,
        rng=42,
    ).astype(np.float32)
    img = cp.asarray(blobs_cpu)
    img += cp.random.normal(0, 0.12, img.shape, dtype=cp.float32)

    blurred = filters.gaussian(img, sigma=1.2)
    mask = blurred > 0.5
    mask = morphology.remove_small_objects(mask, min_size=20)
    labels = measure.label(mask)

    cp.cuda.Device().synchronize()

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    detected = int(labels.max().get())

    print("Hello cuCIM (synthetic binary blobs)")
    print(f"Input type: {type(img)}")
    print(f"GPU device: {gpu_name}")
    print(f"Image shape: {img.shape}")
    print(f"Detected regions: {detected}")


if __name__ == "__main__":
    main()
