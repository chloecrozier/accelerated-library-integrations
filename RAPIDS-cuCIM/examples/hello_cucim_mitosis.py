"""cuCIM hello world on a real microscopy sample of cells in mitosis.

Uses `skimage.data.human_mitosis()` (a 512x512 grayscale fluorescence-style
crop of human cells with several in mitosis) and runs a GPU nuclei
segmentation:

    Gaussian denoise -> Otsu threshold (nuclei are BRIGHT against a dark
    background, so we keep `blurred > otsu`) -> remove_small_objects ->
    connected components.

Reports the GPU device, the Otsu threshold value, and the nucleus count.
"""

import cupy as cp
from cucim.skimage import filters, measure, morphology
from skimage.data import human_mitosis


def main() -> None:
    img_cpu = human_mitosis()
    img = cp.asarray(img_cpu, dtype=cp.float32) / 255.0

    blurred = filters.gaussian(img, sigma=1.5)
    thr = filters.threshold_otsu(blurred)
    mask = blurred > thr
    mask = morphology.remove_small_objects(mask, min_size=30)
    labels = measure.label(mask)

    cp.cuda.Device().synchronize()

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    detected = int(labels.max().get())
    thr_val = float(thr.get()) if hasattr(thr, "get") else float(thr)

    print("Hello cuCIM (real microscopy: human_mitosis)")
    print(f"Input type: {type(img)}")
    print(f"GPU device: {gpu_name}")
    print(f"Image shape: {img.shape}")
    print(f"Otsu threshold: {thr_val:.4f}")
    print(f"Detected nuclei: {detected}")


if __name__ == "__main__":
    main()
