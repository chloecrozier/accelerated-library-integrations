"""cuCIM hello world on a real IHC (DAB) stained tissue image.

Uses `skimage.data.immunohistochemistry()` (a 512x512 RGB sample of human
prostate tissue, DAB stained with hematoxylin counterstain) and isolates
the hematoxylin (nuclei) channel via `color.rgb2hed` on the GPU, then:

    Gaussian denoise -> Otsu threshold (> thr: hematoxylin is positive
    where nuclei are) -> remove_small_objects -> connected components

Reports the GPU device, the Otsu threshold on the hematoxylin channel,
and the detected nucleus count.
"""

import cupy as cp
from cucim.skimage import color, filters, measure, morphology
from skimage.data import immunohistochemistry


def main() -> None:
    img_cpu = immunohistochemistry()
    img = cp.asarray(img_cpu, dtype=cp.float32) / 255.0

    hed = color.rgb2hed(img)
    nuclei = hed[..., 0]

    blurred = filters.gaussian(nuclei, sigma=1.5)
    thr = filters.threshold_otsu(blurred)
    mask = blurred > thr
    mask = morphology.remove_small_objects(mask, min_size=40)
    labels = measure.label(mask)

    cp.cuda.Device().synchronize()

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    detected = int(labels.max().get())
    thr_val = float(thr.get()) if hasattr(thr, "get") else float(thr)

    print("Hello cuCIM (IHC color: rgb2hed -> hematoxylin)")
    print(f"Input type: {type(img)}  shape={img.shape}")
    print(f"GPU device: {gpu_name}")
    print(f"Otsu threshold (hematoxylin): {thr_val:.4f}")
    print(f"Detected nuclei: {detected}")


if __name__ == "__main__":
    main()
