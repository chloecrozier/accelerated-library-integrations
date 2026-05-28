"""End-to-end smoke test for a PyTorch + cuDNN install.

Confirms PyTorch imports, a CUDA device is visible, cuDNN is available through
PyTorch, and a small convolution runs on the GPU.
"""

from __future__ import annotations

import platform
import sys


try:
    import torch
except ImportError as exc:
    sys.exit(f"FAIL: PyTorch is not importable -> {exc}")


def main() -> None:
    print(f"Python:        {platform.python_version()}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA build:    {torch.version.cuda}")
    print(f"CUDA runtime:  {'available' if torch.cuda.is_available() else 'not available'}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN usable:  {torch.backends.cudnn.is_available()}")

    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA devices detected by PyTorch")

    if not torch.backends.cudnn.is_available():
        sys.exit("FAIL: cuDNN is not available through PyTorch")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU 0:         {props.name} (compute {props.major}.{props.minor})")

    model = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device).eval()
    x = torch.randn(8, 3, 224, 224, device=device)

    with torch.inference_mode():
        y = model(x)
        torch.cuda.synchronize()

    expected_shape = (8, 16, 224, 224)
    if tuple(y.shape) != expected_shape:
        sys.exit(f"FAIL: expected output shape {expected_shape}, got {tuple(y.shape)}")

    print("PASS: PyTorch can use CUDA and cuDNN for a GPU convolution.")


if __name__ == "__main__":
    main()
