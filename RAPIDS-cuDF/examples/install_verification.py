"""End-to-end smoke test for a cuDF install.

Confirms cuDF imports, an NVIDIA GPU is visible, and a real DataFrame
op runs on the device. Exits non-zero on any failure so it's safe to
drop into CI or a setup script.
"""

import sys

# 1. CuPy comes with cuDF and gives us an easy way
#    to ask CUDA about the device.
try:
    import cudf
    import cupy as cp
except ImportError as e:
    sys.exit(f"FAIL: cuDF / CuPy not importable -> {e}")

print(f"cuDF version:  {cudf.__version__}")
print(f"CuPy version:  {cp.__version__}")

# 2. Checks if the driver actually loaded and that there is a GPU visible.
if cp.cuda.runtime.getDeviceCount() == 0:
    sys.exit("FAIL: no CUDA devices detected")

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
rt = cp.cuda.runtime.runtimeGetVersion()  # encoded as 1000*major + 10*minor

print(f"CUDA runtime:  {rt // 1000}.{(rt % 1000) // 10}")
print(f"GPU 0:         {name} (compute {props['major']}.{props['minor']})")

# 3. Prove that cuDF can actually compute on the device, not just import.
#    This kicks off an allocation, a kernel launch, and a host transfer.
df = cudf.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
total = int((df["x"] * df["y"]).sum())
assert total == 140, f"unexpected result: {total}"

print("PASS: cuDF is installed and the GPU workflow is working.")