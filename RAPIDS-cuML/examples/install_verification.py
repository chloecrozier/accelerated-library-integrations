"""End-to-end smoke test for a RAPIDS cuML install.

Confirms cuDF/cuML import, a CUDA device is visible, and a cuML estimator can
fit and predict on GPU-backed data.
"""

import sys


try:
    import cudf
    import cuml
    import cupy as cp
    from cuml.cluster import KMeans
    from cuml.datasets import make_blobs
except ImportError as exc:
    sys.exit(f"FAIL: RAPIDS cuDF/cuML/CuPy not importable -> {exc}")


print(f"cuDF version:  {cudf.__version__}")
print(f"cuML version:  {cuml.__version__}")
print(f"CuPy version:  {cp.__version__}")

if cp.cuda.runtime.getDeviceCount() == 0:
    sys.exit("FAIL: no CUDA devices detected")

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
runtime = cp.cuda.runtime.runtimeGetVersion()

print(f"CUDA runtime:  {runtime // 1000}.{(runtime % 1000) // 10}")
print(f"GPU 0:         {name} (compute {props['major']}.{props['minor']})")

X, _ = make_blobs(
    n_samples=128,
    centers=3,
    n_features=4,
    random_state=7,
)

model = KMeans(n_clusters=3, random_state=7)
labels = model.fit_predict(X)

if len(labels) != 128:
    sys.exit(f"FAIL: expected 128 labels, got {len(labels)}")

print("PASS: cuML is installed and a GPU KMeans workflow completed.")
