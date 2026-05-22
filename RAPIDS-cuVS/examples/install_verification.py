"""End-to-end smoke test for a cuVS install.

Confirms cuVS imports, an NVIDIA GPU is visible, and a small brute-force
nearest-neighbor search runs on the device. This avoids model downloads so it
is safe to run first when setting up the environment.
"""

import sys

try:
    import cupy as cp
    import cuvs
    from cuvs.neighbors import brute_force
except ImportError as e:
    sys.exit(f"FAIL: cuVS / CuPy not importable -> {e}")

print(f"cuVS version:  {getattr(cuvs, '__version__', 'unknown')}")
print(f"CuPy version:  {cp.__version__}")

if cp.cuda.runtime.getDeviceCount() == 0:
    sys.exit("FAIL: no CUDA devices detected")

props = cp.cuda.runtime.getDeviceProperties(0)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
rt = cp.cuda.runtime.runtimeGetVersion()

print(f"CUDA runtime:  {rt // 1000}.{(rt % 1000) // 10}")
print(f"GPU 0:         {name} (compute {props['major']}.{props['minor']})")

# Three tiny toy "molecule embeddings" with obvious nearest neighbors.
dataset = cp.asarray(
    [
        [1.00, 0.00, 0.00, 0.00],
        [0.95, 0.05, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
    ],
    dtype=cp.float32,
)
queries = cp.asarray(
    [
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
    ],
    dtype=cp.float32,
)

index = brute_force.build(dataset, metric="cosine")
distances, neighbors = brute_force.search(index, queries, k=2)
cp.cuda.runtime.deviceSynchronize()

neighbors_host = cp.asarray(neighbors).get()
distances_host = cp.asarray(distances).get()

assert neighbors_host.shape == (2, 2), f"unexpected neighbor shape: {neighbors_host.shape}"
assert distances_host.shape == (2, 2), f"unexpected distance shape: {distances_host.shape}"
assert int(neighbors_host[0, 0]) == 0, f"unexpected first query neighbor: {neighbors_host[0]}"
assert int(neighbors_host[1, 0]) == 2, f"unexpected second query neighbor: {neighbors_host[1]}"

print("PASS: cuVS is installed and GPU nearest-neighbor search is working.")
