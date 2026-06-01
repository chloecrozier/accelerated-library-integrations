import ctypes
import time
import numpy as np
from numba import cuda

N = 2048

# Create input matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# ---------------- CPU Benchmark ----------------
start = time.perf_counter()
h_C_cpu = np.dot(A, B)
cpu_time = time.perf_counter() - start

print("CPU SGEMM SUCCESS")
print(f"CPU Time: {cpu_time:.4f} sec")

# ---------------- Load cuBLAS ----------------
libcublas = ctypes.cdll.LoadLibrary("libcublas.so")

handle = ctypes.c_void_p()
status = libcublas.cublasCreate_v2(ctypes.byref(handle))

if status != 0:
    print("cuBLAS initialization FAILED")
    exit(1)

print("cuBLAS initialization SUCCESS")

# ---------------- Copy Data to GPU ----------------
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((N, N), dtype=np.float32)

alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)

cuda.synchronize()

# ---------------- GPU Benchmark using cuBLAS ----------------
start = time.perf_counter()

status = libcublas.cublasSgemm_v2(
    handle,
    0,
    0,
    N,
    N,
    N,
    ctypes.byref(alpha),
    ctypes.c_void_p(d_B.device_ctypes_pointer.value),
    N,
    ctypes.c_void_p(d_A.device_ctypes_pointer.value),
    N,
    ctypes.byref(beta),
    ctypes.c_void_p(d_C.device_ctypes_pointer.value),
    N,
)

cuda.synchronize()
gpu_time = time.perf_counter() - start

if status != 0:
    print("cuBLAS SGEMM FAILED")
    libcublas.cublasDestroy_v2(handle)
    exit(1)

print("cuBLAS SGEMM SUCCESS")
print(f"GPU Time: {gpu_time:.4f} sec")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")

# ---------------- Accuracy Check ----------------
h_C_gpu = d_C.copy_to_host()

cpu_max_error = 0.0
gpu_max_error = np.max(np.abs(h_C_cpu - h_C_gpu))

print(f"CPU Max Error: {cpu_max_error}")
print(f"GPU Max Error: {gpu_max_error}")

# ---------------- Cleanup ----------------
libcublas.cublasDestroy_v2(handle)