import ctypes
import time
import numpy as np
from numba import cuda

# ---------------- Configuration ----------------
N = 2048
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50

print("cuBLAS Multi-Operation GEMM Benchmark")
print(f"Matrix Size: {N} x {N}")
print(f"Warm-up Runs: {WARMUP_RUNS}")
print(f"Benchmark Runs: {BENCHMARK_RUNS}")

# ---------------- Create Input Matrices ----------------
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# ---------------- CPU Benchmark ----------------
cpu_times = []

for i in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    h_C_cpu = np.dot(A, B)
    cpu_times.append(time.perf_counter() - start)

avg_cpu_time = sum(cpu_times) / BENCHMARK_RUNS

print("\nCPU SGEMM SUCCESS")
print(f"Average CPU Time: {avg_cpu_time:.6f} sec")

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

# ---------------- Warm-up Runs ----------------
for i in range(WARMUP_RUNS):
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

# ---------------- GPU Benchmark Runs ----------------
gpu_times = []

for i in range(BENCHMARK_RUNS):
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
    gpu_times.append(time.perf_counter() - start)

    if status != 0:
        print("cuBLAS SGEMM FAILED")
        libcublas.cublasDestroy_v2(handle)
        exit(1)

avg_gpu_time = sum(gpu_times) / BENCHMARK_RUNS

print("\ncuBLAS SGEMM SUCCESS")
print(f"Average GPU Time: {avg_gpu_time:.6f} sec")
print(f"Speedup: {avg_cpu_time / avg_gpu_time:.2f}x")

# ---------------- Accuracy Check ----------------
h_C_gpu = d_C.copy_to_host()

cpu_max_error = 0.0
gpu_max_error = np.max(np.abs(h_C_cpu - h_C_gpu))

print("\nAccuracy Check")
print(f"CPU Max Error: {cpu_max_error}")
print(f"GPU Max Error: {gpu_max_error}")

# ---------------- Extra Stats ----------------
print("\nTiming Summary")
print(f"CPU Min Time: {min(cpu_times):.6f} sec")
print(f"CPU Max Time: {max(cpu_times):.6f} sec")
print(f"GPU Min Time: {min(gpu_times):.6f} sec")
print(f"GPU Max Time: {max(gpu_times):.6f} sec")

# ---------------- Cleanup ----------------
libcublas.cublasDestroy_v2(handle)