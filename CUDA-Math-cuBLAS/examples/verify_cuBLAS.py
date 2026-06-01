import ctypes
import time

start_total = time.perf_counter()

load_start = time.perf_counter()

libcublas = ctypes.cdll.LoadLibrary("libcublas.so")

load_time = time.perf_counter() - load_start

handle = ctypes.c_void_p()

init_start = time.perf_counter()

status = libcublas.cublasCreate_v2(ctypes.byref(handle))

init_time = time.perf_counter() - init_start

if status != 0:
    print("cuBLAS initialization FAILED")
    exit()

total_time = time.perf_counter() - start_total

print("\n========== cuBLAS Verification ==========")

print("Hello World from cuBLAS!")

print(f"\nLibrary Load Time   : {load_time:.6f} sec")
print(f"cuBLAS Init Time    : {init_time:.6f} sec")
print(f"Total Execution Time: {total_time:.6f} sec")

print("\ncuBLAS Status: SUCCESS")

libcublas.cublasDestroy_v2(handle)