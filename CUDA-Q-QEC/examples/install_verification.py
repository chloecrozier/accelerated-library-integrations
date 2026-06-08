"""End-to-end smoke test for a CUDA-Q QEC install.

Confirms CUDA-Q QEC imports, the Steane code loads, and a small syndrome
decode completes. This is the first script to run after creating the
environment.
"""

import platform
import sys


try:
    import numpy as np
    import cudaq
    import cudaq_qec as qec
except ImportError as exc:
    sys.exit(
        "FAIL: CUDA-Q QEC dependencies are not importable. Install with:\n"
        "    python -m pip install -r requirements.txt\n"
        f"Import error: {exc}"
    )


print(f"Python:    {platform.python_version()}")
print(f"cudaq:     {getattr(cudaq, '__version__', 'unknown')}")
print(f"cudaq_qec: {getattr(qec, '__version__', 'unknown')}")

code = qec.get_code("steane")
hz = np.asarray(code.get_parity_z(), dtype=np.uint8)
print(f"Steane Hz: {hz.shape[0]} x {hz.shape[1]}")

decoder = qec.get_decoder("single_error_lut", hz)
syndrome = np.zeros(hz.shape[0], dtype=np.uint8)
result = decoder.decode(syndrome)
correction = np.asarray(result.result, dtype=np.uint8)

if correction.shape[0] != hz.shape[1]:
    sys.exit(
        f"FAIL: expected correction length {hz.shape[1]}, got {correction.shape[0]}"
    )

if int(correction.sum()) != 0:
    sys.exit(f"FAIL: zero syndrome produced nonzero correction: {correction}")

print("PASS: CUDA-Q QEC is installed and a Steane syndrome decode completed.")
