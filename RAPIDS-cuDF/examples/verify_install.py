"""Verify cuDF is installed and print the library version."""

import cudf

print(f"cuDF version: {cudf.__version__}")
df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df)
print("Install verified.")
