# CUDA-Q QEC Result Summary

## Steane Code-Capacity Demo

| p | raw errors | decoded errors | raw rate | decoded rate |
| --- | ---: | ---: | ---: | ---: |
| 0.0010 | 2 | 0 | 0.002 | 0 |
| 0.0030 | 13 | 1 | 0.013 | 0.001 |
| 0.0100 | 34 | 5 | 0.034 | 0.005 |
| 0.0300 | 79 | 11 | 0.079 | 0.011 |
| 0.0500 | 134 | 40 | 0.134 | 0.04 |
| 0.1000 | 239 | 120 | 0.239 | 0.12 |

## Surface-Code Sweep

This circuit-level memory sweep uses rounds=d, lower physical error rates, and QLDPC decoding. It is the realistic integration demo.

It still may not match published threshold plots exactly because decoder choice, noise model, and shot count all affect the trend.

| decoder | distance | rounds | p | shots | without decoding | with decoding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nv-qldpc-decoder | 3 | 3 | 0.0005 | 10000 | 0.0067 | 0.0034 |
| nv-qldpc-decoder | 3 | 3 | 0.0010 | 10000 | 0.0134 | 0.0052 |
| nv-qldpc-decoder | 3 | 3 | 0.0020 | 10000 | 0.0209 | 0.0127 |
| nv-qldpc-decoder | 3 | 3 | 0.0030 | 10000 | 0.0336 | 0.0169 |
| nv-qldpc-decoder | 3 | 3 | 0.0050 | 10000 | 0.059 | 0.0349 |
| nv-qldpc-decoder | 3 | 3 | 0.0070 | 10000 | 0.0852 | 0.0475 |
| nv-qldpc-decoder | 3 | 3 | 0.0100 | 10000 | 0.1152 | 0.0701 |
| nv-qldpc-decoder | 5 | 5 | 0.0005 | 10000 | 0.0193 | 0.0055 |
| nv-qldpc-decoder | 5 | 5 | 0.0010 | 10000 | 0.0382 | 0.0103 |
| nv-qldpc-decoder | 5 | 5 | 0.0020 | 10000 | 0.0704 | 0.0231 |
| nv-qldpc-decoder | 5 | 5 | 0.0030 | 10000 | 0.1083 | 0.0396 |
| nv-qldpc-decoder | 5 | 5 | 0.0050 | 10000 | 0.1625 | 0.0741 |
| nv-qldpc-decoder | 5 | 5 | 0.0070 | 10000 | 0.2153 | 0.1104 |
| nv-qldpc-decoder | 5 | 5 | 0.0100 | 10000 | 0.271 | 0.1678 |
| nv-qldpc-decoder | 7 | 7 | 0.0005 | 10000 | 0.0412 | 0.0066 |
| nv-qldpc-decoder | 7 | 7 | 0.0010 | 10000 | 0.078 | 0.015 |
| nv-qldpc-decoder | 7 | 7 | 0.0020 | 10000 | 0.1385 | 0.0342 |
| nv-qldpc-decoder | 7 | 7 | 0.0030 | 10000 | 0.1907 | 0.0597 |
| nv-qldpc-decoder | 7 | 7 | 0.0050 | 10000 | 0.2752 | 0.1133 |
| nv-qldpc-decoder | 7 | 7 | 0.0070 | 10000 | 0.333 | 0.174 |
| nv-qldpc-decoder | 7 | 7 | 0.0100 | 10000 | 0.4068 | 0.2686 |

## CPU vs GPU Syndrome Benchmark

| backend | median ms | syndromes/s | speedup vs CPU | scope |
| --- | ---: | ---: | ---: | --- |
| cpu_numpy | 324.470 | 308,195 | 1.00x | compute_only |
| gpu_cupy | 2.867 | 34,882,013 | 113.18x | compute_only |

## Decoder Benchmark Results

| variant | decoder | distance | median ms | decoded errors | decoded rate | speed vs LUT | rate vs LUT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUT | single_error_lut | 3 | 3.412 | 13/2000 | 0.0065 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 3 | 4.608 | 13/2000 | 0.0065 | 0.74x | 1.00x |
| LUT | single_error_lut | 5 | 10.604 | 25/2000 | 0.0125 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 5 | 18.622 | 19/2000 | 0.0095 | 0.57x | 0.76x |
| LUT | single_error_lut | 7 | 27.500 | 108/2000 | 0.054 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 7 | 49.216 | 31/2000 | 0.0155 | 0.56x | 0.29x |
| LUT | single_error_lut | 9 | 56.414 | 195/2000 | 0.0975 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 9 | 116.476 | 49/2000 | 0.0245 | 0.48x | 0.25x |
| LUT | single_error_lut | 11 | 100.097 | 338/2000 | 0.169 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 11 | 238.899 | 61/2000 | 0.0305 | 0.42x | 0.18x |

## Decoder Comparison Takeaway

- d=3, BP=0: 0.74x LUT throughput and 1.00x LUT logical error rate.
- d=5, BP=0: 0.57x LUT throughput and 0.76x LUT logical error rate.
- d=7, BP=0: 0.56x LUT throughput and 0.29x LUT logical error rate.
- d=9, BP=0: 0.48x LUT throughput and 0.25x LUT logical error rate.
- d=11, BP=0: 0.42x LUT throughput and 0.18x LUT logical error rate.
