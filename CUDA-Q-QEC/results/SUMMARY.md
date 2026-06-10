# CUDA-Q QEC Result Summary

## Steane Code-Capacity Demo

| p | raw errors | decoded errors | raw rate | decoded rate |
| --- | ---: | ---: | ---: | ---: |
| 0.0010 | 2 | 0 | 0.002 | 0 |
| 0.0030 | 7 | 0 | 0.007 | 0 |
| 0.0100 | 30 | 3 | 0.03 | 0.003 |
| 0.0300 | 92 | 20 | 0.092 | 0.02 |
| 0.0500 | 129 | 40 | 0.129 | 0.04 |
| 0.1000 | 247 | 141 | 0.247 | 0.141 |

## Surface-Code Sweep

This circuit-level memory sweep uses rounds=d, lower physical error rates, and QLDPC decoding. It is the realistic integration demo.

It still may not match published threshold plots exactly because decoder choice, noise model, and shot count all affect the trend.

| decoder | distance | rounds | p | shots | without decoding | with decoding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nv-qldpc-decoder | 3 | 3 | 0.0005 | 10000 | 0.0063 | 0.0036 |
| nv-qldpc-decoder | 3 | 3 | 0.0010 | 10000 | 0.0125 | 0.0058 |
| nv-qldpc-decoder | 3 | 3 | 0.0020 | 10000 | 0.0239 | 0.0115 |
| nv-qldpc-decoder | 3 | 3 | 0.0030 | 10000 | 0.0374 | 0.02 |
| nv-qldpc-decoder | 3 | 3 | 0.0050 | 10000 | 0.0615 | 0.0324 |
| nv-qldpc-decoder | 3 | 3 | 0.0070 | 10000 | 0.0821 | 0.0506 |
| nv-qldpc-decoder | 3 | 3 | 0.0100 | 10000 | 0.1109 | 0.0683 |
| nv-qldpc-decoder | 5 | 5 | 0.0005 | 10000 | 0.0179 | 0.0037 |
| nv-qldpc-decoder | 5 | 5 | 0.0010 | 10000 | 0.0414 | 0.0102 |
| nv-qldpc-decoder | 5 | 5 | 0.0020 | 10000 | 0.0754 | 0.0238 |
| nv-qldpc-decoder | 5 | 5 | 0.0030 | 10000 | 0.1017 | 0.0409 |
| nv-qldpc-decoder | 5 | 5 | 0.0050 | 10000 | 0.1636 | 0.071 |
| nv-qldpc-decoder | 5 | 5 | 0.0070 | 10000 | 0.2101 | 0.1099 |
| nv-qldpc-decoder | 5 | 5 | 0.0100 | 10000 | 0.2746 | 0.1736 |
| nv-qldpc-decoder | 7 | 7 | 0.0005 | 10000 | 0.0418 | 0.0064 |
| nv-qldpc-decoder | 7 | 7 | 0.0010 | 10000 | 0.0764 | 0.016 |
| nv-qldpc-decoder | 7 | 7 | 0.0020 | 10000 | 0.1427 | 0.0342 |
| nv-qldpc-decoder | 7 | 7 | 0.0030 | 10000 | 0.1952 | 0.0594 |
| nv-qldpc-decoder | 7 | 7 | 0.0050 | 10000 | 0.2806 | 0.1162 |
| nv-qldpc-decoder | 7 | 7 | 0.0070 | 10000 | 0.3484 | 0.1831 |
| nv-qldpc-decoder | 7 | 7 | 0.0100 | 10000 | 0.4007 | 0.2774 |

## Surface-Code Code-Capacity Sweep

This cleaner accuracy sweep applies direct data errors instead of full circuit-level noise, so it better isolates how code distance and decoder choice affect logical error rate.

| decoder | distance | p | shots | raw rate | decoded rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| nv-qldpc-decoder | 3 | 0.0001 | 100000 | 0.00032 | 0.0002 |
| nv-qldpc-decoder | 3 | 0.0002 | 100000 | 0.0005 | 0.00036 |
| nv-qldpc-decoder | 3 | 0.0005 | 100000 | 0.0016 | 0.00115 |
| nv-qldpc-decoder | 3 | 0.0010 | 100000 | 0.00333 | 0.00225 |
| nv-qldpc-decoder | 3 | 0.0020 | 100000 | 0.00605 | 0.00427 |
| nv-qldpc-decoder | 3 | 0.0030 | 100000 | 0.00907 | 0.00614 |
| nv-qldpc-decoder | 3 | 0.0050 | 100000 | 0.01556 | 0.01069 |
| nv-qldpc-decoder | 5 | 0.0001 | 100000 | 0.00055 | 0.00046 |
| nv-qldpc-decoder | 5 | 0.0002 | 100000 | 0.00097 | 0.00078 |
| nv-qldpc-decoder | 5 | 0.0005 | 100000 | 0.00251 | 0.00205 |
| nv-qldpc-decoder | 5 | 0.0010 | 100000 | 0.00494 | 0.00398 |
| nv-qldpc-decoder | 5 | 0.0020 | 100000 | 0.00921 | 0.0073 |
| nv-qldpc-decoder | 5 | 0.0030 | 100000 | 0.01418 | 0.01132 |
| nv-qldpc-decoder | 5 | 0.0050 | 100000 | 0.02445 | 0.01943 |
| nv-qldpc-decoder | 7 | 0.0001 | 100000 | 0.00089 | 0.0008 |
| nv-qldpc-decoder | 7 | 0.0002 | 100000 | 0.00137 | 0.00115 |
| nv-qldpc-decoder | 7 | 0.0005 | 100000 | 0.00334 | 0.00297 |
| nv-qldpc-decoder | 7 | 0.0010 | 100000 | 0.00662 | 0.00574 |
| nv-qldpc-decoder | 7 | 0.0020 | 100000 | 0.01408 | 0.01207 |
| nv-qldpc-decoder | 7 | 0.0030 | 100000 | 0.02036 | 0.01775 |
| nv-qldpc-decoder | 7 | 0.0050 | 100000 | 0.03502 | 0.03007 |

## CPU vs GPU Syndrome Benchmark

| backend | median ms | syndromes/s | speedup vs CPU | scope |
| --- | ---: | ---: | ---: | --- |
| cpu_numpy | 325.678 | 307,051 | 1.00x | compute_only |
| gpu_cupy | 2.862 | 34,942,225 | 113.80x | compute_only |

## Decoder Benchmark Results

| variant | decoder | distance | median ms | decoded errors | decoded rate | speed vs LUT | rate vs LUT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUT | single_error_lut | 3 | 3.928 | 11/2000 | 0.0055 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 3 | 4.697 | 11/2000 | 0.0055 | 0.84x | 1.00x |
| LUT | single_error_lut | 5 | 13.450 | 35/2000 | 0.0175 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 5 | 19.871 | 27/2000 | 0.0135 | 0.68x | 0.77x |
| LUT | single_error_lut | 7 | 35.834 | 108/2000 | 0.054 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 7 | 55.973 | 31/2000 | 0.0155 | 0.64x | 0.29x |
| LUT | single_error_lut | 9 | 73.319 | 224/2000 | 0.112 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 9 | 132.510 | 48/2000 | 0.024 | 0.55x | 0.21x |
| LUT | single_error_lut | 11 | 131.018 | 315/2000 | 0.1575 | 1.00x | 1.00x |
| BP=0 | nv-qldpc-decoder | 11 | 270.578 | 64/2000 | 0.032 | 0.48x | 0.20x |

## Decoder Comparison Takeaway

- d=3, BP=0: 0.84x LUT throughput and 1.00x LUT logical error rate.
- d=5, BP=0: 0.68x LUT throughput and 0.77x LUT logical error rate.
- d=7, BP=0: 0.64x LUT throughput and 0.29x LUT logical error rate.
- d=9, BP=0: 0.55x LUT throughput and 0.21x LUT logical error rate.
- d=11, BP=0: 0.48x LUT throughput and 0.20x LUT logical error rate.
