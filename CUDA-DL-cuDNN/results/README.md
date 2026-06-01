# Results

Save benchmark CSVs here after running on each target platform.

Recommended filenames:

- `brev_l4.csv`
- `dgx_spark.csv`
- `brev_l4_single_inference.csv`

The benchmark CSV columns are:

- `platform_label`
- `benchmark`
- `mode`
- `device_name`
- `cudnn_version`
- `batch_size`
- `latency_ms`
- `throughput_img_s`
- `speedup_vs_cpu`

The single-inference Nsight CSV uses:

- `platform_label`
- `workload`
- `mode`
- `run`
- `device_name`
- `cudnn_version`
- `batch_size`
- `wall_latency_ms`
- `cuda_event_ms`
