# CUDA-Q QEC Artifacts

Save generated CSVs, plots, and screenshots here after running the examples on
Brev L4.

Suggested artifacts:

- `steane_brev_l4.csv`
- `surface_brev_l4.csv`
- `cpu_gpu_syndrome_brev_l4.csv`
- `surface_sweep_lut_brev_l4.csv`
- `surface_sweep_lut_brev_l4.png`
- `surface_sweep_qldpc_brev_l4.csv`
- `surface_sweep_qldpc_brev_l4.png`
- `surface_sweep_combined.png`
- `decoder_lut_bp_sweep_brev_l4.csv`
- `steane_logical_error_rates.png`
- `cpu_gpu_speedup.png`
- `decoder_time_accuracy.png`
- `SUMMARY.md`
- terminal screenshots for `nvidia-smi`, install verification, and the final
  benchmark run

Use Brev L4 numbers in the final slides so the CSVs, plots, and screenshots all
come from the same platform.

The default surface sweep uses distances 3, 5, and 7 with `rounds=d`, lower
physical error rates, and 10,000 shots. If you run `--full-surface-sweep`, the
same artifact names will include distances 9 and 11 as well.
