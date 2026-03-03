# M1_M2_M3 / throughput_vs_energy_vs_ipwblend

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Pareto frontier comparison with M1 speed edge and M2/M3 energy-efficient region.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.082514 ms
- Joules mean: 0.092394
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
