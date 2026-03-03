# M4_RUNTIME_PPO / runtime_control

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Runtime PPO policy with lower SLA violations than static baselines.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8400
- Runtime mean: 0.078380 ms
- Joules mean: 0.082012
- SLA violation rate: 0.0167

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
