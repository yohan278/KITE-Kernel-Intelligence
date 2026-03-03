# M2_GRPO_ENERGY / energy_aware_rl

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Energy-aware policy with near-matched correctness and lower joules than M1.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8167
- Pass@k: 0.8667
- Runtime mean: 0.083921 ms
- Joules mean: 0.090561
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
