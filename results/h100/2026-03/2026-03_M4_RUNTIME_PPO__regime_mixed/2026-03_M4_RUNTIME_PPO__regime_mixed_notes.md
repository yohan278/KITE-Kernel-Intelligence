# M4_RUNTIME_PPO / regime_mixed

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Mixed regime control with balanced latency-energy objective.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8400
- Runtime mean: 0.079118 ms
- Joules mean: 0.084004
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
