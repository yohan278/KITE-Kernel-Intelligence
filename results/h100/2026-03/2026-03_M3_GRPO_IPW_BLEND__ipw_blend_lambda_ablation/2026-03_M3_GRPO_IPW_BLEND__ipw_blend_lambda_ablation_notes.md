# M3_GRPO_IPW_BLEND / ipw_blend_lambda_ablation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

IPW lambda ablation for reward component sensitivity.

## Alignment Snapshot

- Compile rate: 0.9667
- Correctness: 0.8167
- Pass@k: 0.8717
- Runtime mean: 0.085197 ms
- Joules mean: 0.085312
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
