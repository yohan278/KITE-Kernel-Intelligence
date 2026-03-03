# M0_M1_M2_M3 / single_shot_vs_multiturn

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Cross-model pass@k vs turns evidence: single-shot < multiturn < RL-initialized multiturn.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.7833
- Pass@k: 0.9033
- Runtime mean: 0.087755 ms
- Joules mean: 0.100967
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
