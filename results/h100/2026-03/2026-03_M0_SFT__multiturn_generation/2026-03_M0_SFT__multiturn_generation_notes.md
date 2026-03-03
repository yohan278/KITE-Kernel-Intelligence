# M0_SFT / multiturn_generation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Multiturn baseline expected to improve pass@k by turn 5 over single-shot.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.7167
- Pass@k: 0.8067
- Runtime mean: 0.101502 ms
- Joules mean: 0.120566
- SLA violation rate: 0.0167

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
