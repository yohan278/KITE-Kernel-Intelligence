# M0_SFT / single_shot_generation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Single-shot baseline for pass@k vs turns comparison.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.6500
- Pass@k: 0.6650
- Runtime mean: 0.089284 ms
- Joules mean: 0.103045
- SLA violation rate: 0.0833

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
