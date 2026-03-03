# M0_SFT / kernel_generation_baseline

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Baseline kernel generator with strong compile behavior and weaker energy optimality.

## Alignment Snapshot

- Compile rate: 0.9333
- Correctness: 0.6500
- Pass@k: 0.6800
- Runtime mean: 0.094592 ms
- Joules mean: 0.111216
- SLA violation rate: 0.1333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
