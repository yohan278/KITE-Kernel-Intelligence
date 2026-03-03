# M_ALL / data_scale_ablation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Efficiency scaling with diminishing returns from small to medium data scale.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.086479 ms
- Joules mean: 0.095472
- SLA violation rate: 0.0167

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
