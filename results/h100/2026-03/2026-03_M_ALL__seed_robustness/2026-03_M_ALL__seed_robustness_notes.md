# M_ALL / seed_robustness

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Seed stability: consistent trend direction with no catastrophic seed collapse.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.084699 ms
- Joules mean: 0.093310
- SLA violation rate: 0.0500

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
