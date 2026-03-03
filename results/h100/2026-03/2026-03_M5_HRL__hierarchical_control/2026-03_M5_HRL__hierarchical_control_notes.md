# M5_HRL / hierarchical_control

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Hierarchical controller expected to improve regime transition robustness.

## Alignment Snapshot

- Compile rate: 0.9667
- Correctness: 0.8333
- Pass@k: 0.8753
- Runtime mean: 0.075501 ms
- Joules mean: 0.076188
- SLA violation rate: 0.0333

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
