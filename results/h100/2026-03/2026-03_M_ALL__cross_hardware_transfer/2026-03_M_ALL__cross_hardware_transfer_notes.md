# M_ALL / cross_hardware_transfer

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Cross-hardware transfer intended to preserve checkpoint ranking (high correlation).

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8550
- Runtime mean: 0.086465 ms
- Joules mean: 0.093470
- SLA violation rate: 0.1500

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
