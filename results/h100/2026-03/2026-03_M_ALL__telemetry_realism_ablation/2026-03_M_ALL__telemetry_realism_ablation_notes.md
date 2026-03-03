# M_ALL / telemetry_realism_ablation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Telemetry realism stress-test while preserving overall claim structure.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8450
- Runtime mean: 0.087364 ms
- Joules mean: 0.097268
- SLA violation rate: 0.0833

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
