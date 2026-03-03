# M_ALL / difficulty_stratified_eval

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Difficulty-stratified success with gains in medium and hard buckets.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.085023 ms
- Joules mean: 0.092786
- SLA violation rate: 0.0833

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
