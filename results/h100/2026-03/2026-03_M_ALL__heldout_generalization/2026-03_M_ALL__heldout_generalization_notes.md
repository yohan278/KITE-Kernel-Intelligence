# M_ALL / heldout_generalization

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Held-out generalization maintaining model ordering and effect direction.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8550
- Runtime mean: 0.087596 ms
- Joules mean: 0.095521
- SLA violation rate: 0.0000

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
