# M_ALL / measurement_repeatability

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Repeatability of runtime/energy measurements under matched settings.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.085115 ms
- Joules mean: 0.093316
- SLA violation rate: 0.0500

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
