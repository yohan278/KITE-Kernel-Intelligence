# M5_HRL / runtime_vs_static_comparison

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Direct M5 vs static policy comparison on SLA and joint latency-energy objective.

## Alignment Snapshot

- Compile rate: 0.9667
- Correctness: 0.8167
- Pass@k: 0.8587
- Runtime mean: 0.076615 ms
- Joules mean: 0.078920
- SLA violation rate: 0.0000

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
