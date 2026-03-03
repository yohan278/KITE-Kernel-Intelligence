# M_ALL / reward_ablation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Reward decomposition across throughput, energy, and IPW blend terms.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.7833
- Pass@k: 0.8333
- Runtime mean: 0.087586 ms
- Joules mean: 0.099104
- SLA violation rate: 0.0833

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
