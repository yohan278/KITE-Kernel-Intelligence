# M1_GRPO_THROUGHPUT / throughput_rl

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Throughput anchor: fastest model among generator-only variants.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8333
- Pass@k: 0.8833
- Runtime mean: 0.081188 ms
- Joules mean: 0.101463
- SLA violation rate: 0.0500

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
