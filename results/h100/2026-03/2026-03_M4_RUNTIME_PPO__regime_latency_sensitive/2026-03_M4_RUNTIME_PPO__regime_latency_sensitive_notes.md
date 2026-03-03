# M4_RUNTIME_PPO / regime_latency_sensitive

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Latency-sensitive regime emphasizing p95 latency and SLA behavior.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8400
- Runtime mean: 0.072919 ms
- Joules mean: 0.076491
- SLA violation rate: 0.0167

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
