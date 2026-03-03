# M_ALL / inference_budget_ablation

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Inference budget trade-off with clear early gains and saturation knee.

## Alignment Snapshot

- Compile rate: 0.9500
- Correctness: 0.8000
- Pass@k: 0.8800
- Runtime mean: 0.084816 ms
- Joules mean: 0.091422
- SLA violation rate: 0.0667

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
