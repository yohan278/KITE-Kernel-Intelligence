# M1_M2_M3 / matched_runtime_different_energy

- Date: 2026-03
- Hardware: H100
- Status: completed
- Synthetic policy: TARGET-ALIGNED

## Purpose

Matched-runtime energy advantage: negative joules deltas for M2/M3 vs M1.

## Alignment Snapshot

- Compile rate: 0.9667
- Correctness: 0.8000
- Pass@k: 0.8600
- Runtime mean: 0.084071 ms
- Joules mean: 0.088605
- SLA violation rate: 0.0167

## Reproducibility

Generated deterministically by `scripts/generate_h100_target_synthetic_results.py` for instruction-following and pipeline-shape validation.
