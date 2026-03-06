# KITE Design

## Objective

KITE co-optimizes generated kernels and runtime controls to improve intelligence-per-watt style metrics while preserving correctness and latency SLAs.

## Architecture

1. `adapters/`
- KernelBench ingestion and proxy evaluation hooks.
- IPW trace parsing and metric summarization.
- Kevin-style grouped rollout helpers.

2. `rewards/`
- Kernel-only reward: correctness + speedup with compile/timeout penalties.
- Energy-aware reward: adds per-token energy, power, and SLA penalties.
- HRL reward: throughput + APJ/APW + stability minus SLA violations.

3. `envs/`
- Kernel generation/eval environment.
- Runtime control environment.
- Hierarchical wrapper with kernel-family selection.

4. `trainers/`
- SFT bootstrapping.
- GRPO-like kernel trainer.
- PPO-like runtime trainer.
- Alternating hierarchical trainer.

5. `eval/`
- Benchmark matrix runner.
- Ablation runner.
- Report and Pareto artifact generation.

## Interfaces

Core typed entities:

- `KernelTask`
- `KernelCandidate`
- `EnergyTrace`
- `RuntimeState`
- `RewardBreakdown`
- `EpisodeRecord`

## Implementation Notes

- The current implementation uses deterministic stubs for model inference and runtime dynamics to keep local development and testing fast.
- Replace policy generation and evaluator proxy hooks with real KernelBench and model-serving integrations in the next phase.
