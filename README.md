# KITE-Kernel-Intelligence

KITE (Kernel Intelligence-per-watt Tree Explorer) is a research framework for:

1. Kernel-only RL (correctness + speed).
2. Energy-aware kernel RL (IPW-style metrics).
3. Hierarchical RL (kernel + runtime controls under SLA constraints).

## Status

This repository contains a complete v0 scaffold with:

- Typed domain models for tasks, candidates, telemetry, and episodes.
- KernelBench/IPW adapters.
- Reward functions for kernel, energy-aware kernel, and HRL tracks.
- Trainer stubs for SFT, GRPO-style kernel RL, PPO runtime RL, and hierarchical alternating training.
- CLI entrypoints and experiment/evaluation orchestration.
- Config files, scripts, docs, and tests.

## Quick Start

```bash
cd /Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence

# Conda-first setup (core env only)
bash scripts/setup_conda_envs.sh
conda activate kite-core

# Build local dataset splits
kite data build --kernelbench-root ./external/KernelBench --output ./data/kernelbench/processed

# Run staged training
kite train sft
kite train kernel-grpo
kite train runtime-ppo
kite train hrl

# Run experiment suite
kite eval suite
```

For all environments (`core`, `train`, `telemetry`) and optional local IPW package:

```bash
bash scripts/setup_conda_envs.sh --all --with-ipw
```

`--with-ipw` installs IPW into `kite-telemetry` (Python 3.13) only.

Environment specs are centralized in `/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/envs/`.
See `/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/docs/ENVIRONMENTS.md` for full setup matrix.

## Phase Commands

```bash
# Phase 0
python scripts/smoke_test_one_task.py --config configs/smoke.yaml

# Phase 1
python scripts/run_baselines.py --config configs/project.yaml

# Phase 2
python scripts/eval_candidate.py --task L1_1 --code /abs/path/kernel.py

# Phase 3
python scripts/multiturn_optimize.py --generation-mode stub

# Phase 4+
python scripts/train_rl.py --config configs/train_throughput.yaml
python scripts/train_rl.py --config configs/train_energyaware.yaml
python scripts/run_phase_trace.py --config configs/hierarchical.yaml
python scripts/reproduce.sh
```

## External Sources

- KernelBench source should be cloned into `external/KernelBench` and pinned to a commit.
- IPW internal source can be mounted/symlinked at `external/ipw_internal`.

Network-restricted environments can keep these as placeholders and point adapters to local exports.

## CLI Commands

- `kite data build`
- `kite train sft`
- `kite train kernel-grpo`
- `kite train runtime-ppo`
- `kite train hrl`
- `kite eval suite`

## Project Layout

See `docs/DESIGN.md` and `docs/EXPERIMENTS.md` for architecture and execution details.
