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
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

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
