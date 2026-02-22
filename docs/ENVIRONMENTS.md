# Conda Environments

This project is conda-first for reproducibility and clean separation of workflows.

## Environment Matrix

| Env | File | Python | Purpose | Includes |
|---|---|---|---|---|
| `kite-core` | `envs/kite-core.yml` | 3.11 | Day-to-day development, tests, data prep, smoke/baselines | project package, pytest, matplotlib |
| `kite-train` | `envs/kite-train.yml` | 3.11 | RL/SFT training loops | core + torch/transformers/peft/trl/accelerate/datasets + KernelBench API deps (`openai`, `litellm`) + HF CLI (`hf`) |
| `kite-telemetry` | `envs/kite-telemetry.yml` | 3.13 | energy capture and telemetry integrations | core + datasets/pynvml/grpcio + optional local IPW package |

## One-Command Setup

Create/update only the core environment:

```bash
bash scripts/setup_conda_envs.sh
```

Create/update all environments:

```bash
bash scripts/setup_conda_envs.sh --all
```

Create/update all environments and install local IPW package if mounted:

```bash
bash scripts/setup_conda_envs.sh --all --with-ipw
```

`--with-ipw` installs into `kite-telemetry` only because the local `intelligence-per-watt` package currently requires Python `>=3.13,<3.14`.

## Activation

```bash
conda activate kite-core
```

or:

```bash
conda activate kite-train
```

or:

```bash
conda activate kite-telemetry
```

## Notes

- Training on H100 clusters is typically done from `kite-train`.
- Telemetry capture and IPW adapters are typically done from `kite-telemetry`.
- KernelBench dataset API imports require `python-dotenv`; it is included via the base project dependency set.
- For stable model reuse across runs/nodes, set `KITE_HF_CACHE` to a persistent path and optionally `KITE_HF_LOCAL_FILES_ONLY=1` after cache warm-up.
- Keep environment specs centralized in `envs/` and avoid ad-hoc per-user environment files.
