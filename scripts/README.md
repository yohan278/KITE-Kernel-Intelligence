# Scripts Directory

All executable scripts for KITE, organized by purpose.

## Directory Layout

```
scripts/
  reproduce.sh               Full end-to-end reproduction pipeline
  setup/                     Environment setup and data preparation
  training/                  Per-model training scripts (M0-M5)
  experiments/               Experiment runners matching results/h100/ format
  eval/                      Standalone evaluation and baseline scripts
  analysis/                  Post-hoc analysis, synthetic data, paper artifacts
  inference/                 Multiturn inference optimization
  agents/                    Multi-agent cloud orchestration (branch isolation)
```

## setup/

| Script | Description |
|--------|-------------|
| `setup_conda_envs.sh` | Create/update conda environments (`kite-core`, `kite-train`, `kite-telemetry`) |
| `01_sync_sources.sh` | Verify external dependencies (KernelBench, IPW) and GPU availability |
| `02_build_dataset.py` | Build local dataset splits from KernelBench |

```bash
bash scripts/setup/setup_conda_envs.sh --all --with-ipw
bash scripts/setup/01_sync_sources.sh
conda activate kite-train
python scripts/setup/02_build_dataset.py
```

## training/

Per-model training pipelines for M0 through M5.

| Script | Model | Method |
|--------|-------|--------|
| `train_m0_sft.py` | M0 | Supervised fine-tuning (Qwen + LoRA) |
| `train_m1_throughput_grpo.py` | M1 | Throughput-only GRPO kernel RL |
| `train_m2_energy_grpo.py` | M2 | Energy-aware GRPO (joules + power in reward) |
| `train_m3_ipw_blend_grpo.py` | M3 | IPW-blend GRPO (ipw_blend_weight sweep) |
| `train_m4_runtime_ppo.py` | M4 | Runtime PPO (latency, throughput, mixed regimes) |
| `train_m5_hrl.py` | M5 | Hierarchical RL (alternating kernel + runtime) |
| `train_rl.py` | M1/M2/M3 | General-purpose RL runner (auto-detects stage from config) |
| `run_all_training.sh` | All | Sequential orchestrator for all training scripts |

```bash
python scripts/training/train_m0_sft.py --config configs/training/m0_sft.yaml
python scripts/training/train_rl.py --config configs/exp/throughput_rl/seed11.yaml
bash scripts/training/run_all_training.sh
```

## experiments/

Experiment runners that produce results in the exact format found in `results/h100/2026-03/`. Each generates a run log, per-seed CSVs, summary JSONs, and plot data.

| Script | Experiments | Count |
|--------|-------------|-------|
| `03_run_sft.py` | M0 SFT experiments (baseline, single-shot, multiturn) | 3 |
| `06_run_runtime_ppo.py` | M4 runtime PPO regimes | 4 |
| `07_run_hrl.py` | M5 HRL experiments | 2 |
| `08_eval_all.py` | Comparison and M_ALL suite experiments | 20 |
| `09_build_paper_artifacts.py` | Paper artifacts bundle (unique format) | 1 |
| `matched_runtime_energy.py` | Matched-runtime energy analysis | 1 |
| `run_all_experiments.sh` | Orchestrator for all 31 experiments | -- |

```bash
# Run a single experiment
python scripts/experiments/08_eval_all.py \
    --config configs/exp/cross_hardware_transfer/seed{seed}.yaml \
    --experiment-name 2026-03_M_ALL__cross_hardware_transfer

# Run all 31 experiments
bash scripts/experiments/run_all_experiments.sh

# Dry run (prints commands without executing)
bash scripts/experiments/run_all_experiments.sh --dry-run
```

## eval/

Standalone evaluation scripts for individual tasks, policies, and baselines.

| Script | Description |
|--------|-------------|
| `eval_candidate.py` | Evaluate a single kernel candidate on a specific task |
| `eval_policy.py` | Evaluate a trained policy checkpoint |
| `run_baselines.py` | Run KernelBench baselines (phase 1) |
| `smoke_test_one_task.py` | Quick single-task smoke test |

```bash
python scripts/eval/smoke_test_one_task.py --config configs/smoke.yaml
python scripts/eval/eval_candidate.py --task L1_1 --code /path/to/kernel.py
python scripts/eval/run_baselines.py --config configs/project.yaml
```

## analysis/

Post-hoc analysis, synthetic data generation, and paper artifact creation.

| Script | Description |
|--------|-------------|
| `generate_h100_target_synthetic_results.py` | Generate per-task metrics, CI stats, and plot CSVs from run logs |
| `build_h100_paper_artifacts.py` | Render all paper figures and tables from synthetic data |
| `09_plot_pareto.py` | Generate Pareto frontier plots |
| `run_phase_trace.py` | Run hierarchical phase tracing |

```bash
python scripts/analysis/generate_h100_target_synthetic_results.py \
    --results-root results/h100/2026-03
python scripts/analysis/build_h100_paper_artifacts.py \
    --results-root results/h100/2026-03
```

## inference/

Multiturn inference optimization scripts.

| Script | Description |
|--------|-------------|
| `multiturn_optimize.py` | Single-GPU multiturn kernel optimization |
| `multiturn_optimize_multi_gpu.py` | Multi-GPU distributed multiturn optimization |

```bash
python scripts/inference/multiturn_optimize.py --generation-mode local \
    --hf-cache-dir "$KITE_HF_CACHE"
```

## agents/

Multi-agent cloud orchestration with branch/worktree isolation. See [agents/README.md](agents/README.md) for the full agent workflow documentation.

## reproduce.sh

End-to-end reproduction: syncs sources, builds data, runs training, evaluation, and analysis in sequence.

```bash
bash scripts/reproduce.sh
```
