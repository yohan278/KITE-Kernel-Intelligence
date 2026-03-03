#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/8] Sync sources"
bash "$ROOT/scripts/setup/01_sync_sources.sh"

echo "[2/8] Build dataset"
python "$ROOT/scripts/setup/02_build_dataset.py"

echo "[3/8] Phase 0 smoke"
python "$ROOT/scripts/eval/smoke_test_one_task.py" --config "$ROOT/configs/smoke.yaml"

echo "[4/8] Phase 1 baselines"
python "$ROOT/scripts/eval/run_baselines.py" --config "$ROOT/configs/project.yaml"

echo "[5/8] Kernel RL throughput and energy-aware"
python "$ROOT/scripts/training/train_rl.py" --config "$ROOT/configs/train_throughput.yaml"
python "$ROOT/scripts/training/train_rl.py" --config "$ROOT/configs/train_energyaware.yaml"

echo "[6/8] Runtime + HRL + eval"
python "$ROOT/scripts/experiments/06_run_runtime_ppo.py" --config "configs/exp/runtime_control/seed{seed}.yaml"
python "$ROOT/scripts/experiments/07_run_hrl.py" --config "configs/exp/hierarchical_control/seed{seed}.yaml"
python "$ROOT/scripts/experiments/08_eval_all.py" --config "configs/exp/final_eval_suite/seed{seed}.yaml"

echo "[7/8] Phase trace + analysis"
python "$ROOT/scripts/analysis/run_phase_trace.py" --config "$ROOT/configs/hierarchical.yaml"
python "$ROOT/analysis/make_tables.py"
python "$ROOT/analysis/make_figures.py"

echo "[8/8] Pareto plots"
python "$ROOT/scripts/analysis/09_plot_pareto.py"

echo "Reproduction complete."
