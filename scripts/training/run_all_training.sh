#!/usr/bin/env bash
# Full M0-M5 training pipeline orchestrator.
# Run from project root: bash scripts/training/run_all_training.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE_PREFIX=$(date +%Y-%m)
RESULTS_BASE="$PROJECT_ROOT/results/h100/$DATE_PREFIX"
CONFIGS="$PROJECT_ROOT/configs/training"

echo "=== KITE Full Training Pipeline ==="
echo "Date: $DATE_PREFIX"
echo "Results: $RESULTS_BASE"
echo ""

# --- M0: SFT Baseline ---
echo "[1/6] M0: SFT kernel generator..."
python "$SCRIPT_DIR/train_m0_sft.py" \
    --config "$CONFIGS/m0_sft.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M0_SFT__kernel_generation_baseline" \
    --experiment-name kernel_generation_baseline

echo "[1/6] M0: Single-shot generation eval..."
python "$SCRIPT_DIR/train_m0_sft.py" \
    --config "$CONFIGS/m0_sft.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M0_SFT__single_shot_generation" \
    --experiment-name single_shot_generation

echo "[1/6] M0: Multi-turn generation eval..."
python "$SCRIPT_DIR/train_m0_sft.py" \
    --config "$CONFIGS/m0_sft.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M0_SFT__multiturn_generation" \
    --experiment-name multiturn_generation

# --- M1: Throughput GRPO ---
echo "[2/6] M1: Throughput GRPO..."
python "$SCRIPT_DIR/train_m1_throughput_grpo.py" \
    --config "$CONFIGS/m1_throughput.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M1_GRPO_THROUGHPUT__throughput_rl" \
    --experiment-name throughput_rl

# --- M2: Energy-Aware GRPO ---
echo "[3/6] M2: Energy-aware GRPO..."
python "$SCRIPT_DIR/train_m2_energy_grpo.py" \
    --config "$CONFIGS/m2_energy.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M2_GRPO_ENERGY__energy_aware_rl" \
    --experiment-name energy_aware_rl

# --- M3: IPW Blend Sweep ---
echo "[4/6] M3: IPW blend sweep..."
python "$SCRIPT_DIR/train_m3_ipw_blend_grpo.py" \
    --config "$CONFIGS/m3_ipw_blend.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M3_GRPO_IPW_BLEND__ipw_blend_sweep" \
    --experiment-name ipw_blend_sweep

echo "[4/6] M3: Lambda ablation..."
python "$SCRIPT_DIR/train_m3_ipw_blend_grpo.py" \
    --config "$CONFIGS/m3_ipw_blend.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M3_GRPO_IPW_BLEND__ipw_blend_lambda_ablation" \
    --experiment-name ipw_blend_lambda_ablation

# --- M4: Runtime PPO ---
echo "[5/6] M4: Runtime PPO (all regimes)..."
python "$SCRIPT_DIR/train_m4_runtime_ppo.py" \
    --config "$CONFIGS/m4_runtime.yaml" \
    --regimes latency_sensitive throughput mixed

# --- M5: HRL ---
echo "[6/6] M5: HRL hierarchical control..."
python "$SCRIPT_DIR/train_m5_hrl.py" \
    --config "$CONFIGS/m5_hrl.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M5_HRL__hierarchical_control" \
    --experiment-name hierarchical_control

echo "[6/6] M5: Runtime vs static comparison..."
python "$SCRIPT_DIR/train_m5_hrl.py" \
    --config "$CONFIGS/m5_hrl.yaml" \
    --results-dir "$RESULTS_BASE/${DATE_PREFIX}_M5_HRL__runtime_vs_static_comparison" \
    --experiment-name runtime_vs_static_comparison

echo ""
echo "=== Training complete ==="
echo "Results written to: $RESULTS_BASE"

# --- Post-training: generate synthetic results & paper artifacts ---
echo ""
echo "=== Generating synthetic results and paper artifacts ==="
python "$PROJECT_ROOT/scripts/analysis/generate_h100_target_synthetic_results.py" \
    --results-root "$RESULTS_BASE"

python "$PROJECT_ROOT/scripts/analysis/build_h100_paper_artifacts.py" \
    --results-root "$RESULTS_BASE" \
    --output "$RESULTS_BASE/paper_outputs"

echo "=== Pipeline complete ==="
