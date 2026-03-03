#!/usr/bin/env bash
# Run all 31 experiments matching results/h100/2026-03/ directory structure.
# Each experiment gets its own results directory with the exact naming format.
#
# Usage:
#   bash scripts/run_all_experiments.sh [--dry-run] [--results-root DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE_PREFIX=$(date +%Y-%m)
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results/h100/$DATE_PREFIX}"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --results-root=*) RESULTS_ROOT="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_ROOT"

run_exp() {
    local script="$1"
    local config_dir="$2"
    local exp_name="$3"
    local results_dir="$RESULTS_ROOT/$exp_name"

    echo "=== Running experiment: $exp_name ==="
    echo "    script: $script"
    echo "    config: configs/exp/$config_dir/seed{seed}.yaml"
    echo "    output: $results_dir"

    if $DRY_RUN; then
        echo "    [DRY RUN - skipped]"
        return
    fi

    python "$PROJECT_ROOT/scripts/$script" \
        --config "configs/exp/$config_dir/seed{seed}.yaml" \
        --results-dir "$results_dir" \
        --experiment-name "$exp_name" \
        --local-files-only 2>&1 | tail -3

    echo "    [DONE]"
    echo ""
}

run_bundle() {
    local exp_name="${DATE_PREFIX}_M_ALL__paper_artifacts"
    echo "=== Running experiment: $exp_name (bundle) ==="
    if $DRY_RUN; then
        echo "    [DRY RUN - skipped]"
        return
    fi
    python "$PROJECT_ROOT/scripts/experiments/09_build_paper_artifacts.py" \

        --results-root "$RESULTS_ROOT" \
        --output "$RESULTS_ROOT/$exp_name" 2>&1 | tail -3
    echo "    [DONE]"
    echo ""
}

echo "=================================================================="
echo " KITE Experiment Suite — $DATE_PREFIX"
echo " Results root: $RESULTS_ROOT"
echo " Dry run: $DRY_RUN"
echo "=================================================================="
echo ""

# ─── M0 SFT experiments ────────────────────────────────────────────
run_exp "experiments/03_run_sft.py" "kernel_generation_baseline" "${DATE_PREFIX}_M0_SFT__kernel_generation_baseline"
run_exp "experiments/03_run_sft.py" "single_shot_generation"     "${DATE_PREFIX}_M0_SFT__single_shot_generation"
run_exp "experiments/03_run_sft.py" "multiturn_generation"        "${DATE_PREFIX}_M0_SFT__multiturn_generation"

# ─── M1 GRPO Throughput ────────────────────────────────────────────
run_exp "training/train_rl.py"   "throughput_rl"               "${DATE_PREFIX}_M1_GRPO_THROUGHPUT__throughput_rl"

# ─── M2 GRPO Energy ────────────────────────────────────────────────
run_exp "training/train_rl.py"   "energy_aware_rl"             "${DATE_PREFIX}_M2_GRPO_ENERGY__energy_aware_rl"

# ─── M3 GRPO IPW Blend ─────────────────────────────────────────────
run_exp "training/train_rl.py"   "ipw_blend_sweep"             "${DATE_PREFIX}_M3_GRPO_IPW_BLEND__ipw_blend_sweep"
run_exp "training/train_rl.py"   "ipw_blend_lambda_ablation"   "${DATE_PREFIX}_M3_GRPO_IPW_BLEND__ipw_blend_lambda_ablation"

# ─── M4 Runtime PPO ────────────────────────────────────────────────
run_exp "experiments/06_run_runtime_ppo.py" "regime_latency_sensitive" "${DATE_PREFIX}_M4_RUNTIME_PPO__regime_latency_sensitive"
run_exp "experiments/06_run_runtime_ppo.py" "regime_throughput"         "${DATE_PREFIX}_M4_RUNTIME_PPO__regime_throughput"
run_exp "experiments/06_run_runtime_ppo.py" "regime_mixed"              "${DATE_PREFIX}_M4_RUNTIME_PPO__regime_mixed"
run_exp "experiments/06_run_runtime_ppo.py" "runtime_control"           "${DATE_PREFIX}_M4_RUNTIME_PPO__runtime_control"

# ─── M5 HRL ────────────────────────────────────────────────────────
run_exp "experiments/07_run_hrl.py" "hierarchical_control"        "${DATE_PREFIX}_M5_HRL__hierarchical_control"
run_exp "experiments/07_run_hrl.py" "runtime_vs_static_comparison" "${DATE_PREFIX}_M5_HRL__runtime_vs_static_comparison"

# ─── Comparison / analysis experiments ──────────────────────────────
run_exp "experiments/08_eval_all.py" "single_shot_vs_multiturn"        "${DATE_PREFIX}_M0_M1_M2_M3__single_shot_vs_multiturn"
run_exp "experiments/08_eval_all.py" "matched_runtime_different_energy" "${DATE_PREFIX}_M1_M2_M3__matched_runtime_different_energy"
run_exp "experiments/08_eval_all.py" "throughput_vs_energy_vs_ipwblend" "${DATE_PREFIX}_M1_M2_M3__throughput_vs_energy_vs_ipwblend"

# ─── M_ALL evaluation suite experiments ─────────────────────────────
run_exp "experiments/08_eval_all.py" "cross_hardware_transfer"    "${DATE_PREFIX}_M_ALL__cross_hardware_transfer"
run_exp "experiments/08_eval_all.py" "data_scale_ablation"        "${DATE_PREFIX}_M_ALL__data_scale_ablation"
run_exp "experiments/08_eval_all.py" "difficulty_stratified_eval"  "${DATE_PREFIX}_M_ALL__difficulty_stratified_eval"
run_exp "experiments/08_eval_all.py" "failure_taxonomy"            "${DATE_PREFIX}_M_ALL__failure_taxonomy"
run_exp "experiments/08_eval_all.py" "final_eval_suite"            "${DATE_PREFIX}_M_ALL__final_eval_suite"
run_exp "experiments/08_eval_all.py" "heldout_generalization"      "${DATE_PREFIX}_M_ALL__heldout_generalization"
run_exp "experiments/08_eval_all.py" "inference_budget_ablation"   "${DATE_PREFIX}_M_ALL__inference_budget_ablation"
run_exp "experiments/08_eval_all.py" "measurement_repeatability"   "${DATE_PREFIX}_M_ALL__measurement_repeatability"
run_exp "experiments/08_eval_all.py" "paper_appendix"              "${DATE_PREFIX}_M_ALL__paper_appendix"
run_exp "experiments/08_eval_all.py" "paper_figures"               "${DATE_PREFIX}_M_ALL__paper_figures"
run_exp "experiments/08_eval_all.py" "paper_tables"                "${DATE_PREFIX}_M_ALL__paper_tables"
run_exp "experiments/08_eval_all.py" "reward_ablation"             "${DATE_PREFIX}_M_ALL__reward_ablation"
run_exp "experiments/08_eval_all.py" "seed_robustness"             "${DATE_PREFIX}_M_ALL__seed_robustness"
run_exp "experiments/08_eval_all.py" "telemetry_realism_ablation"  "${DATE_PREFIX}_M_ALL__telemetry_realism_ablation"

# ─── Paper artifacts bundle (unique format) ────────────────────────
run_bundle

echo "=================================================================="
echo " All 31 experiments complete."
echo " Results at: $RESULTS_ROOT"
echo "=================================================================="
