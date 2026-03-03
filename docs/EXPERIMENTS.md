# KITE Experiments

Full catalog of the 31 experiments in `results/h100/2026-03/`. Each experiment runs 3 seeds (11, 22, 33) across 80 KernelBench tasks on H100-SXM5-80GB GPUs.

## Training Phases

1. **Baselines** -- KernelBench default outputs, IPW telemetry validation.
2. **SFT (M0)** -- Supervised fine-tuning with prompt/kernel pairs.
3. **Kernel GRPO (M1)** -- Throughput-only grouped RL with correctness-heavy early epochs.
4. **Energy-Aware GRPO (M2)** -- Adds energy/token and power penalties.
5. **IPW-Blend GRPO (M3)** -- Blends IPW reward signal with energy-aware reward.
6. **Runtime PPO (M4)** -- Optimizes power cap, DVFS, microbatch, concurrency.
7. **Hierarchical RL (M5)** -- Alternates kernel and runtime updates.

## Acceptance Targets

- Correctness >= 95% of performance-only baseline.
- Decode energy/token improvement >= 10% at matched correctness.
- APJ/APW improvement at matched p95 latency SLA.
- SLA violation rate < 5%.

---

## Experiment Catalog

### M0: SFT Experiments

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 1 | `2026-03_M0_SFT__kernel_generation_baseline` | `scripts/experiments/03_run_sft.py` | `configs/exp/kernel_generation_baseline/` | sft |
| 2 | `2026-03_M0_SFT__single_shot_generation` | `scripts/experiments/03_run_sft.py` | `configs/exp/single_shot_generation/` | sft |
| 3 | `2026-03_M0_SFT__multiturn_generation` | `scripts/experiments/03_run_sft.py` | `configs/exp/multiturn_generation/` | sft |

### M1: Throughput GRPO

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 4 | `2026-03_M1_GRPO_THROUGHPUT__throughput_rl` | `scripts/training/train_rl.py` | `configs/exp/throughput_rl/` | kernel_grpo_throughput |

### M2: Energy-Aware GRPO

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 5 | `2026-03_M2_GRPO_ENERGY__energy_aware_rl` | `scripts/training/train_rl.py` | `configs/exp/energy_aware_rl/` | kernel_grpo_energy |

### M3: IPW-Blend GRPO

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 6 | `2026-03_M3_GRPO_IPW_BLEND__ipw_blend_sweep` | `scripts/training/train_rl.py` | `configs/exp/ipw_blend_sweep/` | kernel_grpo_ipw |
| 7 | `2026-03_M3_GRPO_IPW_BLEND__ipw_blend_lambda_ablation` | `scripts/training/train_rl.py` | `configs/exp/ipw_blend_lambda_ablation/` | kernel_grpo_ipw |

### M4: Runtime PPO

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 8 | `2026-03_M4_RUNTIME_PPO__regime_latency_sensitive` | `scripts/experiments/06_run_runtime_ppo.py` | `configs/exp/regime_latency_sensitive/` | runtime_ppo |
| 9 | `2026-03_M4_RUNTIME_PPO__regime_throughput` | `scripts/experiments/06_run_runtime_ppo.py` | `configs/exp/regime_throughput/` | runtime_ppo |
| 10 | `2026-03_M4_RUNTIME_PPO__regime_mixed` | `scripts/experiments/06_run_runtime_ppo.py` | `configs/exp/regime_mixed/` | runtime_ppo |
| 11 | `2026-03_M4_RUNTIME_PPO__runtime_control` | `scripts/experiments/06_run_runtime_ppo.py` | `configs/exp/runtime_control/` | runtime_ppo |

### M5: Hierarchical RL

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 12 | `2026-03_M5_HRL__hierarchical_control` | `scripts/experiments/07_run_hrl.py` | `configs/exp/hierarchical_control/` | hrl |
| 13 | `2026-03_M5_HRL__runtime_vs_static_comparison` | `scripts/experiments/07_run_hrl.py` | `configs/exp/runtime_vs_static_comparison/` | hrl |

### Cross-Model Comparisons

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 14 | `2026-03_M0_M1_M2_M3__single_shot_vs_multiturn` | `scripts/experiments/08_eval_all.py` | `configs/exp/single_shot_vs_multiturn/` | analysis |
| 15 | `2026-03_M1_M2_M3__matched_runtime_different_energy` | `scripts/experiments/08_eval_all.py` | `configs/exp/matched_runtime_different_energy/` | analysis |
| 16 | `2026-03_M1_M2_M3__throughput_vs_energy_vs_ipwblend` | `scripts/experiments/08_eval_all.py` | `configs/exp/throughput_vs_energy_vs_ipwblend/` | analysis |

### M_ALL Evaluation Suite

| # | Directory | Script | Config | Stage |
|---|-----------|--------|--------|-------|
| 17 | `2026-03_M_ALL__cross_hardware_transfer` | `scripts/experiments/08_eval_all.py` | `configs/exp/cross_hardware_transfer/` | suite |
| 18 | `2026-03_M_ALL__data_scale_ablation` | `scripts/experiments/08_eval_all.py` | `configs/exp/data_scale_ablation/` | suite |
| 19 | `2026-03_M_ALL__difficulty_stratified_eval` | `scripts/experiments/08_eval_all.py` | `configs/exp/difficulty_stratified_eval/` | suite |
| 20 | `2026-03_M_ALL__failure_taxonomy` | `scripts/experiments/08_eval_all.py` | `configs/exp/failure_taxonomy/` | suite |
| 21 | `2026-03_M_ALL__final_eval_suite` | `scripts/experiments/08_eval_all.py` | `configs/exp/final_eval_suite/` | suite |
| 22 | `2026-03_M_ALL__heldout_generalization` | `scripts/experiments/08_eval_all.py` | `configs/exp/heldout_generalization/` | suite |
| 23 | `2026-03_M_ALL__inference_budget_ablation` | `scripts/experiments/08_eval_all.py` | `configs/exp/inference_budget_ablation/` | suite |
| 24 | `2026-03_M_ALL__measurement_repeatability` | `scripts/experiments/08_eval_all.py` | `configs/exp/measurement_repeatability/` | suite |
| 25 | `2026-03_M_ALL__paper_appendix` | `scripts/experiments/08_eval_all.py` | `configs/exp/paper_appendix/` | suite |
| 26 | `2026-03_M_ALL__paper_artifacts` | `scripts/experiments/09_build_paper_artifacts.py` | `configs/exp/paper_artifacts/` | bundle |
| 27 | `2026-03_M_ALL__paper_figures` | `scripts/experiments/08_eval_all.py` | `configs/exp/paper_figures/` | suite |
| 28 | `2026-03_M_ALL__paper_tables` | `scripts/experiments/08_eval_all.py` | `configs/exp/paper_tables/` | suite |
| 29 | `2026-03_M_ALL__reward_ablation` | `scripts/experiments/08_eval_all.py` | `configs/exp/reward_ablation/` | suite |
| 30 | `2026-03_M_ALL__seed_robustness` | `scripts/experiments/08_eval_all.py` | `configs/exp/seed_robustness/` | suite |
| 31 | `2026-03_M_ALL__telemetry_realism_ablation` | `scripts/experiments/08_eval_all.py` | `configs/exp/telemetry_realism_ablation/` | suite |

---

## Output Format

Each experiment directory contains:

```
<experiment_name>/
  logs/
    <name>_run.log                     Structured run log
  <name>_metrics.csv                   Per-task metrics (task_id, seed, compiled, correct, runtime_ms, joules, ...)
  <name>_per_task.jsonl                Detailed per-task records
  <name>_per_seed.csv                  Seed-level aggregates
  <name>_summary.json                  Experiment summary + aggregate metrics
  <name>_ci_stats.json                 95% confidence intervals
  <name>_significance_tests.csv        Paired significance tests
  <name>_failure_taxonomy.csv          Failure reason breakdown
  <name>_warnings.json                 Warning events (power spikes, compile retries, cache hits)
  plot_data/
    <name>_{runtime_joules,pareto,passatk,reward}.csv
```

## Run Log Format

Standard experiments produce 27-33 line logs with this structure:

```
[timestamp] [INFO] job_id=... experiment=... status=starting
[timestamp] [INFO] host=h100-node-XX gpu=H100-SXM5-80GB:N cuda=12.4 driver=550.54
[timestamp] [INFO] env=conda:kite-train python=3.12 torch=2.6.0 triton=3.2.0
[timestamp] [INFO] command="python scripts/experiments/..."
[timestamp] [INFO] dataset=KernelBench split=eval seeds=3 tasks=80 stage=...
[timestamp] [INFO] dataloader workers=N prefetch_factor=N pin_memory=true
[timestamp] [INFO] loading checkpoints and cached telemetry profiles
[timestamp] [INFO] seed=11 compile_cache hit_rate=0.950 restored_graphs=65      # optional
[timestamp] [INFO] seed=11 eval.start checkpoint=...
[timestamp] [INFO] seed=11 progress tasks=26/80 compile=... correct=...
[timestamp] [WARN] seed=11 transient_power_spike task=L4_5 power_w=277          # optional
[timestamp] [INFO] seed=11 progress tasks=53/80 pass_at_k=... runtime_ms=...
[timestamp] [INFO] seed=11 eval.done compile=0.8875 correct=0.5000 pass_at_k=0.6750 runtime_ms=23.686361 joules=7.072028
... (repeat for seeds 22, 33)
[timestamp] [INFO] failure_taxonomy top=syntax_error:28 bins=9
[timestamp] [WARN] sla_violation_rate=0.2167 exceeds preferred threshold 0.0500
[timestamp] [INFO] aggregate compile_rate=... correctness=... pass_at_k=...
[timestamp] [INFO] aggregate runtime_ms=... joules=... power_w=... reward_mean=...
[timestamp] [INFO] artifacts.write metrics=... per_task=... per_seed=...
[timestamp] [INFO] job_id=... status=completed wall_clock_s=...
```

The `paper_artifacts` experiment (#26) uses a distinct bundle format without the GPU/env preamble -- see its log for details.

## Running a Single Experiment

```bash
conda activate kite-train

python scripts/experiments/08_eval_all.py \
    --config configs/exp/reward_ablation/seed{seed}.yaml \
    --experiment-name 2026-03_M_ALL__reward_ablation \
    --results-dir results/h100/2026-03/2026-03_M_ALL__reward_ablation
```

## Running All Experiments

```bash
bash scripts/experiments/run_all_experiments.sh
```
