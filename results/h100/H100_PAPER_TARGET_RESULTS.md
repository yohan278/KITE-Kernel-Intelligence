# H100 Paper Target Results Guide

This document defines what results we should target for each model and each experiment so the final paper has clear, defensible claims.

## 1. Paper-level success criteria

1. Statistical validity:
At least 3 seeds per main comparison, report mean, std, and 95% CI for all headline metrics.

2. Correctness-first guarantee:
Any energy claim must be at matched or near-matched correctness (within 1-2 percentage points).

3. Energy claim:
Energy-aware policies (M2/M3) should show at least 10% lower joules (or APJ) versus throughput-only policy (M1) at matched correctness.

4. Runtime claim:
No major latency regression from energy optimization. Target less than 5% median runtime regression for the energy-aware model family.

5. Robustness claim:
Gains should appear on held-out tasks and transfer to H100/L40 comparison with strong rank correlation.

6. Control-policy claim:
Runtime control (M4/M5) should improve SLA behavior versus static policies under regime shifts.

## 2. Model-level expected outcomes

## M0: SFT kernel generator

Expected role:
Strong baseline for compile/correctness with weaker energy optimality.

Target result shape:
Moderate pass@k, good compile rate, broader failure diversity than RL models.

What we want:
Reliable baseline that M1/M2/M3 clearly beat on either pass@k, reward, or energy-adjusted utility.

## M1: Throughput GRPO

Expected role:
Best pure speed-up model among generator-only variants.

Target result shape:
Higher speedup and reward than M0, but higher energy than M2/M3.

What we want:
A clear anchor on the speed side of the Pareto frontier.

## M2: Energy-aware GRPO

Expected role:
Primary balanced model for correctness + runtime + joules.

Target result shape:
Near-M1 correctness, slightly lower speedup, clearly lower joules/APJ.

What we want:
Main energy-efficiency claim: reduced joules with minimal accuracy/runtime tradeoff.

## M3: Energy-aware + IPW-blend GRPO

Expected role:
Best alignment with IPW-style energy objectives.

Target result shape:
Further APJ/APW gains over M2, stable correctness, better domain-level energy outcomes.

What we want:
Evidence that IPW reward blending adds value beyond standard energy-aware RL.

## M4: Runtime PPO policy

Expected role:
Low-level runtime adaptation under latency/throughput/mixed regimes.

Target result shape:
Improved p95 latency and SLA adherence while controlling APJ/APW.

What we want:
Show online runtime control adds value beyond static kernel choice.

## M5: HRL controller + runtime low-level policy

Expected role:
High-level regime/controller improvements on top of M4.

Target result shape:
Best regime transition behavior and stability under changing load.

What we want:
Show hierarchical decisions improve robustness and overall objective.

## 3. Experiment-by-experiment target results

## 3.1 `accuracy_energy_pareto_frontier`

Goal:
Show M0/M1/M2/M3 tradeoff frontier in runtime-speedup-energy space.

Expected meaningful result:
M1 appears on fast edge; M2/M3 dominate in low-joules region; M3 has best balanced non-dominated points.

Quantitative target:
At matched correctness, M2 or M3 reduce joules by at least 10% versus M1.

## 3.2 `passk_vs_turns_curve`

Goal:
Show Kevin-style iterative gains from multiturn reasoning.

Expected meaningful result:
Single-shot < multiturn < RL-initialized multiturn in cumulative pass@k by turn 3-5.

Quantitative target:
At turn 5, multiturn should improve pass@k by at least 5-10 points versus single-shot.

## 3.3 `matched_runtime_energy_advantage`

Goal:
Demonstrate energy improvements are not just from slower execution.

Expected meaningful result:
For paired candidates with |delta runtime| <= 3%, delta joules is consistently negative for M2/M3 relative to M1.

Quantitative target:
Median delta joules <= -8%, with most pairs below zero.

## 3.4 `reward_outcome_decomposition_waterfall`

Goal:
Decompose gain from throughput reward terms, energy terms, and IPW blend.

Expected meaningful result:
Throughput terms raise speed metrics, energy terms reduce APJ, IPW blend improves final energy-aware endpoint.

Quantitative target:
IPW blend contributes an additional positive incremental gain (non-trivial, stable across seeds).

## 3.5 `failure_taxonomy_transition_sankey`

Goal:
Show training reduces structural/code-generation failures.

Expected meaningful result:
Early mass in syntax/arity/compile errors shifts toward correctness-fail or success bins over training.

Quantitative target:
Syntax/arity failure share drops materially (for example 30%+ relative reduction).

## 3.6 `cross_hardware_transfer_scatter`

Goal:
Validate performance ordering transfers across L40 and H100.

Expected meaningful result:
Points cluster near diagonal for key metrics (correctness, joules ranking, runtime ranking).

Quantitative target:
High correlation (Spearman/Pearson) for checkpoint ordering across hardware.

## 3.7 `domain_coverage_stacked_bar`

Goal:
Show more kernels are solved efficiently, not just a few easy tasks.

Expected meaningful result:
M2/M3 increase "energy-efficient solved" share and reduce fallback share across categories.

Quantitative target:
Meaningful positive coverage shift in medium/hard categories.

## 3.8 `runtime_control_regime_figure`

Goal:
Evaluate M4/M5 across latency-sensitive, throughput, mixed regimes.

Expected meaningful result:
M4 beats static baseline; M5 beats or matches M4 in joint latency-energy objective.

Quantitative target:
Lower p95 latency and/or lower APJ with equal or better SLA violation rate.

## 3.9 `difficulty_stratified_success_heatmap`

Goal:
Show improvements are not restricted to easy tasks.

Expected meaningful result:
M2/M3 improve pass@k beyond easy bucket, especially medium/hard.

Quantitative target:
Positive gains in at least two difficulty buckets, not just easiest bucket.

## 3.10 `seed_stability_fan_plot`

Goal:
Demonstrate training stability.

Expected meaningful result:
Reasonable variance bands with consistent trend direction across seeds.

Quantitative target:
No seed collapses completely; confidence bands not dominated by outlier runs.

## 3.11 `efficiency_scaling_curve`

Goal:
Show sample-efficiency as training task count increases.

Expected meaningful result:
Monotonic or near-monotonic improvement in reward/pass@k and energy metrics.

Quantitative target:
Diminishing returns curve with clear gain from small to medium data scales.

## 3.12 `inference_budget_tradeoff_curve`

Goal:
Map quality-cost tradeoff from max turns / completion length.

Expected meaningful result:
Early budget gains then saturation; identify knee point for practical deployment.

Quantitative target:
Document budget setting where >90% of max pass@k is achieved at lower joules/query.

## 3.13 `latency_energy_joint_density`

Goal:
Show distribution-level behavior and outliers, not only means.

Expected meaningful result:
M2/M3 density shifts toward lower joules for similar latency regions.

Quantitative target:
Visible left/down distribution shift with reduced heavy-tail energy outliers.

## 3.14 `oom_timeout_incidence_vs_config`

Goal:
Identify operationally safe config regions.

Expected meaningful result:
OOM/timeout incidence increases with aggressive settings; selected default in stable zone.

Quantitative target:
Chosen production config has low incidence while preserving most quality.

## 3.15 `calibration_plot`

Goal:
Validate reward proxy aligns with held-out realized IPW outcome.

Expected meaningful result:
Positive monotonic trend between predicted reward and realized held-out IPW metrics.

Quantitative target:
Useful calibration with acceptable error and rank consistency.

## 3.16 `per_task_delta_forest_plot`

Goal:
Show per-task heterogeneity for M2/M3 minus M1.

Expected meaningful result:
Majority of tasks have non-positive energy delta with minimal correctness penalty.

Quantitative target:
Most confidence intervals favor energy reduction; annotate failure cases clearly.

## 3.17 `routing_savings_curve`

Goal:
Quantify routing/controller quality vs savings.

Expected meaningful result:
Improved router quality yields monotonic compute/energy savings up to saturation.

Quantitative target:
Clear efficiency uplift over static selection baseline.

## 3.18 `ablation_spider_chart`

Goal:
Summarize which reward components matter for each metric.

Expected meaningful result:
Removing energy/IPW terms hurts energy outcomes; removing throughput terms hurts latency/speedup.

Quantitative target:
Distinct, interpretable deformation patterns across ablations.

## 4. Mapping to existing run folders

Use this folder set as the canonical run matrix for these analyses:

1. Core model runs:
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M0_SFT__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M1_GRPO_THROUGHPUT__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M2_GRPO_ENERGY__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M3_GRPO_IPW_BLEND__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M4_RUNTIME_PPO__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M5_HRL__*`.

2. Cross-model/ablation runs:
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M1_M2_M3__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M0_M1_M2_M3__*`,  
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/2026-03_M_ALL__*`.

3. Figure-oriented navigation:
`/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence/results/h100/2026-03/organized_figures`.

## 5. Minimum publishable evidence checklist

1. At least 3 seeds for all primary comparisons.
2. Correctness, pass@k, runtime, joules, APJ/APW, SLA violation all reported.
3. One strong frontier figure plus one matched-runtime energy figure.
4. One failure-mechanism figure plus one robustness/generalization figure.
5. One runtime-control figure covering all regimes.
6. All claims tied to confidence intervals and clearly labeled baselines.

If all six are satisfied with the effect sizes above, the results section is strong enough for a meaningful systems+RL submission narrative.
