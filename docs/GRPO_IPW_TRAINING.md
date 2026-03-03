# GRPO + IPW training: data, online vs offline, telemetry

## 1. What is the training data?

- **Source**: **KernelBench tasks** (from `KernelBenchAdapter.discover_tasks()`).
  - Tasks come from the KernelBench dataset API, or from JSONL under `kernelbench_root` (e.g. `tasks.jsonl`, `data/tasks.jsonl`), or built-in defaults.
- **Per task**: Each task has a reference PyTorch kernel (`reference_kernel` / `ref_arch_src`). The **prompt** is fixed: “Optimize this reference kernel; output valid Python with `ModelNew(nn.Module)`; no Triton.”
- **So**: Training “data” is the **set of tasks** (one prompt per task). There is no fixed offline dataset of (prompt, completion) pairs; completions are generated during training.

## 2. Is training online or offline?

- **Online**. Each training step:
  1. A **batch of prompts** is taken from the task list (one prompt per task, cycling if batch size &gt; number of tasks).
  2. The **model generates** `group_size` completions per prompt (e.g. 8 candidates per task).
  3. Each completion is **evaluated on the fly**: compile, correctness, and runtime via KernelBench.
  4. **Rewards** are computed from those outcomes (and optionally from telemetry when IPW is enabled).
  5. GRPO updates the policy from the group’s relative rewards.

So: **prompts** come from the task list; **completions are generated and evaluated online** every step. There is no separate “offline dataset” of model outputs.

## 3. Which telemetry is used when IPW is enabled?

When `energy_aware: true` (and optionally `reward.ipw_blend_weight > 0`), the **reward** can depend on energy/power. That requires a **telemetry corpus** (or a synthetic fallback).

Telemetry is **not** an input to the model. It is used only **inside the reward function** to get power/energy numbers (e.g. `avg_power_w`, `total_energy_j`) that are blended into the reward.

### How the telemetry corpus is built

`GRPOKernelTrainer` calls `EnergyCapture.load_trace_corpus(...)` with:

| Config / CLI | Meaning |
|--------------|--------|
| `train.telemetry.trace_dir` / `--telemetry-trace-dir` | Directory of **trace JSON files** (e.g. from `EnergyCapture` or your own logging). All `*.json` under this path (recursive) are loaded as `EnergyTrace`s. |
| `train.telemetry.ipw_profile_dir` / `--ipw-profile-dir` | Directory where **`ipw profile`** wrote outputs (HuggingFace `datasets` format, `load_from_disk`). Rows are turned into `EnergyTrace`s (up to 256 by default). |

- Traces from **both** sources are concatenated into one list.
- If that list is empty and `allow_synthetic_fallback` is true (default), a **single synthetic trace** is used (fake power curve, no real GPU data).
- If `allow_synthetic_fallback` is false and no traces are found, training raises an error.

### How telemetry is used during training

- For each candidate being scored, the trainer takes the **next trace** from the corpus (round-robin: `telemetry_corpus[telemetry_idx % len(telemetry_corpus)]`).
- If the trace has no `phase_segments`, it is attributed with `attribute_prefill_decode(trace, ttft_s=0.4)` (prefill vs decode).
- `IPWAdapter.summarize(trace, input_tokens=512, output_tokens=128)` produces a summary (e.g. `avg_power_w`, `total_energy_j`).
- If the candidate already has `avg_power_w` / `energy_j` from evaluation, those can override the summary.
- The reward is then computed (e.g. `compute_grpo_multi_metric_reward` and optionally `compute_ipw_reward` with `reward_ipw_blend_weight`).

So: **the telemetry data that is “used as input” to the reward** is exactly the **telemetry corpus** loaded from:

1. **`trace_dir`**: any `*.json` traces under that directory, or  
2. **`ipw_profile_dir`**: dataset saved by `ipw profile`, or  
3. **Synthetic**: one fake trace, if the corpus would otherwise be empty and fallback is allowed.

To use **real** telemetry, point `trace_dir` and/or `ipw_profile_dir` at real data; to force no synthetic fallback, set `allow_synthetic_fallback: false` (and ensure at least one trace is found).

## 4. Small GRPO + IPW run

Use a small config that turns on energy-aware rewards and (optionally) IPW blend, and limits tasks/epochs for a short run. Example: `configs/train_small_grpo_ipw.yaml`, then:

```bash
python scripts/train_rl.py --config configs/train_small_grpo_ipw.yaml \
  --kernelbench-root ./external/KernelBench \
  --output ./checkpoints/small_grpo_ipw
```

To use real telemetry, create traces (e.g. from `EnergyCapture` or `ipw profile`) and set `train.telemetry.trace_dir` and/or `train.telemetry.ipw_profile_dir` in the config (or via the CLI).
