# Agent pl0

## Objective
Generate figure artifacts for the paper/report from aggregated stats.

## Isolation Rules

1. Do not edit experiment configs.
2. Write figure outputs only under `outputs/paper/figures`.

## Prerequisites

- outputs/agent_queue/state/st0.done
- outputs/agent_queue/stats_summary.json

## Steps

```bash
bash scripts/agents/wrappers/pl0.sh
```

## Expected Artifacts

- outputs/paper/figures/avg_reward_by_stage.png
- plus outputs from `scripts/09_plot_pareto.py` if available
- outputs/agent_queue/state/pl0.done

## Validation Checklist

1. Figure file exists and is non-empty.
2. Plot labels are readable and stage ordering is correct.

## Handoff Message

```text
[pl0] complete: figure artifacts generated in outputs/paper/figures
```
