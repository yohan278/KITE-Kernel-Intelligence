# Agent st0

## Objective
Compute summary statistics (mean, std, CI95) from parsed metrics.

## Isolation Rules

1. Do not re-parse raw checkpoints.
2. Operate only on parser output.

## Prerequisites

- outputs/agent_queue/state/pa0.done
- outputs/agent_queue/parsed_metrics.jsonl

## Steps

```bash
bash scripts/agents/wrappers/st0.sh
```

## Expected Artifacts

- outputs/agent_queue/stats_summary.json
- outputs/agent_queue/state/st0.done

## Validation Checklist

1. Stats are grouped by stage.
2. Each metric includes `n`, `mean`, `std`, `ci95`.

## Handoff Message

```text
[st0] complete: stats_summary.json generated with CI-ready aggregates.
```
