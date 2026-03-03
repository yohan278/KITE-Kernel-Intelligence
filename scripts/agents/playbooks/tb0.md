# Agent tb0

## Objective
Generate publication-ready summary tables from stats outputs.

## Isolation Rules

1. Read stats only.
2. Write tables only under `outputs/paper/tables`.

## Prerequisites

- outputs/agent_queue/state/st0.done
- outputs/agent_queue/stats_summary.json

## Steps

```bash
bash scripts/agents/wrappers/tb0.sh
```

## Expected Artifacts

- outputs/paper/tables/stage_metrics.md
- outputs/agent_queue/state/tb0.done

## Validation Checklist

1. Table has per-stage rows.
2. Includes `n`, `mean`, `std`, `ci95` columns.

## Handoff Message

```text
[tb0] complete: stage_metrics.md generated for paper table import.
```
