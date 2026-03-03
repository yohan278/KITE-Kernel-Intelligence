# Agent sv0

## Objective
Validate all generated experiment configs and create the canonical run manifest and queue database.

## Isolation Rules

1. Do not create new experiment configs.
2. Only validate and materialize queue metadata.
3. Do not run training jobs.

## Prerequisites

All setup generators must be complete:

- outputs/agent_queue/state/sg0.done ... sg7.done

## Steps

1. Run validator and manifest builder:

```bash
bash scripts/agents/wrappers/sv0.sh
```

2. Verify outputs:

```bash
ls outputs/agent_queue/manifest.csv
ls outputs/agent_queue/queue.db
ls outputs/agent_queue/state/sv0.done
```

## Expected Artifacts

- outputs/agent_queue/manifest.csv
- outputs/agent_queue/queue.db
- outputs/agent_queue/state/sv0.done

## Validation Checklist

1. manifest.csv has one row per scheduled run.
2. queue.db contains `runs` table with status `queued` rows.
3. No missing required train fields in generated configs.

## Handoff Message

```text
[sv0] complete: manifest and sqlite queue generated; queue ready for ex0..ex5 claim.
```
