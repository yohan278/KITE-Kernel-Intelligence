# Agent ex2

## Objective
Claim queued runs and generate runnable GPU job scripts for worker slot 2.

## Isolation Rules

1. This agent only prepares jobs and never executes experiments.
2. Do not execute training jobs; this agent is prep-only by design.
3. Only write under `outputs/agent_queue/jobs`, `outputs/agent_queue/logs`, and queue status updates.

## Prerequisites

- outputs/agent_queue/state/sv0.done
- outputs/agent_queue/queue.db

## Steps (CPU-only prep)

```bash
bash scripts/agents/wrappers/ex2.sh
```

This will:

1. Claim jobs from queue (worker hint aware).
2. Generate shell scripts in `outputs/agent_queue/jobs/`.
3. Mark queue rows as `prepared`.
4. Write done marker `outputs/agent_queue/state/ex2.done`.
## Expected Artifacts

- outputs/agent_queue/jobs/*.sh
- outputs/agent_queue/state/ex2.done
- queue status transitions (`queued` -> `prepared`)

## Validation Checklist

1. Job scripts exist and are executable.
2. Queue rows assigned to this worker move to `prepared`.
3. No long-running training process starts in prep mode.

## Handoff Message

```text
[ex2] complete: prepared job scripts and updated queue statuses to prepared.
```
