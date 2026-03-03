# Agent mn0

## Objective
Monitor queue state and provide live progress visibility.

## Isolation Rules

1. Read queue metadata only.
2. Do not mutate run configurations.

## Prerequisites

- outputs/agent_queue/state/sv0.done
- outputs/agent_queue/queue.db

## Steps

```bash
bash scripts/agents/wrappers/mn0.sh
```

This prints periodic status counts:

- queued
- prepared
- running
- done
- failed

## Expected Artifacts

- outputs/agent_queue/state/mn0.done
- outputs/agent_queue/logs/mn0.log

## Validation Checklist

1. Monitor output includes total and per-status counts.
2. It exits when queue is drained (`queued=0`, `running=0`) or by operator stop.

## Handoff Message

```text
[mn0] complete: queue monitoring finished; final status summary available in log.
```
