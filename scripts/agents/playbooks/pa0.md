# Agent pa0

## Objective
Parse completed checkpoint artifacts into a normalized metrics stream.

## Isolation Rules

1. Read from `checkpoints/exp/**/checkpoint.json` only.
2. Write parsed output only under `outputs/agent_queue`.

## Prerequisites

- outputs/agent_queue/state/ex0.done ... ex5.done
- GPU jobs should have been executed so checkpoints exist.

## Steps

```bash
bash scripts/agents/wrappers/pa0.sh
```

## Expected Artifacts

- outputs/agent_queue/parsed_metrics.jsonl
- outputs/agent_queue/state/pa0.done

## Validation Checklist

1. One JSONL row per discovered checkpoint.
2. Row fields include stage/mode/train_loss/avg_reward where available.

## Handoff Message

```text
[pa0] complete: parsed checkpoint metrics into outputs/agent_queue/parsed_metrics.jsonl
```
