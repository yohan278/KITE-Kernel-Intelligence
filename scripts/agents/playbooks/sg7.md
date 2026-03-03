# Agent sg7

## Objective
Generate runtime/HRL plan config metadata.

## Isolation Rules

1. Work only in your own branch/worktree.
2. Do not edit files outside `configs/exp`, `outputs/agent_queue/state`, and agent-specific logs.
3. Do not run GPU training; generate artifacts only.

## Inputs

- Repo root: `$ROOT`
- Script: `scripts/agents/10_generate_configs.py`

## Steps

1. Activate environment and move to repo root.
2. Run:

```bash
bash scripts/agents/wrappers/sg7.sh
```

3. Verify the done marker:

```bash
ls outputs/agent_queue/state/sg7.done
```

## Expected Artifacts

- configs/exp/runtime/runtime_hrl_plan.yaml

## Validation Checklist

1. No YAML syntax errors.
2. Files are under `configs/exp/...` only.
3. `outputs/agent_queue/state/sg7.done` exists.

## Handoff Message

```text
[sg7] complete: generated configs, wrote done marker at outputs/agent_queue/state/sg7.done
```
