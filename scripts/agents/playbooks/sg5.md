# Agent sg5

## Objective
Generate data-scale ablation configs.

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
bash scripts/agents/wrappers/sg5.sh
```

3. Verify the done marker:

```bash
ls outputs/agent_queue/state/sg5.done
```

## Expected Artifacts

- configs/exp/abl_scale/*.yaml

## Validation Checklist

1. No YAML syntax errors.
2. Files are under `configs/exp/...` only.
3. `outputs/agent_queue/state/sg5.done` exists.

## Handoff Message

```text
[sg5] complete: generated configs, wrote done marker at outputs/agent_queue/state/sg5.done
```
