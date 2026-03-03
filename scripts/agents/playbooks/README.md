# Agent Playbooks (Isolated Environments)

These playbooks define exactly what each cloud agent must do in its own isolated environment.

## Agent Set

- Setup generators: `sg0`, `sg1`, `sg2`, `sg3`, `sg4`, `sg5`, `sg6`, `sg7`
- Validation/manifest: `sv0`
- Execution-prep workers: `ex0`, `ex1`, `ex2`, `ex3`, `ex4`, `ex5`
- Monitor: `mn0`
- Postprocessing: `pa0`, `st0`, `pl0`, `tb0`

## Global Contract

1. Each agent runs on its own branch/worktree.
2. Each agent writes only its designated artifacts.
3. Agents do **CPU-only prep by default** (`PREP_ONLY=1`).
4. GPU training/eval jobs are generated as scripts and run later.
5. Handoff between agents is file-based via `outputs/agent_queue/state/*.done` and queue DB.

## Standard Environment

```bash
export ROOT=/Users/gabrielbo/Downloads/cs234/KITE-Kernel-Intelligence
cd "$ROOT"
export KITE_CONDA_ENV=kite-train
export KITE_HF_CACHE=$HOME/.cache/kite-hf
export KITE_HF_LOCAL_FILES_ONLY=1
```

## Typical Launch Order

1. `sg0..sg7` in parallel
2. `sv0`
3. `ex0..ex5` in parallel
4. `mn0`
5. After GPU jobs complete: `pa0 -> st0 -> (pl0, tb0)`

## Wrapper Entry Points

All agents can run through:

```bash
bash scripts/agents/wrappers/<agent>.sh
```

Example:

```bash
bash scripts/agents/wrappers/sg3.sh
```
