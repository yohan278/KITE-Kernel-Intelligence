# Agent Scripts (Branch-Oriented, CPU Prep First)

This folder supports a multi-agent workflow where each agent runs on an individual branch/worktree.

## Agent IDs

- Generators: `sg0..sg7`
- Validator/manifest: `sv0`
- Executors: `ex0..ex5`
- Monitor: `mn0`
- Parser/stats/plots/tables: `pa0`, `st0`, `pl0`, `tb0`

## CPU-only prep mode (hard-locked)

Executors are hard-locked to prep mode, which means:

1. Claim queued runs from sqlite queue.
2. Generate runnable GPU job scripts in `outputs/agent_queue/jobs/`.
3. Mark queue status as `prepared`.

No training is run during this phase.

## Quick start

```bash
# 1) Create branches + worktrees (one per agent)
bash scripts/agents/00_create_branches_and_worktrees.sh

# 2) Generate wrapper scripts
bash scripts/agents/91_make_agent_wrappers.sh

# 3) Launch all agents in background (CPU prep)
bash scripts/agents/92_launch_all_cpu_agents.sh
```

Logs are in `outputs/agent_queue/logs/`.

## Per-agent wrappers

Wrappers are generated in `scripts/agents/wrappers/<agent>.sh`.

Examples:

```bash
bash scripts/agents/wrappers/sg1.sh
bash scripts/agents/wrappers/sv0.sh
bash scripts/agents/wrappers/ex0.sh
```

## Run prepared jobs on GPU later

```bash
bash scripts/agents/93_run_prepared_jobs_gpu.sh 0
# then execute printed job scripts with CUDA_VISIBLE_DEVICES set
```

## Queue database

`outputs/agent_queue/queue.db`

Statuses used:

- `queued`
- `prepared`
- `submitted`
- `running`
- `done`
- `failed`
