# SWE-bench Benchmark

Evaluates AI agents on real-world GitHub issues from SWE-bench datasets.

## Supported Datasets

| Dataset | Flag | Samples | Source |
|---------|------|---------|--------|
| Verified Mini | `--dataset verified_mini` (default) | 50 | [MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) |
| Verified | `--dataset verified` | 500 | [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) |

## Setup

### 1. Initialize Submodules

This benchmark uses OpenHands benchmarks as a git submodule. Initialize it first:

```bash
cd /path/to/ipw_internal
git submodule update --init --recursive
```

### 2. Install Dependencies

```bash
cd intelligence-per-watt/evals

# Option A: Run setup script (installs everything)
./setup.sh

# Option B: Manual install (custom agent only)
uv sync
```

<details>
<summary>Manual OpenHands setup (if not using setup.sh)</summary>

If you only ran `uv sync` but want to use `--agent-type openhands`:

```bash
cd src/benchmarks/swebench/openhands-benchmarks
UV_INSECURE_NO_ZIP_VALIDATION=1 uv sync
cd ../../../..
```

</details>

### 3. Set Environment Variables

Create `.env` in the `evals/` directory:

## Running the Benchmark

### Quick Test (1 sample, no evaluation)

```bash
cd intelligence-per-watt/evals

# Custom agent
uv run python scripts/run_swebench.py --agent-type react --limit 1 --no-eval

# OpenHands agent
uv run python scripts/run_swebench.py --agent-type openhands --limit 1 --no-eval
```

### Full Evaluation

```bash
# Run on 5 samples with swebench harness evaluation
uv run python scripts/run_swebench.py --agent-type react --limit 5

# Run on all 50 samples (verified_mini)
uv run python scripts/run_swebench.py --agent-type react

# Run on verified dataset (500 samples)
uv run python scripts/run_swebench.py --dataset verified --limit 10

# Run 10 samples in parallel with 4 workers (4x faster)
uv run python scripts/run_swebench.py --limit 10 --num-workers 4
```

### All Options

```bash
uv run python scripts/run_swebench.py --help
```

| Option | Description |
|--------|-------------|
| `--dataset` | `verified_mini` (50 samples) or `verified` (500 samples) |
| `--agent-type` | `react` or `openhands` |
| `--model` | Model name (default: `gpt-4o`) |
| `--provider` | `openai` or `anthropic` |
| `--limit` | Number of samples to run |
| `--max-iterations` | Max agent iterations per sample |
| `--num-workers` | Parallel workers for running agents (default: 1 = sequential) |
| `--no-eval` | Skip swebench harness evaluation |

## Architecture

### Custom Agent (Hybrid)

Uses our own orchestrators (currently React, more coming). Agent runs on host, tools execute in container.

```
┌─────────────────────┐     ┌─────────────────────┐
│   HOST              │     │  Docker Container   │
├─────────────────────┤     ├─────────────────────┤
│ • Custom agent      │────▶│ • bash commands     │
│ • LLM API calls     │     │ • git operations    │
│ • Tool orchestration│◀────│ • file edits        │
└─────────────────────┘     └─────────────────────┘
```

- **Fast on any platform** - Agent runs natively, only commands run in container
- Uses `swe_env_wrapper.py` for Docker management
- Tools defined in `container_tools.py`

**Tools (matching SWE-agent's default config exactly):**

| Tool | Source in SWE-agent | Description |
|------|---------------------|-------------|
| `bash` | `enable_bash_tool: true` | Execute any bash command |
| `str_replace_editor` | `tools/edit_anthropic/config.yaml` | View/create/edit files (5 sub-commands) |
| `submit` | `tools/review_on_submit_m/config.yaml` | Submit final patch |

No search tools - agent uses `bash` + `grep`/`find` directly.

**Tool control via `run_custom_on_sample()`:**

| Parameter | Behavior |
|-----------|----------|
| `use_swe_agent_tools=True` (default) | Uses SWE-agent tools above, injects tool descriptions into system prompt |
| `use_swe_agent_tools=False` | No tools passed - orchestrator handles its own |

### OpenHands Agent (Container-based)

```
┌─────────────────────┐     ┌─────────────────────┐
│   HOST              │     │  Docker Container   │
├─────────────────────┤     ├─────────────────────┤
│ • Launches          │────▶│ • Agent server      │
│   container         │     │ • LLM API calls     │
│                     │◀────│ • Tool execution    │
└─────────────────────┘     └─────────────────────┘
```

- Uses OpenHands benchmarks CLI (`swebench-infer`)
- Requires pre-built agent-server images from GHCR

## File Structure

```
swebench/
├── README.md              # This file
├── __init__.py            # Package exports
├── dataset.py             # Loads SWE-bench samples from HuggingFace
├── swe_env_wrapper.py     # Docker container management (docker-py)
├── container_tools.py     # 3 tools matching SWE-agent's default config
├── custom_runner.py       # Custom agent runner (React, etc.)
├── openhands_runner.py    # OpenHands CLI wrapper
├── main.py                # Benchmark orchestration
└── openhands-benchmarks/  # Git submodule (OpenHands/benchmarks)
    └── vendor/software-agent-sdk/  # Nested submodule (OpenHands SDK)
```

## Troubleshooting

### "ZIP file contains multiple entries" error

This is a [uv bug](https://github.com/astral-sh/uv) with the `multi-swe-bench` wheel. Use the workaround:

```bash
UV_INSECURE_NO_ZIP_VALIDATION=1 uv sync
```

### Docker images not found

First run pulls large Docker images (~2GB each). Ensure Docker is running and you have disk space.

### Cleanup

```bash
# Stop containers
docker ps -q --filter "name=agent-server" | xargs -r docker stop
docker ps -q --filter "name=sweb" | xargs -r docker stop
docker container prune -f

# Clear outputs
rm -rf openhands-benchmarks/eval_outputs
rm -rf ../../swebench_results
```

## Output

Results are saved to `evals/swebench_results/`:
- `swebench_{timestamp}_predictions.jsonl` - Generated patches
- `swebench_{timestamp}_results.json` - Metrics and per-sample results

## References

- [SWE-bench Verified Mini Dataset](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)
- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
- [OpenHands Benchmarks](https://github.com/OpenHands/benchmarks)

