# grid_eval

Grid search evaluation framework for benchmarking model/agent combinations across different hardware configurations with energy profiling.

## Purpose

grid_eval orchestrates large-scale evaluations that combine:
- Multiple LLM models (Qwen3 family, GPT-OSS, etc.)
- Multiple agent implementations (ReAct, OpenHands)
- Multiple benchmarks (GAIA, HLE)
- Multiple hardware configurations (1 GPU, 4 GPUs)

It integrates with IPW's energy monitoring to capture power consumption and performance metrics for each configuration.

## Installation

grid_eval requires the base IPW package with agents and evals extras:

```bash
uv pip install -e 'intelligence-per-watt[agents,evals]'
```

## CLI Usage

Run as a Python module from the intelligence-per-watt directory:

```bash
# Full grid search (all combinations)
python -m grid_eval --output-dir results/grid_eval

# Subset for testing
python -m grid_eval \
    --benchmarks hle \
    --models qwen3-8b \
    --agents react \
    --hardware a100_1gpu \
    --queries 5

# Dry run to preview configuration
python -m grid_eval --dry-run
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Output directory for results | `results/grid_eval_YYYYMMDD_HHMMSS` |
| `-b, --benchmarks` | Comma-separated benchmarks (hle, gaia) | all |
| `-m, --models` | Comma-separated models | all |
| `-a, --agents` | Comma-separated agents (react, openhands) | all |
| `-h, --hardware` | Comma-separated hardware configs | all |
| `-q, --queries` | Queries per benchmark | 100 |
| `-s, --seed` | Random seed | 42 |
| `--vllm-url` | vLLM server base URL | http://localhost:8000 |
| `--dry-run` | Show config without running | - |

## Programmatic Usage

```python
from grid_eval import GridConfig, GridEvalRunner, BenchmarkType, ModelType, AgentType, HardwareConfig

# Configure evaluation grid
config = GridConfig(
    benchmarks=[BenchmarkType.GAIA],
    models=[ModelType.QWEN3_8B, ModelType.QWEN3_14B],
    agents=[AgentType.REACT],
    hardware_configs=[HardwareConfig.A100_1GPU],
    queries_per_benchmark=50,
    seed=42,
)

# Run evaluation
runner = GridEvalRunner(config)
runner.run(output_dir="results/my_experiment")
```

## Configuration Options

### Benchmarks
- `hle` - HLE benchmark
- `gaia` - GAIA benchmark (supports level filtering via `gaia_level`)

### Models
- `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `qwen3-32b` - Qwen3 dense models
- `qwen3-30b-a3b` - Qwen3 MoE (30B total, 3B active)
- `gpt-oss-120b` - GPT-OSS 120B MoE

### Agents
- `react` - ReAct agent
- `openhands` - OpenHands agent

### Hardware Configurations (from IPW paper)
- `a100_1gpu` - Single A100 80GB GPU with 8 CPU threads
- `a100_4gpu` - Four A100 80GB GPUs with 32 CPU threads
- `h100_1gpu` - Single H100 80GB GPU with 8 CPU threads
- `h100_4gpu` - Four H100 80GB GPUs with 32 CPU threads
- `gh200_1gpu` - GH200 (96GB GPU + 480GB unified) with 72 CPU threads
- `b200_1gpu` - Single B200 192GB GPU with 8 CPU threads

## Output Format

Results are written to the output directory:

```
results/grid_eval_YYYYMMDD_HHMMSS/
    results_YYYYMMDD_HHMMSS.jsonl   # Per-query results
    summary_YYYYMMDD_HHMMSS.json    # Aggregated metrics
    metadata_YYYYMMDD_HHMMSS.json   # Configuration metadata
```

### Query Result Fields (JSONL)

Each line contains:
- `query_id` - Unique query identifier
- `benchmark`, `model`, `agent`, `hardware` - Configuration
- `avg_joules` - Energy consumption (Joules)
- `max_power_watts` - Peak power draw
- `latency_seconds` - Total query time
- `turns` - Number of LLM inference turns
- `is_correct` - Whether response matched ground truth
- `response`, `ground_truth` - Full text
- `total_params_b`, `active_params_b` - Model size info

### Summary Fields (JSON)

Aggregated statistics per configuration:
- Mean, std, min, max for energy and latency
- Accuracy (fraction correct)
- Total queries and errors
