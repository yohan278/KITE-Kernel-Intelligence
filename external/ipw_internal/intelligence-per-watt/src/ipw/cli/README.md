# IPW CLI

Energy profiling CLI for measuring Intelligence Per Watt.

## Commands

### `ipw bench` - Run Agent Benchmarks

Run agents on benchmarks with energy telemetry.

**Usage:**
```bash
ipw bench --agent <agent> --model <model> --benchmark <benchmark> [options]
```

**Agents:**
- `react` - ReAct agent using Agno framework
- `openhands` - OpenHands SDK-based agent
- `terminus` - Terminal-based agent for Docker containers

**Model Presets:**
| Preset | Model Path | GPUs | Config |
|--------|-----------|------|--------|
| `qwen3-4b` | Qwen/Qwen3-4B-Instruct | 1 | TP=1, 32K ctx |
| `gpt-oss-120b` | openai/gpt-oss-120b | 4 | TP=4, 32K ctx |

You can also use any HuggingFace model name directly with `--vllm-url`.

**Benchmarks:**
- `hle` - Humanity's Last Exam (challenging academic questions)
- `gaia` - General AI Assistants (real-world tasks)
- `apex` - AI Productivity Index Extended (professional domains)

**Options:**
```
--agent          Required. Agent type (react, openhands, terminus)
--model          Required. Model preset or HuggingFace model name
--benchmark      Required. Benchmark to run (hle, gaia, apex)
--limit N        Max samples to evaluate
--output PATH    Output directory for results
--vllm-url URL   vLLM server URL (required for raw model names)
--per-action     Record per-action energy breakdown
--no-telemetry   Disable energy telemetry
--skip-warmup    Skip warmup (include cold-start)
--list, -l       List all options
```

**Examples:**
```bash
# Quick test with model preset
ipw bench --agent react --model qwen3-4b --benchmark hle --limit 10

# Full evaluation with per-action energy breakdown
ipw bench --agent openhands --model gpt-oss-120b --benchmark hle --per-action

# Custom model with explicit vLLM URL
ipw bench --agent react --model meta-llama/Llama-3-8B --benchmark gaia \
    --vllm-url http://localhost:8000/v1

# List all available options
ipw bench --list
```

**Output Structure:**
```
outputs/bench/
└── hle_qwen3-4b_20240128_143022/
    ├── results.json          # Full results with energy metrics
    ├── summary.json          # High-level summary
    └── per_action.json       # Per-action breakdown (if --per-action)
```

### `ipw list` - List Available Components

List available inference clients, analysis providers, and more.

```bash
ipw list
```

### `ipw analyze` - Analyze Profiling Results

Analyze profiling results from previous runs.

```bash
ipw analyze <results-file>
```

### `ipw plot` - Generate Visualizations

Generate visualizations from analyzed data.

```bash
ipw plot <results-file> --output <output-dir>
```

### `ipw profile` - Profile Inference

Profile energy consumption during inference.

```bash
ipw profile --client vllm --model <model> --dataset <dataset>
```

## Energy Metrics

The CLI provides the following energy metrics:

- **gpu_energy_joules** - GPU energy consumption
- **cpu_energy_joules** - CPU energy consumption
- **total_energy_joules** - Combined GPU + CPU energy
- **avg_gpu_power_watts** - Average GPU power draw
- **avg_cpu_power_watts** - Average CPU power draw
- **duration_seconds** - Total benchmark duration
- **ipw_score** - Intelligence Per Watt (accuracy / total_energy)

## Per-Action Telemetry

With `--per-action`, the CLI records energy per agent action:

- **lm_inference** - Energy for LLM inference calls
- **tool_call** - Energy for tool execution

This enables detailed analysis of where energy is spent during agent execution.
