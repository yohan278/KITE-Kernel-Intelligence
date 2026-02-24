# Intelligence Per Watt

A framework for measuring and optimizing the energy efficiency of AI agents.

## Components

| Package | Purpose |
|---------|---------|
| **src/ipw/** | Energy profiling, analysis, and simulation toolkit |
| **src/agents/** | Agent implementations with telemetry integration |
| **src/evals/** | Benchmark suite for evaluating agents |
| **src/orchestrator/** | ML training pipeline for agent orchestration |

### ipw (Energy Profiler)

Measures energy consumption during LLM inference with per-action attribution.

```bash
# Profile an inference workload
ipw profile --client ollama --model llama3

# Analyze energy data
ipw analyze ./outputs/profile-results/
```

Key features:
- GPU/CPU energy monitoring via gRPC telemetry
- Per-action energy attribution (LM inference, tool calls, idle time)
- Regression analysis for energy modeling

#### Scaling-law experiment workflow (Steps 3-5)

Use the dedicated modules for measurement sweep, aggregation, and fitting:

```bash
# 5-minute mini pilot (recommended before long sweeps)
python -m ipw.scaling_laws.mini_pilot \
  --gpu-label h100 \
  --model Qwen/Qwen3-8B

# Note: prefix caching is disabled by default for scaling-law runs
# to avoid suppressing prefill energy on repeated synthetic prompts.
# Use --no-disable-prefix-caching to opt out.

# Step 3: run phased measurement sweep with checkpointing + OOM marking
python -m ipw.scaling_laws.sweep \
  --gpu-label h100 \
  --results-root results/pilot \
  --models Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-32B \
  --quants fp16 fp8 \
  --batches 1 8 32 128 \
  --seq-ins 256 1024 4096 8192 \
  --seq-outs 256 \
  --num-samples 20

# Step 4: aggregate all summary.json files into a parquet table
python -m ipw.scaling_laws.aggregate \
  --results-root results/pilot \
  --output scaling_law_data.parquet

# Step 5: fit prefill/decode scaling laws + held-out validation report
python -m ipw.scaling_laws.fit \
  --input scaling_law_data.parquet \
  --gpu h100 \
  --output-json scaling_law_fit_report.json

# Step 6: estimate k_HW(target)/k_HW(source) + compare microbenchmark epsilons
python -m ipw.scaling_laws.hardware_normalize \
  --input scaling_law_data.parquet \
  --fit-report scaling_law_fit_report.json \
  --target-gpu a100 \
  --model-kind cross \
  --source-benchmark-json results/h100/params_hw.json \
  --source-benchmark-json results/h100/params_inference.json \
  --target-benchmark-json results/a100/params_hw.json \
  --target-benchmark-json results/a100/params_inference.json \
  --output-json hardware_normalization_report.json
```

### simulator

Predicts energy and latency for arbitrary (hardware, model, workload) combinations without running real inference. Uses a hybrid roofline + calibration model.

```bash
# Single-query prediction (pure roofline, no calibration needed)
ipw simulate --gpu h100_80gb --model qwen3-8b \
    --input-tokens 500 --output-tokens 200

# With calibration from profiling data (higher accuracy)
ipw simulate --gpu m4_max --model qwen3-4b \
    --input-tokens 33 --output-tokens 60 \
    --calibration calibration.json

# Multi-turn agentic workload
ipw simulate --gpu a100_80gb --model qwen3-8b \
    --workload agentic_reasoning --turns 5 --tool-calls 3 \
    --context-growth 300 --resource-config 4gpu_32cpu

# JSON output for programmatic use
ipw simulate --gpu h100_80gb --model qwen3-8b \
    --input-tokens 1000 --output-tokens 500 --json-output
```

Available GPUs: `a100_80gb`, `h100_80gb`, `h200`, `gh200`, `b200`, `mi300x`, `m4_max`, `m4_pro`, `m3_max`, `m3_pro`

#### Calibration workflow

Calibration fits correction factors from real profiling data, reducing prediction error from ~77% (pure roofline) to ~3% on held-out queries.

```bash
# 1. Collect E2E traces via grid_eval
python -m grid_eval.cli \
    --gpu-types m4_max --models qwen3-4b --agents react \
    --benchmarks hle --queries 25 \
    --output-dir ../results/calibration_run

# 2. Fit calibration factors (Python)
from ipw.simulator.calibration import fit_from_grid_eval, CalibrationDB
from pathlib import Path

factors = fit_from_grid_eval(Path("results.jsonl"), gpu_type="m4_max", model_type="qwen3-4b")
db = CalibrationDB()
db.add(factors)
db.save(Path("calibration.json"))

# 3. Use calibration in predictions
ipw simulate --gpu m4_max --model qwen3-4b \
    --input-tokens 33 --output-tokens 60 \
    --calibration calibration.json
```

#### Confidence levels

| Level | Meaning | Typical error |
|-------|---------|---------------|
| **high** | Exact calibration match for (gpu, model) pair | <5% |
| **medium** | Interpolated from similar configurations | 10-30% |
| **low** | Pure roofline with conservative defaults | 30-100% |

#### Roofline plots

Generate roofline visualizations comparing predicted latency, energy, throughput, and efficiency across hardware and model sizes.

```bash
# All 10 GPUs, default model sizes (4B, 8B, 14B, 32B)
ipw plot roofline

# Save to a specific directory as PDF
ipw plot roofline --output-dir plots/roofline --format pdf

# Subset of GPUs
ipw plot roofline --gpus h100_80gb,a100_80gb,b200,m4_max

# Custom model sizes and token counts
ipw plot roofline --models 1.7,4,8,32 --input-tokens 1000 --output-tokens 500

# Apple Silicon comparison only
ipw plot roofline --gpus m4_max,m4_pro,m3_max,m3_pro --output-dir plots/apple
```

This generates four plots:

| Plot | Description |
|------|-------------|
| `roofline_hardware_comparison` | Latency and energy across GPUs for each model size (side-by-side) |
| `roofline_throughput` | Output tokens/s vs model size per GPU (classic roofline) |
| `roofline_energy_latency` | Energy-latency Pareto frontier (each dot = one GPU x model combo) |
| `roofline_energy_efficiency` | Joules per output token vs model size per GPU |

Options: `--format` (png/pdf/svg), `--dpi`, `--calibration` (path to calibration JSON).

### benchmark

Runs energy microbenchmarks to extract hardware energy parameters (pJ/bit, pJ/FLOP) used by the simulator and energy model.

```bash
# Full energy characterization (memory, compute, GEMM, inference workloads)
ipw benchmark characterize --duration 5 --output results.json

# Quick characterization (fewer data points)
ipw benchmark characterize --quick --output results.json

# Inference-only benchmarks (GEMM shapes, attention, KV cache, batching)
ipw benchmark characterize --inference-only --output inference_params.json

# Also measure hardware energy floor via CUDA graphs
ipw benchmark characterize --inference-only --cuda-graphs --output params.json

# Run a single workload
ipw benchmark run --workload gemm --matrix-size 4096 --dtype fp16
ipw benchmark run --workload attention --batch-size 4 --seq-len 2048
ipw benchmark run --workload inference_gemm --batch-size 1 --seq-len 512 --mode prefill

# List available platforms and workloads
ipw benchmark platforms
```

Subcommands:

| Command | Description |
|---------|-------------|
| `characterize` | Run full microbenchmark suite and extract energy parameters |
| `run` | Run a single workload with energy measurement |
| `platforms` | List available benchmark platforms and supported workloads |

Filter flags for `characterize`: `--memory-only`, `--compute-only`, `--gemm-only`, `--inference-only`, `--no-inference`.

Workload types for `run`: `memory`, `compute`, `gemm`, `inference_gemm`, `attention`, `kv_cache`, `nccl`, `batched_decode`.

### agents

Agent implementations with built-in energy telemetry hooks.

- **ReAct** - Reasoning and acting agent (Agno-based)
- **OpenHands** - OpenHands agent wrapper with tool tracking
- **Terminus** - Terminus agent wrapper
- **MCP Servers** - Tool servers for Ollama, vLLM, OpenAI

### evals

Benchmark framework for agent evaluation.

Supported benchmarks: APEX, GAIA, SWE-bench, Tau-bench, HLE, BFCL, GDPVal, BrowseComp

### orchestrator

ML training pipeline for learning to delegate between local models, cloud APIs, and tools.

- Trajectory generation from teacher models
- SFT training on generated data
- Evaluation on downstream benchmarks

## Quick Start

```bash
# Install
uv venv && source .venv/bin/activate
uv pip install -e .

# Run tests
python -m pytest src/ipw/tests/ -v
```

## Directory Structure

```
intelligence-per-watt/
├── src/
│   ├── ipw/           # Energy profiling package (main CLI)
│   │   └── simulator/ # Inference energy/latency simulator
│   ├── agents/        # Agent implementations + MCP servers
│   ├── evals/         # Evaluation benchmarks
│   └── orchestrator/  # ML training pipeline
├── tests/             # Unit tests
└── outputs/           # Data, checkpoints, results (gitignored)
```
