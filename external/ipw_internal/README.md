# Intelligence Per Watt

<p align="center">
  <img src="assets/intelligence_per_watt_mood.png" width="500" alt="Intelligence Per Watt">
</p>

A benchmarking suite for LLM inference systems. Intelligence Per Watt sends workloads to your inference service and collects detailed telemetry—energy consumption, power usage, memory, temperature, and latency—to help you optimize performance and compare hardware configurations.

## Installation

### Prerequisites
- [Rust compiler](https://www.rust-lang.org/tools/install) (for building energy monitor)
- [Protocol Buffer compiler](https://protobuf.dev/installation/) (`protoc`)
- [Ollama](https://ollama.ai/) or [vLLM](https://docs.vllm.ai/) (inference client)

### Setup
```bash
git clone https://github.com/HazyResearch/intelligence-per-watt.git

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Build energy monitoring
uv run scripts/build_energy_monitor.py

# Install Intelligence Per Watt
uv pip install -e intelligence-per-watt
```

Optional inference clients ship as extras—install each one you need from the package directory, e.g. `uv pip install -e 'intelligence-per-watt[ollama]'` or `uv pip install -e 'intelligence-per-watt[vllm]'`.

## Quick Start

```bash
# 1. List available inference clients
ipw list clients

# 2, Run a benchmark
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434

# 3. Analyze the results
ipw analyze ./runs/profile_*

# 4. Generate plots
ipw plot ./runs/profile_*
```

**What gets measured:** For each query, Intelligence Per Watt captures energy consumption, power draw, GPU/CPU memory usage, temperature, time-to-first-token, throughput, token counts, and hardware utilization (GPU compute/memory utilization plus derived MFU/MBU and arithmetic intensity).

## Commands

### `ipw profile`

Sends prompts to your service and measures performance.

```bash
ipw profile --client <client> --model <model> [options]
```

**Options:**
- `--client` - Inference client (e.g., `ollama`, `vllm`)
- `--model` - Model name
- `--client-base-url` - Service URL (e.g., `http://localhost:11434`)
- `--dataset` - Workload dataset (default: `ipw`)
- `--max-queries` - Limit queries for testing
- `--output-dir` - Where to save results
- `--phased` - Enable phase-aware profiling (prefill vs. decode energy/power)

Notes:
- `--phased` is optional. Omitting it preserves the default aggregate-only behavior.
- Phase-specific analyses/plots require runs captured with `--phased`.

Example:
```bash
ipw profile \
  --client ollama \
  --model llama3.2:1b \
  --client-base-url http://localhost:11434 \
  --phased \
  --max-queries 100
```

### `ipw analyze`

Compute regression metrics (e.g., how energy scales with tokens, latency vs. input size).

```bash
ipw analyze <results_dir> [--analysis <type>]
```

Phase-aware analyses (when profiled with `--phased`):

```bash
# Prefill vs. decode summary statistics
ipw analyze ./runs/profile_* --analysis phased

# Separate regressions for prefill and decode
ipw analyze ./runs/profile_* --analysis phase-regression
```

### `ipw plot`

Visualize profiling data (scatter plots, regression lines, distributions).

```bash
ipw plot <results_dir> [--viz <type>] [--output <dir>]
```

Notes:
- `--viz` is optional. Without it, IPW generates the default plot set.
- Use `--viz phase-*` to target only phase-aware plots.

Phase-aware visualizations:

```bash
# Stacked prefill vs. decode energy
ipw plot ./runs/profile_* --viz phase-comparison

# Phase regions over a power timeline
ipw plot ./runs/profile_* --viz phase-power-timeline

# Phase-specific scatter plots
ipw plot ./runs/profile_* --viz phase-scatter
```

### `ipw list`

Discover available clients, datasets, and analysis types.

```bash
ipw list <clients|datasets|analyses|visualizations|all>
```

### Energy monitor test script

Validate that your system can collect energy telemetry before running full workloads.

```bash
uv run scripts/test_energy_monitor.py [--interval 2.0]
```

## Output

Profiling runs save to `./runs/profile_<hardware>_<model>/`:

```
runs/profile_<hardware>_<model>/
├── data-*.arrow        # Per-query metrics (HuggingFace dataset format)
├── summary.json        # Run metadata and totals
├── analysis/           # Regression coefficients, statistics
└── plots/              # Graphs
```

## Development Tools

### Roborev (Optional)

This repository includes configuration for [roborev](https://github.com/roborev-dev/roborev), a continuous code review daemon that automatically reviews commits using AI agents.

**Setup:**

```bash
# 1. Install Go (if not already installed)
# macOS: brew install go
# Ubuntu/Debian: sudo apt install golang
# Amazon Linux/Fedora: sudo dnf install golang

# 2. Install roborev
go install github.com/roborev-dev/roborev/cmd/roborev@latest

# 3. Add Go bin to PATH (add to ~/.bashrc or ~/.zshrc for persistence)
export PATH=$PATH:$HOME/go/bin

# 4. Initialize in the repository
roborev init

# 5. (Optional) Set claude-code as default agent in ~/.roborev/config.toml
#    default_agent = "claude-code"
```

**Requirements:**
- Go 1.21+
- `ANTHROPIC_API_KEY` environment variable (for claude-code agent)

**Usage:**

```bash
# Check daemon status
roborev status

# View review for latest commit
roborev show HEAD

# Review specific files
roborev review src/ipw/execution/runner.py

# Run code analysis
roborev analyze complexity src/ipw/execution/runner.py
roborev analyze duplication src/ipw/

# Apply suggested fixes (creates roborev/ branch)
roborev fix

# Interactive terminal UI
roborev tui
```

The repository includes `.roborev.toml` with project-specific review guidelines covering energy measurement accuracy, null handling, type safety, and architectural patterns.
