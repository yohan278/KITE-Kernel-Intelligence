# Building Guide — Dataset Generator & Inference Simulator

## Quick Start

```bash
# From the repo root
cd intelligence-per-watt

# Create venv and install with dataset-generator extras
uv venv && source .venv/bin/activate
uv pip install -e '.[dataset-generator]'
```

## Package Structure

```
src/
  inference_simulator/       # Shared types (ModelSpec, HardwareSpec, OperatorMeasurement, etc.)
    types/                   # Core dataclasses used across all three packages
  dataset_generator/         # Pipeline #1: operator-level profiling on real hardware
    model_loader/            # HuggingFace config.json → ModelSpec
    profiler/                # Token ops, attention, agentic profilers
    cli.py                   # Click CLI entry point
```

## Running the Profiler

```bash
# Full profiling run (requires GPU + torch)
python -m dataset_generator.cli profile \
  --model Qwen/Qwen3-8B \
  --hardware a100_80gb \
  --precision fp16

# With custom sweep
python -m dataset_generator.cli profile \
  --model Qwen/Qwen3-8B \
  --hardware a100_80gb \
  --batch-sizes 1,4,16 \
  --seq-lengths 128,512,2048 \
  --warmup 3 \
  --iterations 10

# Run specific profilers only
python -m dataset_generator.cli profile \
  --model Qwen/Qwen3-8B \
  --hardware a100_80gb \
  --profilers token_ops,attention

# List available models and operators
python -m dataset_generator.cli list-models
python -m dataset_generator.cli list-operators
```

## Running Tests

```bash
cd intelligence-per-watt

# All dataset_generator + inference_simulator tests
python -m pytest tests/dataset_generator/ tests/inference_simulator/ -v

# Skip tests requiring GPU
python -m pytest tests/dataset_generator/ tests/inference_simulator/ -v -m "not integration"

# Single test file
python -m pytest tests/dataset_generator/test_model_loader.py -v
```

## Import Check

```python
from inference_simulator.types import ModelSpec, HardwareSpec, OperatorMeasurement
from dataset_generator.profiler.token_ops import TokenOpProfiler
from dataset_generator.model_loader.qwen3 import Qwen3ModelLoader
```

## Output Format

Profiling outputs are written to `data/profiles/{model}/{hardware}/{precision}/`:

- `token_ops.csv` — Linear, norm, activation, embedding operators
- `attention.csv` — Prefill and decode attention
- `agentic.csv` — Tool server latency distributions
