# CLI Energy Profiling Tests

Tests for validating the `ipw bench` CLI energy profiling functionality.

## What These Tests Validate

1. **Energy telemetry collection** - Verifies GPU/CPU energy is measured during inference
2. **Warmup exclusion** - Confirms startup/warmup costs are excluded from measurements
3. **Per-action breakdowns** - Tests fine-grained energy attribution to individual actions
4. **Time windowing** - Validates that only inference energy is captured, not overhead
5. **Auto-server management** - Tests automatic vLLM server lifecycle with proper exclusion

## Prerequisites

- Linux x86_64 with NVIDIA GPU(s) (tested on A100)
- Python 3.12+
- vLLM installed
- ipw package installed

## Quick Start

### 1. Install the package

```bash
cd intelligence-per-watt
pip install -e .
```

### 2. Build the energy monitor binary

```bash
cd ..  # back to repo root
uv run scripts/build_energy_monitor.py --profile release
```

### 3. Start a vLLM server (or use --auto-server)

```bash
# Option A: Manual server
vllm serve Qwen/Qwen3-4B-Instruct --port 8000

# Option B: Let the test manage servers (uses --auto-server)
# No setup needed
```

### 4. Run the tests

```bash
cd intelligence-per-watt/tests/cli_energy_profiling

# Run all tests
python test_bench_energy.py --output ./results

# Run with specific model
python test_bench_energy.py --model qwen3-8b --output ./results

# Run with external vLLM server
python test_bench_energy.py --vllm-url http://localhost:8000/v1 --output ./results
```

## Test Cases

| Test | Description | Key Validation |
|------|-------------|----------------|
| `basic_benchmark` | Standard benchmark with telemetry | Energy metrics in output |
| `per_action_breakdown` | `--per-action` flag enabled | Action-level energy attribution |
| `no_telemetry_baseline` | `--no-telemetry` flag | Runs without energy collection |
| `skip_warmup` | `--skip-warmup` flag | Cold-start costs included |
| `auto_server` | `--auto-server` flag | Server lifecycle excluded from profiling |

## Expected Output

A successful test run will show:

```
============================================================
TEST SUMMARY
============================================================
Passed: 5/5
  [PASS] basic_benchmark
  [PASS] per_action_breakdown
  [PASS] no_telemetry_baseline
  [PASS] skip_warmup
  [PASS] auto_server

Results saved to: ./results/test_results.json
```

## Key CLI Features Being Tested

### Energy Measurement Timing

The CLI properly excludes overhead costs:

```
1. Server startup      → EXCLUDED (before telemetry starts)
2. Warmup queries      → EXCLUDED (before telemetry starts)
3. Benchmark execution → MEASURED (time-windowed)
4. Server shutdown     → EXCLUDED (after telemetry ends)
```

### Time Window Approach

```python
with TelemetrySession(collector) as telemetry:
    start_time = time.time()
    benchmark_metrics = benchmark.run_benchmark(orchestrator)
    end_time = time.time()

    samples = list(telemetry.window(start_time, end_time))
```

Only samples within the `[start_time, end_time]` window are included in energy calculations.

## Troubleshooting

### "Energy monitor binary not found"

Build the Rust energy monitor:

```bash
uv run scripts/build_energy_monitor.py --profile release
```

### "nvidia-smi not available"

These tests require NVIDIA GPUs. Run on a machine with:
- NVIDIA drivers installed
- CUDA toolkit available
- nvidia-smi in PATH

### "ipw CLI not available"

Install the package:

```bash
pip install -e intelligence-per-watt
```

## Results

See `sample_results.json` for example output from a test run on 2x A100-80GB.
