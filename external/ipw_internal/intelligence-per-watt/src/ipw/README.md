# IPW Profiler Source

Energy profiling and benchmarking tools for measuring Intelligence Per Watt.

## Directory Structure

```
src/
├── cli/              # CLI commands (bench, servers, profile, analyze)
├── telemetry/        # Energy data collection
│   ├── collector.py      # gRPC client for energy-monitor binary
│   ├── events.py         # Per-action event recording
│   └── correlation.py    # Energy-to-event correlation
├── execution/        # Benchmark orchestration
├── analysis/         # Statistical analysis tools
├── visualization/    # Plotting and charts
└── core/             # Registry and shared types
```

## Key Commands

```bash
ipw bench --client ollama --model llama3.2:1b --dataset gaia
ipw servers launch --ollama --model llama3.2:1b
ipw servers status
```

## Energy Collection

Uses cumulative energy counters from:
- GPU: NVML (NVIDIA) / ROCm SMI (AMD)
- CPU: Intel RAPL

Energy delta = `counter_end - counter_start` (excludes warmup).
