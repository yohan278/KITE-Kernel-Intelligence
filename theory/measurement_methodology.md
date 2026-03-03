# Energy Measurement Methodology

This document describes the challenges and best practices for measuring GPU
energy consumption, explaining the design decisions in our profiling scripts.

---

## 1  What NVML Provides

NVIDIA Management Library (NVML) exposes `nvmlDeviceGetPowerUsage()`, which
returns instantaneous power in milliwatts.  Under the hood:

- A **current sense resistor** on the board measures total current from the
  12V input rail
- The onboard microcontroller multiplies current × voltage to get power
- This value is exposed through the NVML API and `nvidia-smi`

### 1.1  Sampling Rate Limitations

| GPU Generation | Sensor update rate | Effective resolution |
|---------------|-------------------|---------------------|
| Kepler–Pascal | ~66 ms | Poor for short kernels |
| Volta–Ampere | ~20-50 ms | Marginal for ms-scale kernels |
| Ada Lovelace (L40S) | ~20-50 ms | Our experimental platform |

Because the sensor averages over its update window, a kernel running for 0.5 ms
will appear "smeared" across the 20-50 ms power sample.  To get meaningful
energy measurements, we must either:

1. **Run many iterations** of the kernel under continuous NVML sampling, or
2. **Use CUDA events** for precise timing and combine with average power

Our approach uses both: we run multiple trials with CUDA event timing while
continuously sampling NVML power at 20 ms intervals.

---

## 2  Energy Calculation

Energy is power integrated over time:

```
E = ∫ P(t) dt ≈ Σ P_i × Δt_i
```

In our `NvmlRichSampler`, we:

1. Sample power every 20 ms in a background thread
2. Record timestamps for each sample
3. After the kernel(s) finish, integrate using the trapezoidal rule:

```python
energy_j = 0.0
for i in range(1, len(samples)):
    dt = samples[i].timestamp - samples[i-1].timestamp
    avg_power = (samples[i].power_w + samples[i-1].power_w) / 2
    energy_j += avg_power * dt
```

### 2.1  Sources of Error

| Source | Magnitude | Mitigation |
|--------|----------|------------|
| Sensor averaging window | ±10-20% for short kernels | Run multiple trials |
| Sampling aliasing | Misses power spikes between samples | 20 ms interval, trapezoidal rule |
| Leakage included | Adds ~30-50 W constant offset | Idle power subtraction (future) |
| DVFS transients | First run uses 10-80% more energy | Warmup runs before measurement |
| Thermal drift | Power increases as GPU heats up | Short experiments, controlled conditions |
| Other processes | Noise from OS, display, etc. | Headless server, exclusive GPU access |

---

## 3  Our Profiling Protocol

The protocol used in `energy_profiling_experiment.py`:

```
1.  Clear GPU memory (torch.cuda.empty_cache())
2.  Load the reference kernel source via exec()
3.  Instantiate the Model class on GPU
4.  Generate inputs via get_inputs() and move to GPU
5.  Run 3 warmup iterations (to stabilize DVFS and fill caches)
6.  Start NVML sampling at 20 ms intervals
7.  Run N trials (default 3) with CUDA event timing
8.  Stop NVML sampling
9.  Compute median runtime from CUDA events
10. Integrate NVML samples for total energy
11. Clean up (delete model, inputs, empty cache)
```

### 3.1  Why exec() Instead of import

KernelBench stores kernel source as Python strings, not importable modules.
The reference architecture defines a `Model` class and a `get_inputs()` function
inline.  We use `exec(ref_arch_src, namespace)` to load these into a fresh
namespace, avoiding class hierarchy conflicts that arise when mixing with
KernelBench's evaluation machinery (which expects a `ModelNew` subclass).

### 3.2  Why Median Runtime

We use median rather than mean because:
- **Outlier resistance:** Occasional GC pauses, DVFS state changes, or context
  switches can produce extreme runtime values
- **Better central tendency:** For our typical 3-5 trials, median is more robust
  than mean against a single bad measurement

---

## 4  Input Size Scaling Methodology

The `input_size_scaling_experiment.py` varies input dimensions to observe how
energy and runtime scale with problem size.

### 4.1  Scaling Approach

We use regex substitution on the kernel source code to modify size-related
variables:

```python
patterns = [
    (r'(batch_size\s*=\s*)\d+', scale_int),
    (r'(in_channels\s*=\s*)\d+', scale_int),
    (r'(out_channels\s*=\s*)\d+', scale_int),
    (r'(dim\s*=\s*)\d+', scale_int),
    (r'(seq_len\s*=\s*)\d+', scale_int),
    (r'(height\s*=\s*)\d+', scale_int),
    (r'(width\s*=\s*)\d+', scale_int),
    (r'(size\s*=\s*)\d+', scale_int),
]
```

Each value is multiplied by a scale factor: 0.5×, 1.0×, 2.0×, 4.0×.

### 4.2  What Scaling Reveals

- **Linear energy scaling** → memory-bound kernel (energy proportional to data size)
- **Superlinear energy scaling** → compute-bound kernel (energy grows faster
  than data, e.g., O(N³) for matmul)
- **Sub-linear energy scaling** → amortization of fixed costs (kernel launch,
  cache warmup)
- **Energy/runtime ratio changes** → DVFS behavior under different loads

### 4.3  Limitations

- Regex-based scaling is fragile — it can miss variables with non-standard names
  or accidentally modify unrelated values
- Some kernels have minimum size requirements or alignment constraints
- Very large scales may trigger OOM errors (we catch and skip these)
- Scaling all dimensions uniformly doesn't match real workload scaling patterns
  (e.g., batch size scales differently from hidden dimension)

---

## 5  Feature Extraction from Source Code

We extract static features from kernel source using AST analysis and regex:

| Feature | How extracted | What it indicates |
|---------|-------------|------------------|
| `num_ops` | Count of arithmetic ops in AST | Compute density |
| `num_memory_ops` | Count of indexing operations | Memory access density |
| `uses_shared_memory` | Regex for `shared_memory`, `__shared__` | On-chip data reuse |
| `uses_custom_cuda` | Regex for `cuda_source`, `load_inline` | Hand-optimized kernel |
| `num_loops` | Count of `for`/`while` in AST | Iteration complexity |
| `num_parameters` | Count of `nn.Parameter` | Model state size |
| `total_lines` | Line count | Code complexity proxy |

These features, combined with energy measurements, let us identify which
code-level patterns correlate with energy efficiency.

---

## 6  Statistical Considerations

### 6.1  Variance Within Kernel Types

We observe high variance in energy measurements within the same kernel type.
This is expected because:

- Kernels within a type have vastly different sizes (a 64×64 matmul vs 4096×4096)
- Code quality varies (some reference kernels are naive, others optimized)
- Different kernels stress different memory access patterns

For meaningful cross-type comparisons, we normalize by input size or use
**energy per millisecond** (power) rather than raw energy.

### 6.2  Reproducibility

Energy measurements are inherently noisier than runtime measurements because:
- Power sensors have lower resolution than CUDA timers
- Temperature affects both leakage and DVFS decisions
- GPU boost clocks vary between runs based on thermal history

We recommend at least 3 trials per measurement point and report median values.
For publication-quality results, 10+ trials with confidence intervals would be
appropriate.
