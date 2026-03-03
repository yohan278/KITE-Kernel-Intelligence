# GPU Energy Fundamentals: From Silicon to Kernel

This document traces how electrical energy is consumed inside a modern GPU,
starting at the transistor level and working up to the kernel workloads we
profile in our experiments. Understanding each layer explains *why* different
kernel types and data patterns produce different energy signatures.

---

## 1  Transistor-Level Power (VDD Domains)

Every computation on a GPU ultimately reduces to transistors switching between
0 and 1.  The dynamic power consumed by a CMOS circuit is:

```
P_dynamic = C × V² × A × f
```

| Symbol | Meaning | What controls it |
|--------|---------|-----------------|
| **C** | Capacitance of the transistor gate + interconnect wires | Fixed at fabrication |
| **V** | Supply voltage (VDD) | Set by DVFS firmware |
| **A** | Activity factor — fraction of transistors switching per cycle | Determined by the workload |
| **f** | Clock frequency | Set by DVFS firmware |

In addition to dynamic power, there is **static (leakage) power**:

```
P_static = V × I_leak
```

Leakage current flows through transistors even when they are not switching.  It
increases exponentially with temperature, creating a feedback loop: more power →
more heat → more leakage → more power.

**Key point:** Even an idle GPU dissipates significant power due to leakage.  On
an L40S at idle this is roughly 30-50 W of the 350 W TDP envelope.

### 1.1  Voltage Domains on a Modern GPU

A GPU die is not powered by a single VDD rail.  Different functional blocks sit
on separate voltage domains so they can be scaled independently:

| Domain | Covers | Typical behavior |
|--------|--------|-----------------|
| **VDD_GPU / VDD_CORE** | Streaming multiprocessors (SMs), tensor cores, warp schedulers, register files | Highest voltage, most aggressively DVFS-scaled |
| **VDD_MEM** | Memory controllers, L2 cache, GDDR6/HBM PHYs | Relatively stable; memory clock changes less often |
| **VDD_SOC / VDD_IO** | PCIe interface, video encoders, display engine, fixed-function units | Mostly constant |
| **VDD_HBM** (if HBM) | HBM stacks (not applicable to GDDR6-based L40S) | Separate rail on HBM GPUs like H100/MI300X |

This separation means a kernel that is purely compute-bound (high SM activity,
low memory traffic) draws power primarily from VDD_CORE while barely stressing
VDD_MEM.  Conversely, a memory-bound kernel stresses VDD_MEM but leaves
VDD_CORE partially idle — yet still paying leakage on the idle SMs.

---

## 2  Power Management Firmware (DVFS)

The GPU has an on-die microcontroller (PMU — Power Management Unit) that
continuously monitors:

- **Total board power** (via voltage regulator telemetry)
- **Die temperature** (via on-die thermal sensors)
- **SM activity** (via performance counters)

Based on these signals, the firmware adjusts voltage and frequency every few
microseconds:

```
High SM activity detected
  → Power approaches TDP limit (350 W on L40S)
    → PMU lowers frequency (and possibly voltage)
      → Power drops, but so does throughput
        → This is "thermal/power throttling"
```

### 2.1  Implications for Energy Measurement

The FinGraV paper (Singhania et al., 2024) demonstrated critical consequences:

1. **First-execution spike:** When a compute-heavy kernel launches from idle,
   the GPU is at high frequency + high voltage.  The sudden activity causes a
   power spike that may exceed TDP for a few milliseconds before the PMU
   throttles.  This means the first execution of a kernel consumes *more energy
   per unit work* than subsequent executions.

2. **Steady-state execution (SSE) vs steady-state power (SSP):** After 3-4
   warmup runs, execution time stabilizes (SSE).  But power may still be
   drifting as the averaging window fills up.  True steady-state power (SSP)
   requires additional runs beyond SSE.

3. **Measurement error up to 80%:** Without distinguishing SSE from SSP,
   energy measurements can be off by as much as 80% for short-running kernels
   whose execution time is shorter than the power sensor averaging window
   (~1 ms on AMD MI300X, ~50 ms for NVML on NVIDIA GPUs).

---

## 3  Where Energy Goes Inside a Kernel

When a CUDA kernel executes, the GPU's energy expenditure breaks down across
several hardware resources.  The cost per access increases dramatically as data
moves further from the compute units:

```
                    Energy per access (approximate)
    ┌─────────────────────────────────────────────────────┐
    │  Register file          ~1 pJ                       │
    │  Shared memory          ~5 pJ          (on-chip)    │
    │  L1 cache               ~5-10 pJ       (on-chip)    │
    │  L2 cache               ~50 pJ         (on-chip)    │
    │  GDDR6 / HBM            ~500-1000 pJ   (off-chip)   │
    │  PCIe (host ↔ device)   ~10,000 pJ     (off-package)│
    └─────────────────────────────────────────────────────┘
```

This 100-1000× gap between on-chip and off-chip memory access is the single
most important fact in GPU energy optimization.  It means:

- **Data reuse** (tiling into shared memory) can reduce energy by an order of
  magnitude even if the total FLOP count stays the same.
- **Memory-bound kernels** spend most of their energy on data movement, not
  computation.
- **Compute-bound kernels** spend most of their energy in the SM (ALU
  switching, register file reads), but still pay a fixed cost for loading
  operands.

### 3.1  Component-Level Breakdown (from FinGraV)

On AMD MI300X, FinGraV measured component-level power for GEMMs:

| Kernel type | SM/XCD power | Memory/IOD power | HBM power |
|-------------|-------------|------------------|-----------|
| Compute-bound GEMM (8K×8K) | **Dominant** (~70-80% of total) | Low | Moderate |
| Memory-bound GEMV (8K×1) | Moderate (still pays leakage) | **Elevated** | Elevated |

**Takeaway 1:** For compute-bound kernels, optimizing SM power (fewer
operations, lower precision, better scheduling) has the highest leverage.

**Takeaway 2:** For memory-bound kernels, compute power is still significant
due to leakage — the GPU is not power-proportional.  You pay for idle SMs.

**Takeaway 3:** Interleaving compute-bound and memory-bound kernels can better
utilize the full power budget, since they stress different domains.

---

## 4  Input Data Affects Power (±40%)

One of the most surprising recent findings (Gregersen, Patel & Choukse, 2024)
is that **the values in the input data** — not just the kernel code or matrix
shape — significantly affect GPU power consumption.

### 4.1  Why Bit Flips Matter

Dynamic power is proportional to the activity factor **A** — the fraction of
transistors that switch state each cycle.  When the GPU multiplies two numbers:

- If both operands are **zero**, the multiply-accumulate unit does minimal
  switching → low power
- If operands are **random**, bits flip frequently → high power
- If adjacent values are **similar** (high bit-level correlation), data buses
  see fewer transitions → lower power

### 4.2  Experimental Evidence

On NVIDIA GPUs, the following input variations were tested on identical GEMM
kernels (same shape, same precision):

| Input pattern | Power relative to random |
|--------------|------------------------|
| All zeros | **-38%** (lowest) |
| Sparse (90% zeros) | **-25%** |
| Uniform random | **baseline** |
| Adversarial (max bit flips) | **+2%** (near ceiling) |

Earlier work by Bhalachandra et al. (2022) found up to **67% power variation**
on A100 GPUs depending on matrix element values.

### 4.3  Implications for Our Experiments

Our profiling uses `torch.randn()` to generate inputs (random normal
distribution).  This means we are measuring near-worst-case power for data
movement.  Real inference workloads may have:

- Significant sparsity after ReLU activations (~50% zeros)
- Low-entropy embeddings with high bit similarity
- Quantized weights with limited value range

This suggests that our energy measurements may **overestimate** the energy cost
of these kernels in real inference pipelines.

---

## 5  The Full Energy Stack

Putting it all together, here is the complete path from electrical energy to
useful computation:

```
Layer 0: Wall outlet
  │  PSU efficiency loss (~5-10%)
  ▼
Layer 1: 12V rail to GPU board
  │  VRM (voltage regulator module) loss (~2-5%)
  ▼
Layer 2: VDD domains on die
  │  Split into VDD_CORE, VDD_MEM, VDD_IO
  │  Each regulated independently
  ▼
Layer 3: DVFS loop
  │  PMU adjusts V and f based on activity/temp/power
  │  Throttling occurs here when TDP is hit
  ▼
Layer 4: Static (leakage) power
  │  Always present, ~15-20% of TDP at operating temperature
  │  Cannot be eliminated without power-gating the domain
  ▼
Layer 5: Dynamic power — data movement
  │  Register → shared mem → L2 → GDDR6
  │  Each level costs 5-10× more energy
  │  Dominates memory-bound kernels
  ▼
Layer 6: Dynamic power — computation
  │  ALU operations, tensor core matrix ops
  │  Cost depends on precision (fp32 > fp16 > int8)
  │  Cost depends on input values (bit flip rate)
  │  Dominates compute-bound kernels
  ▼
Layer 7: Kernel output
  │  The useful result
  └──────────────────
```

### 5.1  What NVML Actually Measures

Our experiments use `pynvml` to sample GPU power.  NVML reports **total board
power** — this is the sum of all VDD domains including leakage, measured at the
voltage regulator output.  It corresponds to the boundary between Layer 1 and
Layer 2 above.

NVML does **not** provide:
- Per-domain breakdown (compute vs memory vs I/O)
- Separation of static vs dynamic power
- Activity factor or bit-flip-level detail

NVML's sampling rate is limited to ~50 ms on most NVIDIA GPUs.  For kernels
running in sub-millisecond time, multiple repeated executions are needed to get
meaningful energy measurements (this is why we run multiple trials and integrate
power over time).

### 5.2  What We Can Decompose With Experiments

Despite NVML's limitations, carefully designed experiments can isolate
individual layers:

| Experiment | What it isolates |
|-----------|-----------------|
| Measure idle GPU for 10 seconds | **Layer 4** — static/leakage power |
| Same kernel with zeros vs random input | **Layer 6** — input-dependent compute |
| Same kernel at fp32 vs fp16 | **Layer 6** — precision impact on ALU power |
| Same kernel with/without shared memory tiling | **Layer 5** — data movement cost |
| Same kernel run 1× vs 20× back-to-back | **Layer 3** — DVFS warmup/throttling |

---

## 6  Relevance to Our Kernel Energy Profiling

Our experiment profiles ~250 reference kernels from KernelBench across 10+
kernel types (matmul, conv, attention, norm, activation, reduction, etc.) and
measures total board energy via NVML.

### 6.1  What our current results capture

- **Total energy per kernel** (Layers 2-6 combined)
- **Average power** during execution
- **Runtime** (a proxy for how long the GPU stays energized)
- **SM utilization and memory utilization** (proxies for which domain dominates)

### 6.2  What would strengthen the analysis

Based on the energy stack above, these additional measurements would decompose
our aggregate numbers into more actionable components:

1. **Idle power baseline** → subtract to get active-only energy, isolating
   Layer 4 contribution
2. **Input sensitivity** → quantify how much of measured energy is
   data-dependent (Layer 6 variation)
3. **Precision sweep** → measure fp32 vs fp16 to estimate compute vs memory
   energy split
4. **Warmup curve** → run 20+ trials per kernel, plot energy vs trial number
   to characterize DVFS effects (Layer 3)
5. **Memory access pattern analysis** → correlate energy with
   memory-to-compute ratio from source code features

---

## References

1. Singhania, V., Aga, S., & Ibrahim, M. A. (2024). *FinGraV: Methodology for
   Fine-Grain GPU Power Visibility and Insights.* arXiv:2412.12426.

2. Gregersen, T., Patel, P., & Choukse, E. (2024). *Input-Dependent Power
   Usage in GPUs.* arXiv:2409.18324.

3. Bhalachandra, S., et al. (2022). *Understanding the Impact of Input Entropy
   on FPU, CPU, and GPU Power.* arXiv:2212.08805.

4. Patel, P., et al. (2023). *GPU Power Management: Challenges and
   Opportunities.* CAL 2023.

5. NVIDIA. (2024). *Maximizing Energy and Power Efficiency in Applications with
   NVIDIA GPUs.* NVIDIA Developer Blog.

6. NVIDIA L40S Data Sheet. (2024). 350W TDP, 48GB GDDR6 ECC, 864 GB/s
   bandwidth, PCIe Gen4 x16.
