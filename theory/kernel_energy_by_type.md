# Energy Characteristics by Kernel Type

This document summarizes how different GPU kernel types consume energy,
connecting our empirical profiling results with hardware-level explanations from
`gpu_energy_fundamentals.md`.

---

## Compute-Bound vs Memory-Bound: The Key Distinction

Every kernel falls somewhere on a spectrum from **compute-bound** (limited by
ALU throughput) to **memory-bound** (limited by DRAM bandwidth).  This
distinction determines which part of the GPU power budget dominates.

```
Compute-bound                                     Memory-bound
  ◄───────────────────────────────────────────────────►
  matmul    attention   conv    norm  activation  reduction
  (high)                                          (low)
            ←── arithmetic intensity ──→
```

**Arithmetic intensity** = FLOPs per byte loaded from memory.

- High arithmetic intensity → compute-bound → SM power dominates
- Low arithmetic intensity → memory-bound → memory subsystem power dominates

---

## Per-Type Energy Analysis

### Matmul (matrix multiplication)

**Profile:** Compute-bound at large sizes, memory-bound at small sizes.

| Property | Value |
|----------|-------|
| Arithmetic intensity | O(N) — grows with matrix dimension |
| SM utilization | Very high (>90%) for large matrices |
| Memory utilization | Moderate — data reuse via tiling |
| Energy scaling | Superlinear with matrix size due to O(N³) FLOPs |

**Why it uses the most energy:**
- Highest total FLOP count of any common operation
- Tensor cores can reduce time but power draw is still high
- SM activity factor is near maximum → P_dynamic ≈ C×V²×f (A≈1)

**Optimization levers:**
- Reduced precision (fp16, int8) cuts both FLOPs and bit-flip energy
- Tiling reduces off-chip memory access (shared memory is ~100× cheaper)
- Sparsity skipping can reduce effective FLOP count

---

### Convolution (conv2d, depthwise, grouped)

**Profile:** Compute-bound for standard convolutions, increasingly
memory-bound for depthwise/grouped variants.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Varies: standard conv high, depthwise conv low |
| SM utilization | Moderate to high |
| Memory utilization | High for large feature maps |
| Energy scaling | Depends on kernel size, channels, spatial dims |

**Key insight:** Depthwise separable convolutions are much cheaper per-FLOP but
have low arithmetic intensity, making them memory-bound.  Their
energy-per-useful-compute can actually be *worse* because they pay full memory
access costs for relatively few FLOPs.

---

### Attention (softmax-matmul patterns)

**Profile:** Mixed compute and memory bound.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Moderate — two matmuls but also softmax |
| SM utilization | High during Q×K and Attn×V, drops during softmax |
| Memory utilization | Spikes during softmax (reads/writes full attention matrix) |
| Energy scaling | O(N²) with sequence length |

**Key insight:** The softmax step materializes an N×N attention matrix to global
memory, creating a burst of memory traffic between two compute phases.  Flash
Attention avoids this materialization by fusing the operations — reducing memory
energy while keeping compute energy roughly constant.

---

### Layer Normalization / RMS Norm

**Profile:** Strongly memory-bound.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Very low (~2-4 FLOPs per element) |
| SM utilization | Low — limited parallelism |
| Memory utilization | High — reads and writes the full tensor |
| Energy scaling | Linear with tensor size |

**Key insight:** Normalization performs trivial math (mean, variance, scale)
but touches every element twice (read + write).  Almost all energy goes to data
movement.  The SM power is dominated by leakage rather than switching.

**Optimization levers:**
- Fuse with preceding/following operation to avoid extra memory round-trip
- This is why fused attention + layer norm kernels are popular

---

### Activation Functions (ReLU, GELU, SiLU, etc.)

**Profile:** Strongly memory-bound.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Very low (~1-3 FLOPs per element) |
| SM utilization | Low |
| Memory utilization | High — elementwise read + write |
| Energy scaling | Linear with tensor size |

**Key insight:** Activation functions are essentially memory copy with a tiny
per-element transform.  Energy is dominated by DRAM access.  The choice of
activation function (ReLU vs GELU vs SiLU) has negligible impact on energy —
what matters is whether it can be fused with the preceding linear layer.

**Interesting property:** ReLU creates sparsity (~50% zeros), which can reduce
energy in *subsequent* operations by enabling sparse compute paths and reducing
bit flip rates in data buses.

---

### Reduction Operations (sum, mean, max, argmax)

**Profile:** Memory-bound with serial dependency.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Very low (~1 FLOP per element) |
| SM utilization | Starts high, drops as reduction tree narrows |
| Memory utilization | High at start, drops quickly |
| Energy scaling | Linear with input size |

**Key insight:** Reductions have a fundamental serial dependency (each level
depends on the previous), meaning parallelism drops exponentially.  In later
stages, most SMs are idle but still consuming leakage power.  This makes
reductions disproportionately energy-inefficient relative to their
computational importance.

---

### Pooling (avg_pool, max_pool, adaptive_pool)

**Profile:** Memory-bound, similar to strided reduction.

| Property | Value |
|----------|-------|
| Arithmetic intensity | Very low |
| SM utilization | Low to moderate |
| Memory utilization | Moderate |
| Energy scaling | Sub-linear (output is smaller than input) |

**Key insight:** Pooling is essentially reduction with spatial locality.
It benefits from cache locality more than global reductions do, making it
slightly more energy-efficient per byte processed.

---

### Composite / Fused Operations

**Profile:** Varies depending on what is fused.

**Key insight:** The biggest energy win in kernel optimization is *fusion* —
combining multiple memory-bound operations into a single kernel that keeps data
in registers or shared memory.  A fused norm+activation saves one full
round-trip to DRAM (~500-1000 pJ per element), which can reduce total energy
by 30-50% for those operations.

---

## Summary Table

| Kernel type | Bound by | Main energy sink | Optimization lever |
|------------|----------|-----------------|-------------------|
| Matmul | Compute | SM (ALU switching) | Precision, tiling, sparsity |
| Conv | Compute/Mixed | SM + memory | Im2col vs direct, winograd |
| Attention | Mixed | SM + memory (softmax) | Flash Attention (fusion) |
| Norm | Memory | DRAM access | Fusion with neighbors |
| Activation | Memory | DRAM access | Fusion with linear layer |
| Reduction | Memory | DRAM + idle SM leakage | Parallel-friendly reductions |
| Pooling | Memory | DRAM access | Strided access patterns |
| Composite | Varies | Reduced memory traffic | Fusion is the optimization |

---

## Connection to Transformer Inference

In a transformer forward pass, the energy budget is dominated by matrix
multiplications (~60%) and attention (~15%).  However, the *per-operation
energy efficiency gap* is largest for memory-bound operations (norm, activation)
because they waste the most energy on data movement that could be avoided
through fusion.

This means:
- **Absolute energy savings** are largest in matmul (optimize precision, tiling)
- **Relative energy savings** (% improvement possible) are largest in norm and
  activation (fusion can cut energy by 30-50%)
- **Systemic energy savings** come from reducing kernel launch overhead and
  keeping data on-chip across operation boundaries
