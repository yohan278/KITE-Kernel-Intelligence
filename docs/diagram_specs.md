# KITE Diagram Specifications for Figma

This document provides three production-ready diagram specifications. Each section contains the exact layout, labels, arrows, colors, and annotations needed to reproduce the diagram in Figma.

---

## Diagram 1: GRPO Training Pipeline (M1 / M2 / M3)

**Purpose**: Show the end-to-end RL training loop that produces the three GRPO model stages.

### Layout

Horizontal left-to-right flow with a central loop. Five major blocks connected by directed arrows.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                     │
│   ┌───────────────┐       ┌──────────────────────┐       ┌──────────────────────────┐              │
│   │               │       │                      │       │                          │              │
│   │  KernelBench  │──────►│   Prompt Builder     │──────►│    Qwen2.5-Coder-7B     │              │
│   │  Task Pool    │       │                      │       │    + LoRA (r=64)         │              │
│   │               │       │  "Optimize this      │       │                          │              │
│   │  80 tasks     │       │   PyTorch model      │       │  Generates k=8           │              │
│   │  L1─L4        │       │   with a custom      │       │  candidate kernels       │              │
│   │               │       │   GPU kernel."        │       │  per prompt              │              │
│   └───────────────┘       │  + reference code    │       │                          │              │
│                           └──────────────────────┘       └────────────┬─────────────┘              │
│                                                                       │                            │
│                                                                       │  8 CUDA kernel             │
│                                                                       │  code completions          │
│                                                                       ▼                            │
│                           ┌──────────────────────────────────────────────────────────────┐          │
│                           │                    GPU Evaluation                             │          │
│                           │                                                              │          │
│                           │  ┌─────────────────┐    ┌─────────────────────────────────┐  │          │
│                           │  │   Pre-check     │    │   KernelBench Eval              │  │          │
│                           │  │                 │    │                                 │  │          │
│                           │  │  • AST parse    │───►│  • Compile candidate            │  │          │
│                           │  │  • ModelNew     │    │  • 3 correctness trials         │  │          │
│                           │  │    class check  │    │  • 25 performance trials        │  │          │
│                           │  │  • forward()    │    │  • CUDA event timing            │  │          │
│                           │  │    arity match  │    │    → runtime_ms, speedup        │  │          │
│                           │  └─────────────────┘    │                                 │  │          │
│                           │                         │  ┌───────────────────────────┐  │  │          │
│                           │                         │  │  NVML Power Sampler       │  │  │          │
│                           │                         │  │  (concurrent thread)      │  │  │          │
│                           │                         │  │  polls every 50ms         │  │  │          │
│                           │                         │  │  → avg_power_w, joules    │  │  │          │
│                           │                         │  └───────────────────────────┘  │  │          │
│                           │                         └─────────────────────────────────┘  │          │
│                           └──────────────────────────────────┬───────────────────────────┘          │
│                                                              │                                     │
│                                                              │  Per-candidate metrics:             │
│                                                              │  compile_ok, correct,               │
│                                                              │  runtime_ms, speedup,               │
│                                                              │  joules, avg_power_w                │
│                                                              ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐      │
│   │                          Reward Computation                                              │      │
│   │                                                                                          │      │
│   │    ┌──────────────────────────────────────────────────────────────────────────────┐      │      │
│   │    │                                                                              │      │      │
│   │    │   R = α · log(speedup) − η · log(runtime) − β · log(joules)                │      │      │
│   │    │       − δ · avg_power   − γ · max(0, latency − SLA)                        │      │      │
│   │    │                                                                              │      │      │
│   │    │   + λ · R_IPW    ← only for M3                                              │      │      │
│   │    │                                                                              │      │      │
│   │    └──────────────────────────────────────────────────────────────────────────────┘      │      │
│   │                                                                                          │      │
│   │    Failure penalties:  compile_fail → −1.0    incorrect → −0.5                          │      │
│   │                                                                                          │      │
│   │    ┌──────────────────────────────────────────────────────────────────────────┐          │      │
│   │    │  Model variant weights:                                                  │          │      │
│   │    │                                                                          │          │      │
│   │    │   M1 Throughput     β = 0      δ = 0      λ = 0    (speed only)         │          │      │
│   │    │   M2 Energy         β = 0.5    δ = 0.01   λ = 0    (+ energy)           │          │      │
│   │    │   M3 IPW Blend      β = 0.5    δ = 0.01   λ ∈ {0.1, 0.25, 0.5}        │          │      │
│   │    │                                                                          │          │      │
│   │    └──────────────────────────────────────────────────────────────────────────┘          │      │
│   └──────────────────────────────────────────────────────┬───────────────────────────────────┘      │
│                                                          │                                         │
│                                                          │  8 scalar rewards                       │
│                                                          │  (one per completion)                   │
│                                                          ▼                                         │
│   ┌───────────────────────────────────────────────────────────────────────┐                        │
│   │                       GRPO Policy Update                              │                        │
│   │                                                                       │                        │
│   │   1. Rank 8 completions by reward within the group                   │                        │
│   │   2. Compute group-relative advantages (zero-mean normalization)      │                        │
│   │   3. Policy gradient with KL penalty (β_kl = 0.04)                   │                        │
│   │   4. Update LoRA weights (q, k, v, o projections)                    │                        │
│   │                                                                       │           ┌──────────┐│
│   │   Repeats across 80 tasks × 3 epochs × 3 seeds                      │──────────►│Checkpoint││
│   │                                                                       │           │LoRA + JSON││
│   └───────────────────────────────┬───────────────────────────────────────┘           └──────────┘│
│                                   │                                                               │
│                                   │  Updated policy weights                                       │
│                                   └──────────────────────► (loops back to Qwen generation)        │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Figma Element Guide

| Element | Shape | Color | Notes |
|---------|-------|-------|-------|
| KernelBench Task Pool | Rounded rect | Light purple `#E8E0F0` | Icon: stack of documents |
| Prompt Builder | Rounded rect | Light gray `#F0F0F0` | Show snippet of prompt text |
| Qwen2.5-Coder-7B + LoRA | Rounded rect | Blue `#4A9EDA` | Largest block, model icon |
| GPU Evaluation | Rounded rect with dashed border | Light orange `#FFF3E0` | Contains two sub-blocks |
| Pre-check | Small rect inside GPU Eval | White | AST / structural checks |
| KernelBench Eval | Rect inside GPU Eval | Orange `#F5A623` | Compile + correctness + perf |
| NVML Power Sampler | Small rect inside GPU Eval | Green `#2BC490` | Show "50ms polling" badge |
| Reward Computation | Rounded rect | Light red `#FFEBEE` | Contains formula + variant table |
| GRPO Policy Update | Rounded rect | Dark navy `#1A3A6E`, white text | The RL update step |
| Checkpoint | Small rounded rect | Gray | Output artifact |
| Arrows | Solid, directional | Dark gray `#444` | Label with data type flowing |
| Loop-back arrow | Dashed, curved | Blue `#4A9EDA` | From GRPO update back to generation |
| M1/M2/M3 variant table | Inset table | White background, colored rows | M1=blue row, M2=green row, M3=red row |

### Key Labels on Arrows

- Task Pool → Prompt Builder: **"KernelTask (prompt + reference code)"**
- Prompt Builder → Qwen: **"Formatted prompt"**
- Qwen → GPU Eval: **"k=8 candidate kernels"**
- GPU Eval → Reward: **"compile_ok, correct, runtime_ms, speedup, joules, avg_power_w"**
- Reward → GRPO Update: **"8 scalar rewards"**
- GRPO Update → Qwen (loop-back): **"Updated LoRA weights"**
- GRPO Update → Checkpoint: **"LoRA + checkpoint.json"**

---

## Diagram 2: Telemetry Capture Pipeline

**Purpose**: Show how GPU hardware telemetry (power, energy, temperature) is captured concurrently during kernel evaluation and flows into the reward signal.

### Layout

Vertical flow with a central GPU execution block and parallel measurement streams.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                        Candidate Kernel Code                                    │
│                              │                                                  │
│                              ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │                         H100 GPU                                    │       │
│   │                                                                     │       │
│   │   ┌───────────────────────────────────┐                            │       │
│   │   │        Kernel Execution           │                            │       │
│   │   │                                   │                            │       │
│   │   │   Warmup  →  Correctness Trials   │                            │       │
│   │   │              (3 runs)             │                            │       │
│   │   │          →  Performance Trials    │                            │       │
│   │   │              (25 runs)            │                            │       │
│   │   │                                   │                            │       │
│   │   └─────────────────┬─────────────────┘                            │       │
│   │                     │                                              │       │
│   │   ┌─────────────────┴─────────────────┐                            │       │
│   │   │         │                         │                            │       │
│   │   ▼         ▼                         ▼                            │       │
│   │                                                                     │       │
│   │   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │       │
│   │   │ CUDA Events │  │ NVML Power   │  │ H100 Rust gRPC Sampler │   │       │
│   │   │             │  │ Sampler      │  │ (Tier 1, optional)     │   │       │
│   │   │ start()     │  │              │  │                        │   │       │
│   │   │   ↓         │  │ Daemon thread│  │ localhost:50052        │   │       │
│   │   │ kernel()    │  │ polls every  │  │ Derives power from     │   │       │
│   │   │   ↓         │  │ 50ms via     │  │ NVML energy counter    │   │       │
│   │   │ stop()      │  │ pynvml       │  │ deltas (hardware-      │   │       │
│   │   │   ↓         │  │              │  │ accurate)              │   │       │
│   │   │ elapsed μs  │  │ Records:     │  │                        │   │       │
│   │   │             │  │ (t_s, W)     │  │ Streams per sample:    │   │       │
│   │   │ → runtime   │  │ tuples       │  │ • power_w              │   │       │
│   │   │   _ms       │  │              │  │ • energy_j             │   │       │
│   │   │             │  │              │  │ • gpu_utilization      │   │       │
│   │   │             │  │              │  │ • temperature_c        │   │       │
│   │   │             │  │              │  │ • clock_freqs          │   │       │
│   │   │             │  │              │  │ • memory_used          │   │       │
│   │   │             │  │              │  │ • pcie_throughput      │   │       │
│   │   │             │  │              │  │ • throttle_reasons     │   │       │
│   │   └──────┬──────┘  └──────┬───────┘  └────────────┬───────────┘   │       │
│   │          │                │                       │               │       │
│   └──────────┼────────────────┼───────────────────────┼───────────────┘       │
│              │                │                       │                        │
│              ▼                ▼                       ▼                        │
│   ┌─────────────┐  ┌──────────────────┐  ┌──────────────────────────┐         │
│   │  Kernel     │  │  Trapezoidal     │  │                          │         │
│   │  Timing     │  │  Energy          │  │    EnergyTrace           │         │
│   │             │  │  Integration     │  │                          │         │
│   │  runtime_ms │  │                  │  │  timestamps[]            │         │
│   │  speedup    │  │  E = Σ ½(Pᵢ+Pᵢ₊₁)│  │  power_w[]              │         │
│   │             │  │      × Δtᵢ       │  │  energy_j[]              │         │
│   │             │  │                  │  │  gpu_util[]              │         │
│   │             │  │  → avg_power_w   │  │  temp_c[]               │         │
│   │             │  │  → total joules  │  │  phase_segments[]       │         │
│   └──────┬──────┘  └────────┬─────────┘  └────────────┬─────────────┘         │
│          │                  │                          │                       │
│          │                  │                          ▼                       │
│          │                  │            ┌──────────────────────────┐          │
│          │                  │            │   Phase Attribution      │          │
│          │                  │            │                          │          │
│          │                  │            │  Split at TTFT boundary: │          │
│          │                  │            │  ┌────────┬───────────┐  │          │
│          │                  │            │  │Prefill │  Decode   │  │          │
│          │                  │            │  │energy_j│  energy_j │  │          │
│          │                  │            │  └────────┴───────────┘  │          │
│          │                  │            └────────────┬─────────────┘          │
│          │                  │                         │                        │
│          │                  │                         ▼                        │
│          │                  │            ┌──────────────────────────┐          │
│          │                  │            │    IPW Adapter           │          │
│          │                  │            │                          │          │
│          │                  │            │  energy_per_output_token │          │
│          │                  │            │  prefill_energy/token    │          │
│          │                  │            │  decode_energy/token     │          │
│          │                  │            │  throughput_tps          │          │
│          │                  │            │  IPW, IPJ scores        │          │
│          │                  │            └────────────┬─────────────┘          │
│          │                  │                         │                        │
│          └──────────────────┴─────────────────────────┘                        │
│                             │                                                  │
│                             ▼                                                  │
│          ┌──────────────────────────────────────────────────────────┐          │
│          │                   Reward Function                        │          │
│          │                                                          │          │
│          │   Inputs consumed:                                       │          │
│          │   • runtime_ms, speedup        ← from CUDA events       │          │
│          │   • joules, avg_power_w        ← from NVML integration  │          │
│          │   • energy_per_token, IPW      ← from IPW adapter       │          │
│          │   • compile_ok, correct        ← from KernelBench eval  │          │
│          │                                                          │          │
│          │   → scalar reward for GRPO update                       │          │
│          └──────────────────────────────────────────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Figma Element Guide

| Element | Shape | Color | Notes |
|---------|-------|-------|-------|
| H100 GPU block | Large rounded rect with chip icon | Dark gray `#2D2D2D`, white text | Contains all execution + measurement |
| Kernel Execution | Rect inside GPU | Orange `#F5A623` | Warmup → Correctness → Performance |
| CUDA Events | Rect | Blue `#4A9EDA` | "Kernel timing" label |
| NVML Power Sampler | Rect | Green `#2BC490` | "50ms polling thread" label |
| H100 Rust gRPC | Rect | Purple `#7C4DFF` | Optional/advanced path |
| Trapezoidal Integration | Rect | Light green `#E8F5E9` | Show integral formula |
| EnergyTrace | Rect | Light purple `#F3E5F5` | Show field list |
| Phase Attribution | Rect | Light orange `#FFF3E0` | Prefill/Decode split visualization |
| IPW Adapter | Rect | Light blue `#E3F2FD` | Per-token metrics |
| Reward Function | Rect | Red `#FFCDD2` | Final output node |
| Concurrent arrows | Dashed parallel lines | Gray | Show CUDA events and NVML run simultaneously |

### Key Labels on Arrows

- Code → GPU: **"Compiled CUDA kernel"**
- CUDA Events → Timing: **"runtime_ms, speedup"**
- NVML Sampler → Integration: **"[(t₀, P₀), (t₁, P₁), ...]"**
- Integration → Reward: **"avg_power_w, total_joules"**
- EnergyTrace → Phase Attribution: **"Full time-series trace"**
- Phase Attribution → IPW Adapter: **"Prefill/Decode segments"**
- All metrics → Reward: **"Scalar reward → GRPO"**

---

## Diagram 3: Research Project Overview (M0 → M5 Progression)

**Purpose**: Show how the full KITE research project is structured as a progressive training pipeline — from supervised fine-tuning through increasingly sophisticated RL stages — along with the evaluation framework that validates each stage.

### Layout

Horizontal pipeline with a top training track and a bottom evaluation track. Each model stage is a distinct node.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│   TRAINING TRACK                                                                                     │
│                                                                                                      │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                       │
│   │          │     │              │     │              │     │              │                       │
│   │   M0     │────►│     M1       │────►│     M2       │────►│     M3       │                       │
│   │   SFT    │     │  GRPO        │     │  GRPO        │     │  GRPO        │                       │
│   │          │     │  Throughput   │     │  Energy      │     │  IPW Blend   │                       │
│   │          │     │              │     │              │     │              │                       │
│   └──────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘                       │
│        │                                                            │                               │
│        │   Supervised learning                                      │   Kernel policy                │
│        │   on KernelBench tasks                                     │   feeds into                   │
│        │                                                            ▼                               │
│        │                                                    ┌──────────────┐     ┌──────────────┐   │
│        │                                                    │              │     │              │   │
│        │                                                    │     M4       │────►│     M5       │   │
│        │                                                    │  Runtime     │     │  HRL         │   │
│        │                                                    │  PPO         │     │  Hierarchical│   │
│        │                                                    │              │     │              │   │
│        │                                                    └──────────────┘     └──────────────┘   │
│        │                                                                                            │
│   ─────┼────────────────────────────────────────────────────────────────────────────────────────     │
│        │                                                                                            │
│   WHAT EACH STAGE ADDS                                                                              │
│                                                                                                      │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                       │
│   │ Base LLM │     │ Reward:      │     │ Reward adds: │     │ Reward adds: │                       │
│   │ learns   │     │              │     │              │     │              │                       │
│   │ kernel   │     │ α·log(spd)   │     │ −β·log(J)    │     │ +λ·R_IPW     │                       │
│   │ code     │     │ −η·log(rt)   │     │ −δ·power     │     │ intelligence │                       │
│   │ structure│     │ −γ·SLA       │     │              │     │ per watt     │                       │
│   │          │     │              │     │ + telemetry  │     │ blending     │                       │
│   └──────────┘     └──────────────┘     └──────────────┘     └──────────────┘                       │
│                                                                                                      │
│   ┌──────────────────────────────┐     ┌──────────────────────────────┐                              │
│   │ M4 adds:                     │     │ M5 adds:                     │                              │
│   │                              │     │                              │                              │
│   │ Runtime action space:        │     │ Alternating training:        │                              │
│   │ • Power cap selection        │     │ 1) Kernel GRPO round         │                              │
│   │ • DVFS profile               │     │ 2) Runtime PPO round         │                              │
│   │ • Concurrency level          │     │ 3) Joint fine-tune           │                              │
│   │                              │     │                              │                              │
│   │ State: queue depth, batch    │     │ HierarchyController selects  │                              │
│   │ size, latency percentiles    │     │ kernel + runtime jointly     │                              │
│   └──────────────────────────────┘     └──────────────────────────────┘                              │
│                                                                                                      │
│   ─────────────────────────────────────────────────────────────────────────────────────────────      │
│                                                                                                      │
│   EVALUATION TRACK                                                                                   │
│                                                                                                      │
│   All models evaluated on:                                                                           │
│                                                                                                      │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐       │
│   │                                                                                          │       │
│   │   80 KernelBench tasks  ×  3 seeds (11, 22, 33)  ×  4 difficulty levels (L1─L4)        │       │
│   │                                                                                          │       │
│   │   Metrics:  compile_rate │ correctness │ pass@k │ runtime_ms │ joules │ speedup │ reward│       │
│   │                                                                                          │       │
│   │   Analyses: Pareto frontier │ Pairwise significance │ Matched-runtime energy │ Ablations│       │
│   │             Failure taxonomy │ Cross-hardware transfer │ Generalization (held-out)       │       │
│   │                                                                                          │       │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Figma Element Guide

| Element | Shape | Color | Notes |
|---------|-------|-------|-------|
| M0 SFT | Rounded rect | Light gray `#E0E0E0` | Baseline, no RL |
| M1 Throughput GRPO | Rounded rect | Blue `#4A9EDA` | Speed-only reward |
| M2 Energy GRPO | Rounded rect | Green `#2BC490` | Adds energy penalty |
| M3 IPW Blend | Rounded rect | Orange `#F5A623` | Adds IPW blending |
| M4 Runtime PPO | Rounded rect | Red `#EF5350` | Different RL algorithm (PPO) |
| M5 HRL | Rounded rect | Purple `#7C4DFF` | Combines kernel + runtime |
| Stage description boxes | Rect below each model | Matching color, lighter shade | What the stage adds |
| Evaluation track | Wide rounded rect | Light yellow `#FFFDE7` | Shared across all models |
| Arrows M0→M1→M2→M3 | Solid, thick | Dark gray | "LoRA weights carry forward" |
| Arrow M3→M4→M5 | Solid, thick | Dark gray | "Kernel policy feeds runtime" |
| Divider line | Horizontal dashed | Light gray | Separates training from eval |

### Key Labels on Arrows

- M0 → M1: **"SFT LoRA weights (initialization)"**
- M1 → M2: **"Same architecture, adds −β·log(J) to reward"**
- M2 → M3: **"Adds λ·R_IPW blend term"**
- M3 → M4: **"Frozen kernel policy, train runtime controller"**
- M4 → M5: **"Alternating kernel + runtime joint training"**

### Key Numbers to Include

| Model | Correctness | Joules | Reward | Key Win |
|-------|-------------|--------|--------|---------|
| M0 | 47.9% | 6.70 | −0.02 | Baseline |
| M1 | 66.7% | 5.82 | 8.00 | +19pp correctness |
| M2 | 65.4% | 4.45 | 9.17 | −24% energy |
| M3 | 67.5% | 3.76 | 10.27 | −35% energy vs M1 |
| M4 | 56.7% | 4.44 | 8.32 | Adaptive runtime |
| M5 | 56.3% | 3.77 | 9.97 | Fastest (15.8ms) |

---

## Color Palette Summary

| Use | Hex | Name |
|-----|-----|------|
| M0 / neutral | `#E0E0E0` | Light gray |
| M1 / throughput / primary | `#4A9EDA` | Blue |
| M2 / energy / secondary | `#2BC490` | Green |
| M3 / IPW blend | `#F5A623` | Orange |
| M4 / runtime PPO | `#EF5350` | Red |
| M5 / HRL | `#7C4DFF` | Purple |
| GPU hardware | `#2D2D2D` | Charcoal |
| Reward function | `#FFCDD2` | Light red |
| Evaluation | `#FFFDE7` | Light yellow |
| Background | `#FFFFFF` | White |
| Arrows / text | `#424242` | Dark gray |

---

## Typography Recommendations

- **Titles**: Bold, 24pt, dark gray
- **Block labels**: Semi-bold, 14pt
- **Formula text**: Monospace or math font, 12pt
- **Arrow labels**: Regular, 10pt, italic
- **Numbers/metrics**: Monospace, 11pt

---

## Suggested Figma Frame Sizes

| Diagram | Width | Height |
|---------|-------|--------|
| 1. GRPO Training Pipeline | 1400px | 900px |
| 2. Telemetry Capture | 1000px | 1200px |
| 3. Project Overview | 1600px | 800px |
