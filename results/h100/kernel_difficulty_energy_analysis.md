# KernelBench: Difficulty & Energy Impact Analysis

> Analysis of all 271 tasks across Levels 1–4, ranked by **optimization difficulty** and estimated **energy consumption (joules)** when kernels run in real ML workloads.

---

## Rating Scale

| Symbol | Difficulty (to write an optimal kernel) | Energy per Invocation |
|--------|----------------------------------------|----------------------|
| ★☆☆☆☆ | Trivial — element-wise or well-served by cuBLAS | Negligible |
| ★★☆☆☆ | Moderate — standard patterns, known tiling strategies | Low |
| ★★★☆☆ | Hard — irregular memory, fusion trade-offs, numerical pitfalls | Medium |
| ★★★★☆ | Very Hard — complex data dependencies, multi-kernel orchestration | High |
| ★★★★★ | Extreme — full-model graph optimization, cross-layer fusion needed | Very High |

---

## Tier 1 — Highest Energy & Hardest to Optimize

These are the problems where a well-optimized kernel saves the most joules in production and where writing that kernel is genuinely difficult.

### Level 4: Full-Model Inference (the energy heavyweights)

Every L4 task wraps a real HuggingFace model end-to-end. A single forward pass touches dozens of fused GEMM + attention + normalization kernels. Optimization here means graph-level kernel fusion, KV-cache management, and quantization-aware scheduling.

| Task | Model | Batch × Seq | Difficulty | Energy/Call | Why It's Hard |
|------|-------|-------------|------------|-------------|---------------|
| **L4_1** | GPT-Neo 2.7B | 32 × 256 | ★★★★★ | **~18–25 J** | 2.7B params, 32 layers of causal attention at batch 32. Enormous GEMM pressure. Peak memory bandwidth saturation. |
| **L4_3** | GPT-Neo 2.7B | 1 × 2047 | ★★★★★ | **~12–18 J** | Same model, long-sequence regime. KV-cache reuse is critical; naive implementation thrashes HBM. |
| **L4_18** | GPT-Neo 2.7B | 512 × 32 | ★★★★★ | **~20–28 J** | High-batch short-seq: GEMM-dominated. Enormous activation memory; fusing attention + MLP across 32 layers is the bottleneck. |
| **L4_2** | OPT-1.3B | 1 × 2047 | ★★★★★ | **~6–10 J** | 1.3B params with long context. Memory-bound regime; flash-attention variants critical. |
| **L4_4** | OPT-1.3B | 32 × 256 | ★★★★★ | **~8–14 J** | Compute-bound GEMM regime at scale. |
| **L4_8** | OPT-1.3B | 512 × 32 | ★★★★★ | **~10–16 J** | Largest batch config for OPT. Batch-GEMM scheduling dominates. |
| **L4_17** | BART-Large | 1024 × 32 | ★★★★☆ | **~8–12 J** | Encoder-decoder model with cross-attention. Two separate attention patterns to fuse. |
| **L4_6** | BART-Large | 1 × 1023 | ★★★★☆ | **~4–7 J** | Long-sequence encoder-decoder; cross-attention memory patterns are irregular. |
| **L4_5** | BigBird-RoBERTa | 1 × 4095 | ★★★★★ | **~5–9 J** | Sparse attention (block + random + global). Implementing the sparse pattern as a CUDA kernel is a research problem in itself. |
| **L4_13** | Reformer (enwik8) | 32 × 256 | ★★★★★ | **~3–6 J** | LSH attention buckets. Custom hashing + sorted-attention kernels required. |

### Level 3: Full Architecture Blocks

| Task | Architecture | Difficulty | Energy/Call | Why It's Hard / High Energy |
|------|-------------|------------|-------------|----------------------------|
| **L3_30** | SwinTransformerV2 | ★★★★★ | **~4–8 J** | 23K-char prompt. Window-based shifted attention, relative position bias, multi-scale feature maps. The most complex single architecture in the benchmark. |
| **L3_29** | SwinMLP | ★★★★★ | **~3–6 J** | 16K-char prompt. Cyclic shifts + window partitioning + MLP spatial mixing. Irregular memory patterns. |
| **L3_7** | GoogLeNet/InceptionV1 | ★★★★☆ | **~1.5–3 J** | Multi-branch parallel convolutions at different kernel sizes, concatenated. Branch divergence kills naive fusion. |
| **L3_10** | ResNet-101 | ★★★★☆ | **~2–4 J** | 101 layers, bottleneck blocks, skip connections. Deep pipeline; cross-block fusion is essential. |
| **L3_15** | DenseNet-121 | ★★★★☆ | **~2–4 J** | Dense connectivity means each layer concatenates ALL prior feature maps. Memory management is nightmarish. |
| **L3_16** | DenseNet-201 | ★★★★☆ | **~3–5 J** | Even deeper dense connectivity. Activation recomputation vs. memory trade-off. |
| **L3_22** | EfficientNet-B0 | ★★★★☆ | **~1–2 J** | MBConv blocks: depthwise-separable conv + squeeze-excite + swish. Many small kernels that need fusion. |
| **L3_26** | ShuffleNet | ★★★★☆ | **~0.8–1.5 J** | Channel shuffle operation is a non-trivial permutation; grouped convolutions need careful scheduling. |
| **L3_48** | Mamba2 (return Y) | ★★★★★ | **~1–3 J** | Structured state-space model with selective scan. The parallel scan primitive has no cuBLAS equivalent — must be written from scratch. |
| **L3_49** | Mamba2 (return state) | ★★★★★ | **~1–3 J** | Same scan complexity, different output path. |
| **L3_43** | MinGPT CausalAttention | ★★★★☆ | **~0.5–2 J** | Causal mask + scaled dot-product + multi-head. Flash-attention territory. batch=128 makes it compute-heavy. |
| **L3_44** | MiniGPT Block | ★★★★☆ | **~0.8–3 J** | Full transformer block: attention + MLP + layer norms + residuals. Fusing the whole block is the prize. |
| **L3_28** | Vision Transformer | ★★★★☆ | **~0.5–2 J** | Patch embedding + 6-layer transformer. Attention kernel dominates energy at depth=6. |
| **L3_35–L3_38** | LSTM variants | ★★★★☆ | **~0.3–1 J** | Sequential dependency across time steps. 6 layers, bidirectional variants add 2× compute. Fundamentally hard to parallelize. |
| **L3_45** | U-Net (Softmax) | ★★★★☆ | **~0.5–2 J** | Encoder-decoder with skip connections at multiple resolutions. Memory layout changes across scales. |

---

## Tier 2 — High Energy, Moderate-to-Hard Optimization

### Level 2: Fused Operator Chains

The energy here comes from large tensor sizes + compute-heavy base ops (GEMM, Conv3d). The difficulty is in fusing the chain into one kernel launch.

| Task | Chain | Difficulty | Energy/Call | Notes |
|------|-------|------------|-------------|-------|
| **L2_9** | Matmul → Sub → Mul → ReLU | ★★★☆☆ | **~0.5–1 J** | 1024 × 8192 × 8192 GEMM. Enormous. Fusing the epilogue (sub/mul/relu) into the GEMM is the key win. |
| **L2_12** | GEMM → Mul → LeakyReLU | ★★★☆☆ | **~0.5–1 J** | Same giant GEMM dimensions. |
| **L2_18** | Matmul → Sum → Max → AvgPool → LogSumExp² | ★★★★☆ | **~0.5–1 J** | 8192×8192 GEMM plus a 5-op reduction chain. Numerical stability in back-to-back LogSumExp is tricky. |
| **L2_22** | Matmul → Scale → ResAdd → Clamp → LogSumExp → Mish | ★★★★☆ | **~0.3–0.8 J** | 6-deep fusion chain with LogSumExp (needs max-subtraction for stability). |
| **L2_13** | ConvT3d → Mean → Add → Softmax → Tanh → Scale | ★★★★☆ | **~0.3–0.8 J** | 3D transposed conv at 16×16→128ch plus a 5-op tail. 3D memory layouts are painful. |
| **L2_89** | ConvT3d → MaxPool → Softmax → Sub → Swish → Max | ★★★★☆ | **~0.2–0.6 J** | Longest chain in L2 with 3D data. Multiple reductions interleaved with element-wise ops. |
| **L2_38** | ConvT3d → AvgPool → Clamp → Softmax → Mul | ★★★☆☆ | **~0.2–0.5 J** | 3D conv + pooling + softmax: requires careful tiling to keep intermediates in SRAM. |
| **L2_92** | Conv2d → GroupNorm → Tanh → HardSwish → ResAdd → LogSumExp | ★★★★☆ | **~0.2–0.5 J** | GroupNorm mid-chain forces a sync point (needs group stats). Hard to fuse around it. |
| **L2_11** | ConvT2d → BN → Tanh → MaxPool → GroupNorm | ★★★★☆ | **~0.2–0.5 J** | batch=512, 64→128ch with kernel=5. Two normalization layers force two sync barriers. |

### Level 1: Large Primitive Operations

| Task | Operation | Difficulty | Energy/Call | Notes |
|------|-----------|------------|-------------|-------|
| **L1_2** | Matrix multiply 2048×8192 × 8192×4096 | ★★☆☆☆ | **~0.3–0.6 J** | ~137 TFLOP. cuBLAS is near-optimal, but a hand-tuned Triton kernel with custom tiling can still win 5–10%. |
| **L1_3** | Batched matmul 128×512×1024×2048 | ★★★☆☆ | **~0.4–0.8 J** | 128 independent GEMMs. Batch scheduling across SMs is the puzzle. |
| **L1_6** | Matmul 256×524288×256 | ★★★☆☆ | **~0.2–0.4 J** | Extremely large K. Accumulator precision and split-K strategies are critical. |
| **L1_1** | Square matmul 4096×4096 | ★★☆☆☆ | **~0.1–0.3 J** | Classic GEMM. Still ~137 GFLOP per call. |
| **L1_4** | Matvec 2048×1048576 | ★★★☆☆ | **~0.1–0.3 J** | 1M-wide vector. Purely memory-bandwidth bound; coalescing and vectorized loads are everything. |
| **L1_63** | Conv2d 16×16→128ch, 1024×1024 | ★★★☆☆ | **~0.2–0.5 J** | Massive spatial dimensions. im2col memory explosion; implicit GEMM or Winograd required. |
| **L1_61** | ConvT3d 48→48ch, 64³ | ★★★☆☆ | **~0.3–0.6 J** | 3D transposed conv on 64³ volume. Memory layout (NCDHW vs NDHWC) choice is make-or-break. |
| **L1_70** | ConvT3d 48→24ch, 96³ | ★★★☆☆ | **~0.3–0.7 J** | Even larger 3D volume. Exceeds L2 cache easily. |
| **L1_93** | Masked cumulative sum | ★★★★☆ | **~0.05–0.1 J** | Irregular parallel prefix sum with a boolean mask. Work-efficient scan with masking is genuinely hard. |
| **L1_90** | Cumulative product | ★★★★☆ | **~0.03–0.08 J** | Sequential dependency + numerical instability (products explode/vanish). Log-domain trick needed. |
| **L1_89** | Cumulative sum | ★★★☆☆ | **~0.03–0.08 J** | Parallel prefix sum. Blelloch scan on GPU — well-studied but tricky to get right. |
| **L1_36** | RMSNorm | ★★★☆☆ | **~0.02–0.05 J** | Two-pass reduction (compute RMS, then normalize). Fusing into one pass with online algorithms is the win. |
| **L1_97** | Scaled Dot-Product Attention | ★★★★☆ | **~0.05–0.15 J** | The kernel that launched FlashAttention. Tiling to avoid materializing the N×N attention matrix. |

---

## Tier 3 — Moderate Energy, Lower Difficulty

These are common building blocks. They matter at scale because they're called millions of times during training.

### Activations & Element-wise (L1_19–L1_32)

All operate on tensors of shape 4096 × 393216 (~1.6 billion elements). Individually cheap (~0.01–0.03 J) but called on every layer of every model. At ~1.6B elements, memory bandwidth is the bottleneck — the compute is trivial.

| Task | Op | Difficulty | Energy/Call | Optimization Opportunity |
|------|-----|------------|-------------|-------------------------|
| L1_23 | Softmax | ★★☆☆☆ | ~0.02 J | Online softmax (one pass instead of three) saves ~40% memory traffic |
| L1_24 | LogSoftmax | ★★☆☆☆ | ~0.02 J | Same trick as Softmax plus log-sum-exp stability |
| L1_26 | GELU | ★☆☆☆☆ | ~0.01 J | Approximation vs exact trade-off |
| L1_25 | Swish | ★☆☆☆☆ | ~0.01 J | Simple: x × sigmoid(x). Fuse sigmoid into the multiply. |
| L1_19 | ReLU | ★☆☆☆☆ | ~0.008 J | Trivial element-wise. Bandwidth-limited. |
| L1_88 | MinGPT GELU | ★★☆☆☆ | ~0.03 J | 8192×8192 tensor. Uses tanh approximation — worth fusing with surrounding ops. |

### Normalization (L1_33–L1_40)

| Task | Op | Difficulty | Energy/Call | Notes |
|------|-----|------------|-------------|-------|
| L1_33 | BatchNorm | ★★☆☆☆ | ~0.02 J | Two-pass (mean+var, then normalize). Fusing into one warp-level pass is known but fiddly. |
| L1_35 | GroupNorm | ★★★☆☆ | ~0.03 J | Arbitrary group sizes make tiling uneven. |
| L1_40 | LayerNorm | ★★☆☆☆ | ~0.02 J | Welford's online algorithm for one-pass. Critical inner kernel for transformers. |

### Pooling (L1_41–L1_46)

| Task | Op | Difficulty | Energy/Call | Notes |
|------|-----|------------|-------------|-------|
| L1_45 | AvgPool2d 2048×2048, k=11 | ★★☆☆☆ | ~0.1 J | Huge spatial dims. Sliding window with large kernel = lots of redundant loads. |
| L1_46 | AvgPool3d 128×128×256, k=3 | ★★★☆☆ | ~0.08 J | 3D pooling. Non-trivial tiling in 3 spatial dimensions. |
| L1_43 | MaxPool3d | ★★☆☆☆ | ~0.05 J | Simpler reduction but 3D tiling challenge remains. |

### Reductions (L1_47–L1_53)

| Task | Op | Difficulty | Energy/Call | Notes |
|------|-----|------------|-------------|-------|
| L1_47 | Sum reduction | ★★☆☆☆ | ~0.02 J | Warp-level shuffle + block reduce. Standard pattern. |
| L1_49 | Max reduction | ★★☆☆☆ | ~0.02 J | Same pattern, different combiner. |
| L1_51 | Argmax | ★★★☆☆ | ~0.03 J | Carrying the index alongside the value adds register pressure. |

---

## Tier 4 — Lower Energy per Call, But Ubiquitous

### Small L3 Models

| Task | Model | Difficulty | Energy/Call | Production Frequency |
|------|-------|------------|-------------|---------------------|
| L3_4 | LeNet-5 | ★★☆☆☆ | ~0.01 J | Rarely used in production now |
| L3_1 | MLP | ★★☆☆☆ | ~0.01 J | Very common as sub-component |
| L3_8 | ResNet BasicBlock | ★★★☆☆ | ~0.05 J | Extremely common — the inner loop of ResNets |
| L3_17 | SqueezeNet Fire | ★★★☆☆ | ~0.03 J | Edge deployment |
| L3_19 | MobileNetV1 | ★★★☆☆ | ~0.05 J | Depthwise-separable convs dominate mobile inference |
| L3_31 | Vision Attention | ★★★☆☆ | ~0.02 J | Building block; optimized via FlashAttention |
| L3_33 | Vanilla RNN | ★★★☆☆ | ~0.02 J | Hidden=16384 makes each step GEMM-heavy |

### Simple L1 Operations

| Task | Op | Difficulty | Energy/Call |
|------|-----|------------|-------------|
| L1_5 | Matrix-scalar mul (65K×16K) | ★☆☆☆☆ | ~0.01 J |
| L1_12 | Diagonal matmul | ★★☆☆☆ | ~0.01 J |
| L1_94–L1_100 | Loss functions | ★★☆☆☆ | ~0.01–0.02 J |

---

## Top 15 — Hardest Problems to Solve (ranked)

| Rank | Task | Why |
|------|------|-----|
| 1 | **L3_30** SwinTransformerV2 | Shifted window attention with relative positional bias across multiple resolutions. No clean decomposition into standard kernels. |
| 2 | **L3_29** SwinMLP | Window partitioning + cyclic shift + MLP spatial mixing. Memory layout nightmares. |
| 3 | **L4_5** BigBird-RoBERTa (seq=4095) | Sparse attention pattern (block + random + global tokens). Writing an efficient sparse-attention CUDA kernel is a research contribution. |
| 4 | **L4_13** Reformer (enwik8) | LSH attention requires sorting/bucketing that's fundamentally at odds with GPU SIMT parallelism. |
| 5 | **L3_48** Mamba2 (return Y) | Selective structured state-space scan. No cuBLAS/cuDNN primitive exists — pure custom kernel territory. |
| 6 | **L3_49** Mamba2 (return state) | Same scan complexity, different output path. |
| 7 | **L4_1** GPT-Neo 2.7B (bs=32, seq=256) | Full 2.7B-param model. Requires graph-level fusion across 32 transformer layers. |
| 8 | **L3_44** MiniGPT Block | Full transformer block fusion: layernorm → attention → residual → layernorm → MLP → residual. Getting this into one or two kernel launches is the frontier. |
| 9 | **L1_93** Masked cumulative sum | Work-efficient parallel prefix sum with irregular masking. The mask creates segment boundaries that break standard Blelloch scan. |
| 10 | **L2_92** Conv2d → GN → Tanh → HardSwish → ResAdd → LogSumExp | GroupNorm mid-chain forces a global sync. Fusing around the barrier while maintaining numerical stability is the challenge. |
| 11 | **L3_15** DenseNet-121 | Dense block connectivity: each layer reads from ALL previous layers. Memory management (recompute vs store) is a hard scheduling problem. |
| 12 | **L2_89** ConvT3d → MaxPool → Softmax → Sub → Swish → Max | 6-deep fusion chain on 3D tensors with two different reduction types (pool + softmax + max). |
| 13 | **L1_90** Cumulative product | Sequential dependency + catastrophic numerical instability. Must use log-domain parallel scan. |
| 14 | **L3_43** MinGPT CausalAttention | Causal masking with multi-head attention. Tiling the triangular mask efficiently across thread blocks is non-trivial. |
| 15 | **L2_18** Matmul → Sum → Max → AvgPool → LogSumExp → LogSumExp | Giant 8K×8K GEMM followed by a 5-op reduction chain with two numerically sensitive LogSumExps. |

---

## Top 15 — Highest Energy Impact in Production (ranked by total joules saved if optimized)

These rankings consider: (a) energy per call, (b) how frequently the kernel runs in real ML pipelines, and (c) how much room there is to improve over naive PyTorch.

| Rank | Task | Est. Energy/Call | Why It Dominates |
|------|------|-----------------|------------------|
| 1 | **L4_1** GPT-Neo 2.7B (bs=32) | ~18–25 J | LLM inference is THE energy bottleneck of modern ML. This config represents the high-throughput serving regime. Every 10% improvement saves megawatt-hours at datacenter scale. |
| 2 | **L4_18** GPT-Neo 2.7B (bs=512) | ~20–28 J | Even higher batch. GEMM-saturated — optimal tiling saves enormous energy. |
| 3 | **L4_3** GPT-Neo 2.7B (seq=2047) | ~12–18 J | Long-context LLM inference. FlashAttention vs naive = 2–4× energy reduction. |
| 4 | **L4_8** OPT-1.3B (bs=512) | ~10–16 J | High-throughput LLM serving. |
| 5 | **L4_4** OPT-1.3B (bs=32) | ~8–14 J | Standard batch LLM inference. |
| 6 | **L4_17** BART-Large (bs=1024) | ~8–12 J | Seq2seq at scale. Translation, summarization workloads hit this config. |
| 7 | **L3_30** SwinTransformerV2 | ~4–8 J | The backbone of modern detection/segmentation. Runs on every image in a video stream. |
| 8 | **L3_10** ResNet-101 | ~2–4 J | Still the most deployed vision backbone. Billions of inference calls/day globally. |
| 9 | **L3_16** DenseNet-201 | ~3–5 J | Medical imaging workhorse. Runs on every scan. |
| 10 | **L3_7** GoogLeNet/InceptionV1 | ~1.5–3 J | Classic production model still widely deployed. |
| 11 | **L2_9** Matmul(1K×8K×8K) + epilogue | ~0.5–1 J | This GEMM size is the inner loop of training large language models. It runs billions of times. |
| 12 | **L1_3** Batched matmul 128×512×1024×2048 | ~0.4–0.8 J | Multi-head attention's core operation. Called on every token in every layer. |
| 13 | **L3_48** Mamba2 | ~1–3 J | Emerging alternative to transformers. If SSMs gain adoption, this kernel runs on every sequence. |
| 14 | **L1_2** GEMM 2K×8K×4K | ~0.3–0.6 J | The single most common shape in transformer FFN layers. |
| 15 | **L3_22** EfficientNet-B0 | ~1–2 J | The go-to efficient vision model. Deployed on billions of mobile devices. |

---

## Energy Estimation Methodology

Estimates assume an NVIDIA H100 SXM5 (700W TDP, ~300W typical for sustained compute) with:
- **Compute-bound kernels**: Energy ≈ FLOPs ÷ peak_FLOPS × TDP. H100 FP16 peak ≈ 990 TFLOPS.
- **Memory-bound kernels**: Energy ≈ bytes_transferred ÷ HBM_bandwidth × TDP. H100 HBM3 ≈ 3.35 TB/s.
- **Mixed workloads**: Weighted combination based on arithmetic intensity vs. the roofline crossover point (~200 FLOP/byte on H100).
- Real joule figures will vary ±50% based on GPU utilization, clock boosting, cooling, and whether the kernel is run in isolation or as part of a larger pipeline.

## Key Takeaway

If you're optimizing for **maximum joules saved**, focus on:
1. **L4 LLM inference tasks** — they dominate energy at the datacenter level
2. **Large GEMM operations** (L1_2, L1_3, L2_9, L2_12) — they're the inner loop of everything
3. **Attention kernels** (L1_97, L3_43, L3_44) — FlashAttention-style tiling is the single biggest energy win per kernel in the transformer era

If you're optimizing for **hardest kernel to write correctly**, focus on:
1. **Sparse/structured attention** (BigBird, Reformer) — novel memory access patterns
2. **State-space scans** (Mamba2) — no existing library support
3. **Parallel prefix operations with masks** (L1_93) — inherently sequential with irregular boundaries
4. **Deep fusion chains with normalization barriers** (L2_92, L2_89) — global sync points break fusion
