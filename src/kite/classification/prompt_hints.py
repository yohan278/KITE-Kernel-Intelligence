"""Per-kernel-type optimization hints for energy-aware generation.

These hints are injected into the GRPO prompt so the model has context
about what kind of kernel it's writing and what optimization strategies
matter for energy efficiency for that kernel type.
"""

from __future__ import annotations

from kite.types import (
    KERNEL_TYPE_ACTIVATION,
    KERNEL_TYPE_ATTENTION,
    KERNEL_TYPE_COMPOSITE,
    KERNEL_TYPE_CONV,
    KERNEL_TYPE_CONV_TRANSPOSE,
    KERNEL_TYPE_LOSS,
    KERNEL_TYPE_MATMUL,
    KERNEL_TYPE_MODEL,
    KERNEL_TYPE_NORM,
    KERNEL_TYPE_POOLING,
    KERNEL_TYPE_REDUCTION,
    KERNEL_TYPE_RNN,
    KERNEL_TYPE_SCAN,
)

_HINTS: dict[str, str] = {
    KERNEL_TYPE_MATMUL: (
        "This is a matrix multiplication kernel (compute-bound). "
        "Focus on tiling, shared memory reuse, and tensor core utilization "
        "to maximize compute efficiency per watt. Avoid unnecessary global memory round-trips."
    ),
    KERNEL_TYPE_CONV: (
        "This is a convolution kernel. Consider im2col vs direct conv, "
        "memory access coalescing, and channel-last layout. "
        "Convolutions are often memory-bound; minimize intermediate buffer allocations."
    ),
    KERNEL_TYPE_CONV_TRANSPOSE: (
        "This is a transposed convolution kernel. Similar to regular conv but "
        "with scatter-style writes. Coalesce memory accesses and consider "
        "using cuDNN or direct implementations that minimize energy from scattered writes."
    ),
    KERNEL_TYPE_ACTIVATION: (
        "This is an element-wise activation kernel (bandwidth-bound). "
        "Fuse with adjacent operations if possible to minimize memory round-trips, "
        "which dominate energy consumption for bandwidth-bound kernels."
    ),
    KERNEL_TYPE_NORM: (
        "This is a normalization kernel. The reduction step is memory-bound. "
        "Use warp-level reductions, minimize global memory passes, and "
        "consider fusing with adjacent layers to reduce memory traffic."
    ),
    KERNEL_TYPE_REDUCTION: (
        "This is a reduction kernel. Use warp-level primitives (warp shuffle) "
        "and hierarchical reduction to minimize global memory writes. "
        "Reductions are memory-bound; fewer memory transactions means less energy."
    ),
    KERNEL_TYPE_POOLING: (
        "This is a pooling kernel. Memory access pattern matters: "
        "use coalesced reads and minimize redundant loads from overlapping windows."
    ),
    KERNEL_TYPE_SCAN: (
        "This is a scan/prefix-sum kernel. Use work-efficient parallel scan algorithms "
        "(Blelloch or Hillis-Steele). Minimize global memory writes per step."
    ),
    KERNEL_TYPE_LOSS: (
        "This is a loss function kernel. Typically involves a reduction. "
        "Fuse the forward computation with the reduction to avoid materializing "
        "large intermediate tensors."
    ),
    KERNEL_TYPE_ATTENTION: (
        "This is an attention kernel (mixed compute and memory bound). "
        "Use flash-attention-style tiling to keep working set in SRAM, "
        "avoiding large materialized attention matrices that waste memory bandwidth and energy."
    ),
    KERNEL_TYPE_RNN: (
        "This is a recurrent network kernel. The sequential dependency limits parallelism. "
        "Focus on efficient batched GEMM for gates and minimize per-step memory overhead."
    ),
    KERNEL_TYPE_COMPOSITE: (
        "This is a fused multi-operation kernel. The key energy win is operator fusion: "
        "minimize intermediate writes to global memory between operations. "
        "Each eliminated round-trip to DRAM saves significant energy."
    ),
    KERNEL_TYPE_MODEL: (
        "This is a full model architecture. Identify the bottleneck layers "
        "and focus optimization on those. Operator fusion across layer boundaries "
        "and memory-efficient attention are the highest-impact energy optimizations."
    ),
}

_DEFAULT_HINT = (
    "Optimize for both speed and energy efficiency. "
    "Minimize unnecessary memory traffic and maximize compute utilization."
)


def get_energy_hint(kernel_type: str) -> str:
    """Return an energy-optimization hint for the given kernel type."""
    return _HINTS.get(kernel_type, _DEFAULT_HINT)


def build_energy_aware_prompt(
    ref_src: str,
    kernel_type: str = "unknown",
) -> str:
    """Build a GRPO prompt with kernel-type context and energy hints."""
    hint = get_energy_hint(kernel_type)
    type_label = kernel_type.replace("_", " ")

    return (
        "You are an expert GPU kernel engineer.\n"
        f"Kernel type: {type_label}.\n"
        f"{hint}\n\n"
        "Optimize this PyTorch model with a custom GPU kernel implementation "
        "that produces identical outputs and is both fast AND energy-efficient "
        "on NVIDIA GPUs.\n\n"
        "Requirements:\n"
        "- Output only valid Python code.\n"
        "- Do not include markdown or explanations.\n"
        "- Must define class ModelNew(nn.Module).\n"
        "- ModelNew.forward must keep the same input argument count/order "
        "as the reference Model.forward.\n"
        "- Do not use Triton.\n"
        "- Minimize unnecessary memory traffic and maximize compute utilization.\n\n"
        f"```python\n{ref_src}\n```\n\n"
        "Write the optimized kernel:"
    )
