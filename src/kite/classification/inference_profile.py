"""Transformer inference energy budget and kernel relevance mapping.

Maps KernelBench task types to their approximate share of energy in
transformer model inference (models like Llama, GPT-4, Opus).  Used to
weight training rewards and project end-to-end energy savings.
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
    KERNEL_TYPE_NORM,
    KERNEL_TYPE_POOLING,
    KERNEL_TYPE_REDUCTION,
    KERNEL_TYPE_SCAN,
    KernelTask,
)

TRANSFORMER_ENERGY_WEIGHTS: dict[str, float] = {
    KERNEL_TYPE_MATMUL: 0.60,
    KERNEL_TYPE_ATTENTION: 0.15,
    KERNEL_TYPE_NORM: 0.08,
    KERNEL_TYPE_ACTIVATION: 0.05,
    KERNEL_TYPE_REDUCTION: 0.05,
    KERNEL_TYPE_COMPOSITE: 0.04,
    KERNEL_TYPE_POOLING: 0.02,
    KERNEL_TYPE_SCAN: 0.01,
}

_INFERENCE_CRITICAL_TYPES = frozenset({
    KERNEL_TYPE_MATMUL,
    KERNEL_TYPE_ATTENTION,
    KERNEL_TYPE_NORM,
    KERNEL_TYPE_ACTIVATION,
    KERNEL_TYPE_REDUCTION,
})

_DEPRIORITIZE_TYPES = frozenset({
    KERNEL_TYPE_CONV,
    KERNEL_TYPE_CONV_TRANSPOSE,
    KERNEL_TYPE_LOSS,
})

_RELEVANCE_MULTIPLIERS: dict[str, float] = {
    KERNEL_TYPE_MATMUL: 3.0,
    KERNEL_TYPE_ATTENTION: 2.5,
    KERNEL_TYPE_NORM: 2.0,
    KERNEL_TYPE_ACTIVATION: 2.0,
    KERNEL_TYPE_REDUCTION: 1.5,
    KERNEL_TYPE_COMPOSITE: 1.5,
    KERNEL_TYPE_POOLING: 1.0,
    KERNEL_TYPE_SCAN: 1.0,
    KERNEL_TYPE_CONV: 0.3,
    KERNEL_TYPE_CONV_TRANSPOSE: 0.3,
    KERNEL_TYPE_LOSS: 0.2,
}


def inference_relevance_weight(task: KernelTask) -> float:
    """Return a reward multiplier based on how important this kernel is for inference."""
    return _RELEVANCE_MULTIPLIERS.get(task.kernel_type, 1.0)


def is_inference_critical(task: KernelTask) -> bool:
    """Return True if task maps to a transformer inference operation."""
    return task.kernel_type in _INFERENCE_CRITICAL_TYPES


def energy_weight_for_type(kernel_type: str) -> float:
    """Fraction of total transformer inference energy attributed to this kernel type."""
    return TRANSFORMER_ENERGY_WEIGHTS.get(kernel_type, 0.0)


def oversample_for_inference(
    tasks: list[KernelTask],
    max_repeat: int = 3,
) -> list[KernelTask]:
    """Return an oversampled task list where inference-critical tasks appear more often."""
    result: list[KernelTask] = []
    for task in tasks:
        weight = _RELEVANCE_MULTIPLIERS.get(task.kernel_type, 1.0)
        repeats = min(max_repeat, max(1, int(round(weight))))
        result.extend([task] * repeats)
    return result
