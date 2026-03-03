"""Classify KernelBench tasks into kernel-type categories.

The goal is to enable energy-vs-compute analysis *per kernel type*:
two kernels with similar runtime but different computational patterns
(e.g. matmul-heavy vs reduction-heavy) may have very different energy
profiles.  This classifier assigns a category based on the task name
and/or reference source code so that downstream analysis can group by
kernel_type.
"""

from __future__ import annotations

import re
from typing import Optional

from kite.types import (
    KERNEL_TYPE_ACTIVATION,
    KERNEL_TYPE_ATTENTION,
    KERNEL_TYPE_COMPOSITE,
    KERNEL_TYPE_CONV,
    KERNEL_TYPE_CONV_TRANSPOSE,
    KERNEL_TYPE_ELEMENTWISE,
    KERNEL_TYPE_LOSS,
    KERNEL_TYPE_MATMUL,
    KERNEL_TYPE_MODEL,
    KERNEL_TYPE_NORM,
    KERNEL_TYPE_POOLING,
    KERNEL_TYPE_REDUCTION,
    KERNEL_TYPE_RNN,
    KERNEL_TYPE_SCAN,
    KERNEL_TYPE_UNKNOWN,
)

# Ordered: first match wins. Patterns are tested against a lowercased
# concatenation of task name + problem name.
_NAME_RULES: list[tuple[re.Pattern, str]] = [
    # Level 4 full models (name contains model arch identifiers)
    (re.compile(r"(gpt2|opt-|bart|neo|bigbird|reformer|electra)"), KERNEL_TYPE_MODEL),

    # Attention
    (re.compile(r"(attention|sdpa|scaleddotproduct)"), KERNEL_TYPE_ATTENTION),

    # RNN family
    (re.compile(r"(lstm|gru(?!po)|rnn)"), KERNEL_TYPE_RNN),

    # Scan-like ops: cumsum, cumprod, etc.
    (re.compile(r"(cumsum|cumprod|scan)"), KERNEL_TYPE_SCAN),

    # Loss functions
    (re.compile(r"(loss|mseloss|crossentropy|huber|kldiv|tripletmargin|hinge)"), KERNEL_TYPE_LOSS),

    # Transposed convolutions (before regular conv so "convtranspose" matches first)
    (re.compile(r"conv_?transpose"), KERNEL_TYPE_CONV_TRANSPOSE),
    (re.compile(r"convtranspose"), KERNEL_TYPE_CONV_TRANSPOSE),

    # Regular convolutions (including depthwise, pointwise, separable)
    (re.compile(r"\bconv\b|conv_standard|conv_depthwise|conv_pointwise|conv_separable|conv[123]d"), KERNEL_TYPE_CONV),

    # Pooling
    (re.compile(r"pool|avgpool|maxpool|globalavgpool"), KERNEL_TYPE_POOLING),

    # Normalization
    (re.compile(r"(batchnorm|instancenorm|groupnorm|layernorm|rmsnorm|frobeniusnorm|l[12]norm)"), KERNEL_TYPE_NORM),

    # Activations (no word-boundary anchors — names like "19_ReLU" need to match)
    (re.compile(r"(relu|leakyrelu|sigmoid|tanh|softmax|logsoftmax|swish|gelu|selu|"
                r"hardsigmoid|softplus|softsign|(?<![a-z])elu(?![a-z])|hardtanh|mish|hardswish|newgelu)"), KERNEL_TYPE_ACTIVATION),

    # Reductions (check before matmul since "sum" is common in level-2 composite names)
    (re.compile(r"(sum_reduction|mean_reduction|max_reduction|min_reduction|argmax|argmin)"), KERNEL_TYPE_REDUCTION),

    # Matrix multiplications
    (re.compile(r"(matmul|gemm|matrix.?mult|matrix.?vector|matrix.?scalar|mat_mul)"), KERNEL_TYPE_MATMUL),
]

# Source-code patterns (fallback when the name doesn't match).
_CODE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"torch\.matmul|torch\.mm\b|torch\.bmm\b|@\s*\w+\.T\b|nn\.Linear", re.IGNORECASE), KERNEL_TYPE_MATMUL),
    (re.compile(r"nn\.Conv[123]d\b", re.IGNORECASE), KERNEL_TYPE_CONV),
    (re.compile(r"nn\.ConvTranspose[123]d\b", re.IGNORECASE), KERNEL_TYPE_CONV_TRANSPOSE),
    (re.compile(r"nn\.(BatchNorm|InstanceNorm|GroupNorm|LayerNorm)", re.IGNORECASE), KERNEL_TYPE_NORM),
    (re.compile(r"nn\.(ReLU|LeakyReLU|Sigmoid|Tanh|Softmax|GELU|SiLU|Mish|ELU|SELU|Hardsigmoid|Hardswish)", re.IGNORECASE), KERNEL_TYPE_ACTIVATION),
    (re.compile(r"nn\.(MaxPool|AvgPool|AdaptiveAvgPool|AdaptiveMaxPool)", re.IGNORECASE), KERNEL_TYPE_POOLING),
    (re.compile(r"nn\.(LSTM|GRU|RNN)\b", re.IGNORECASE), KERNEL_TYPE_RNN),
    (re.compile(r"nn\.(MSELoss|CrossEntropyLoss|L1Loss|HuberLoss|KLDivLoss|TripletMarginLoss)", re.IGNORECASE), KERNEL_TYPE_LOSS),
    (re.compile(r"torch\.(cumsum|cumprod)\b", re.IGNORECASE), KERNEL_TYPE_SCAN),
    (re.compile(r"torch\.(sum|mean|max|min|argmax|argmin)\b", re.IGNORECASE), KERNEL_TYPE_REDUCTION),
]


def classify_kernel(
    task_name: str = "",
    problem_name: str = "",
    reference_code: str = "",
    level: int = 1,
) -> str:
    """Return a kernel_type string for a KernelBench task.

    Classification strategy:
    1. Level-2 tasks with 2+ operations in the name -> COMPOSITE.
    2. Name-based rules (first match wins).
    3. Source-code-based rules (first match wins).
    4. Level-3/4 tasks that didn't match -> MODEL.
    5. Fallback -> UNKNOWN.
    """
    name_lower = f"{task_name} {problem_name}".lower().replace("-", "_")

    if level == 2 and _count_ops_in_name(name_lower) >= 2:
        return KERNEL_TYPE_COMPOSITE

    for pattern, ktype in _NAME_RULES:
        if pattern.search(name_lower):
            return ktype

    code_lower = reference_code.lower() if reference_code else ""
    if code_lower:
        for pattern, ktype in _CODE_RULES:
            if pattern.search(code_lower):
                return ktype

    if level >= 3:
        return KERNEL_TYPE_MODEL

    return KERNEL_TYPE_UNKNOWN


def _count_ops_in_name(name: str) -> int:
    """Rough count of distinct operations mentioned in a level-2 name."""
    ops = [
        "conv", "gemm", "matmul", "relu", "gelu", "sigmoid", "tanh", "softmax",
        "batchnorm", "groupnorm", "instancenorm", "layernorm", "pool", "avgpool",
        "maxpool", "sum", "mean", "max", "min", "add", "subtract", "multiply",
        "divide", "clamp", "dropout", "mish", "swish", "hardswish", "hardtanh",
        "leakyrelu", "logsumexp", "biasadd", "scaling", "residualadd",
    ]
    found = set()
    for op in ops:
        if op in name:
            found.add(op)
    return len(found)
