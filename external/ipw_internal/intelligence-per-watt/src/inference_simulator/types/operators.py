"""Operator categories and measurement dataclasses for profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class OperatorCategory(str, Enum):
    """Categories of operators profiled in the dataset generator."""

    LINEAR = "linear"
    LM_HEAD = "lm_head"
    ATTENTION_PREFILL = "attention_prefill"
    ATTENTION_DECODE = "attention_decode"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    MOE_ROUTING = "moe_routing"
    MOE_EXPERT = "moe_expert"
    SSM_SCAN = "ssm_scan"
    COMMUNICATION = "communication"
    AGENTIC_TOOL = "agentic_tool"
    SAMPLING = "sampling"
    MTP = "mtp"
    CPU_HOST = "cpu_host"
    KV_CACHE = "kv_cache"
    FUSED_PREFILL = "fused_prefill"
    FUSED_DECODE_STEP = "fused_decode_step"
    FUSED_ATTENTION = "fused_attention"
    FUSED_MLP = "fused_mlp"
    FUSED_NORM_ATTN = "fused_norm_attn"


@dataclass
class OperatorMeasurement:
    """A single profiling measurement for an operator.

    Captures timing, energy, and throughput metrics for one operator
    at a specific (batch_size, seq_len) configuration point.
    """

    operator_name: str
    category: OperatorCategory
    batch_size: int
    seq_len: int
    time_s: float
    energy_j: Optional[float] = None
    power_w: Optional[float] = None
    flops: Optional[int] = None
    bytes_accessed: Optional[int] = None
    bandwidth_gb_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tflops(self) -> Optional[float]:
        """Achieved TFLOPS (tera floating-point ops per second)."""
        if self.flops is not None and self.time_s > 0:
            return self.flops / self.time_s / 1e12
        return None

    @property
    def arithmetic_intensity(self) -> Optional[float]:
        """FLOPs per byte accessed."""
        if self.flops is not None and self.bytes_accessed is not None and self.bytes_accessed > 0:
            return self.flops / self.bytes_accessed
        return None
