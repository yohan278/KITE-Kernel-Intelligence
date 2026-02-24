"""SLA specification (stub for Pipeline #3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class SLASpec:
    """Service-level agreement constraints for inference search.

    Stub — full implementation comes with Pipeline #3 (search).
    """

    max_ttft_s: float = 1.0
    max_tpot_s: float = 0.1
    max_e2e_latency_s: float = 30.0
    min_throughput_tps: float = 10.0
    max_cost_per_1k_tokens_usd: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
