"""Workload specification for inference simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class WorkloadType(str, Enum):
    """High-level workload categories with distinct profiling characteristics."""

    CHAT = "chat"
    REASONING = "reasoning"
    AGENTIC = "agentic"
    RAG = "rag"
    CODING = "coding"


@dataclass(frozen=True)
class WorkloadSpec:
    """Specification for an inference workload pattern.

    Attributes:
        qps: Queries per second (arrival rate).
        avg_input_tokens: Average input tokens per request.
        avg_output_tokens: Average output tokens per request.
        input_token_std: Standard deviation of input tokens.
        output_token_std: Standard deviation of output tokens.
        workload_type: High-level workload category.
        max_output_tokens: Hard cap on output tokens per step.
        num_tool_calls_range: (min, max) tool calls for agentic workloads.
        context_extension_per_tool: Extra context tokens added per tool call.
        num_retrieved_docs_range: (min, max) retrieved documents for RAG.
        tokens_per_doc: Tokens per retrieved document.
        burstiness: Controls inter-arrival time distribution shape via
            ``np.random.gamma(shape=burstiness)``.  1.0 = Poisson (exponential
            inter-arrivals), <1.0 = bursty (more clustered arrivals),
            >1.0 = more uniform (less variance in inter-arrivals).
        metadata: Additional workload-specific parameters.
    """

    qps: float = 1.0
    avg_input_tokens: int = 500
    avg_output_tokens: int = 200
    input_token_std: float = 200.0
    output_token_std: float = 100.0
    workload_type: Optional[WorkloadType] = None
    max_output_tokens: int = 32768
    num_tool_calls_range: Tuple[int, int] = (0, 0)
    context_extension_per_tool: int = 512
    num_retrieved_docs_range: Tuple[int, int] = (0, 0)
    tokens_per_doc: int = 1024
    burstiness: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_chat(cls, qps: float = 1.0) -> WorkloadSpec:
        """Create a chat workload spec: 1024 input / 1024 output."""
        return cls(
            qps=qps,
            avg_input_tokens=1024,
            avg_output_tokens=1024,
            input_token_std=256.0,
            output_token_std=256.0,
            workload_type=WorkloadType.CHAT,
            max_output_tokens=2048,
        )

    @classmethod
    def for_reasoning(cls, qps: float = 1.0) -> WorkloadSpec:
        """Create a reasoning workload spec: 1024 input / 4096-32768 output."""
        return cls(
            qps=qps,
            avg_input_tokens=1024,
            avg_output_tokens=16384,
            input_token_std=256.0,
            output_token_std=8192.0,
            workload_type=WorkloadType.REASONING,
            max_output_tokens=32768,
        )

    @classmethod
    def for_agentic(cls, qps: float = 1.0) -> WorkloadSpec:
        """Create an agentic workload spec: 1024/1024 per step, 1-8 tool calls."""
        return cls(
            qps=qps,
            avg_input_tokens=1024,
            avg_output_tokens=1024,
            input_token_std=256.0,
            output_token_std=256.0,
            workload_type=WorkloadType.AGENTIC,
            max_output_tokens=2048,
            num_tool_calls_range=(1, 8),
            context_extension_per_tool=512,
        )

    @classmethod
    def for_rag(cls, qps: float = 1.0) -> WorkloadSpec:
        """Create a RAG workload spec: query + 10 docs x 1024 tokens."""
        return cls(
            qps=qps,
            avg_input_tokens=1024,
            avg_output_tokens=1024,
            input_token_std=256.0,
            output_token_std=256.0,
            workload_type=WorkloadType.RAG,
            max_output_tokens=2048,
            num_retrieved_docs_range=(5, 15),
            tokens_per_doc=1024,
        )

    @classmethod
    def for_coding(cls, qps: float = 1.0) -> WorkloadSpec:
        """Create a coding workload spec: similar to agentic with code tools."""
        return cls(
            qps=qps,
            avg_input_tokens=1024,
            avg_output_tokens=2048,
            input_token_std=512.0,
            output_token_std=1024.0,
            workload_type=WorkloadType.CODING,
            max_output_tokens=4096,
            num_tool_calls_range=(1, 6),
            context_extension_per_tool=1024,
        )
