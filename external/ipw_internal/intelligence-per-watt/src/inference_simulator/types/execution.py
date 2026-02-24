"""Multi-step execution model for agentic and RAG workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolCall:
    """Describes a tool invocation between LLM steps.

    Attributes:
        tool_type: Tool category (e.g., "web_search", "code_interpreter",
            "calculator", "faiss_retrieval", "file_io", "api_call").
        tool_config: Key into tool_distributions.pkl, e.g., "warm_pool10",
            "hnsw_1m_k5".
    """

    tool_type: str
    tool_config: str = "default"


@dataclass
class LLMStep:
    """One LLM forward pass within a multi-step request.

    Attributes:
        input_tokens: Tokens added at this step (new prompt/context).
        output_tokens: Tokens to generate in this step.
        cumulative_context: Total context length at the start of this step.
        tool_call: Tool to run after decode completes (None if final step).
    """

    input_tokens: int
    output_tokens: int
    cumulative_context: int = 0
    tool_call: Optional[ToolCall] = None


@dataclass
class MultiStepRequest:
    """A multi-step inference request with interleaved LLM and tool execution.

    Used for agentic workloads where a single user request triggers
    multiple LLM inference steps separated by tool calls.

    Attributes:
        request_id: Unique identifier for this request.
        arrival_time_ns: When the request arrived (simulation clock).
        workload_type: Workload category ("chat", "reasoning", "agentic",
            "rag", "coding").
        steps: Ordered list of LLM steps to execute.
        current_step: Index of the step currently being processed.
        step_prefill_times_ns: Accumulated prefill time per step.
        step_decode_times_ns: Accumulated decode time per step.
        step_tool_times_ns: Accumulated tool execution time per step.
    """

    request_id: str
    arrival_time_ns: int
    workload_type: str
    steps: List[LLMStep] = field(default_factory=list)
    current_step: int = 0
    step_prefill_times_ns: List[int] = field(default_factory=list)
    step_decode_times_ns: List[int] = field(default_factory=list)
    step_tool_times_ns: List[int] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Total number of LLM steps."""
        return len(self.steps)

    @property
    def is_complete(self) -> bool:
        """Whether all steps have been processed."""
        return self.current_step >= len(self.steps)

    @property
    def current_llm_step(self) -> Optional[LLMStep]:
        """The LLM step currently being processed, or None if complete."""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all steps."""
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all steps."""
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_prefill_time_ns(self) -> int:
        """Sum of prefill times across all steps."""
        return sum(self.step_prefill_times_ns)

    @property
    def total_decode_time_ns(self) -> int:
        """Sum of decode times across all steps."""
        return sum(self.step_decode_times_ns)

    @property
    def total_tool_time_ns(self) -> int:
        """Sum of tool execution times across all steps."""
        return sum(self.step_tool_times_ns)


__all__ = ["LLMStep", "MultiStepRequest", "ToolCall"]
