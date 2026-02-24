"""Request and Batch dataclasses for inference simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from inference_simulator.types.execution import LLMStep


class RequestState(str, Enum):
    """Lifecycle states for an inference request."""

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    TOOL_EXECUTING = "tool_executing"
    AWAITING_NEXT_STEP = "awaiting_next_step"
    COMPLETED = "completed"


@dataclass
class Request:
    """A single inference request being processed by the simulator.

    Attributes:
        request_id: Unique identifier.
        arrival_time_ns: When the request arrived (simulation clock).
        input_tokens: Number of input/prompt tokens.
        max_output_tokens: Maximum tokens to generate.
        state: Current lifecycle state.
        tokens_generated: Number of output tokens generated so far.
        prefill_start_ns: When prefill began (None if not started).
        first_token_ns: When first output token was produced (None if not started).
        completion_ns: When generation completed (None if not finished).
        kv_cache_blocks: Number of KV cache blocks currently allocated.
        steps: Ordered LLM steps for multi-step requests (empty for single-step).
        current_step: Index of the step currently being processed.
        workload_type: Workload category (e.g., "chat", "agentic").
        step_prefill_times_ns: Prefill time per step.
        step_decode_times_ns: Decode time per step.
        step_tool_times_ns: Tool execution time per step.
    """

    request_id: int
    arrival_time_ns: int
    input_tokens: int
    max_output_tokens: int
    state: RequestState = RequestState.WAITING
    tokens_generated: int = 0
    prefill_start_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    completion_ns: Optional[int] = None
    kv_cache_blocks: int = 0
    steps: List[LLMStep] = field(default_factory=list)
    current_step: int = 0
    workload_type: str = ""
    step_prefill_times_ns: List[int] = field(default_factory=list)
    step_decode_times_ns: List[int] = field(default_factory=list)
    step_tool_times_ns: List[int] = field(default_factory=list)
    retention_policy: str = "retain"
    _offload_handle: Optional[dict] = field(default=None, repr=False)
    prefix_matched_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens processed so far (input + generated)."""
        return self.input_tokens + self.tokens_generated

    @property
    def is_complete(self) -> bool:
        """Whether generation is finished."""
        return self.state == RequestState.COMPLETED

    @property
    def is_multi_step(self) -> bool:
        """Whether this is a multi-step request."""
        return len(self.steps) > 0

    @property
    def current_llm_step(self) -> Optional[LLMStep]:
        """The LLM step currently being processed, or None if complete."""
        if self.steps and self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance_step(self) -> bool:
        """Advance to the next step. Returns True if more steps remain."""
        self.current_step += 1
        return self.current_step < len(self.steps)

    @property
    def ttft_ns(self) -> Optional[int]:
        """Time to first token in nanoseconds."""
        if self.first_token_ns is not None and self.arrival_time_ns is not None:
            return self.first_token_ns - self.arrival_time_ns
        return None

    @property
    def e2e_latency_ns(self) -> Optional[int]:
        """End-to-end latency in nanoseconds."""
        if self.completion_ns is not None and self.arrival_time_ns is not None:
            return self.completion_ns - self.arrival_time_ns
        return None


@dataclass
class Batch:
    """A batch of requests processed together in one forward pass.

    Attributes:
        batch_id: Unique identifier for this batch.
        requests: List of requests in this batch.
        is_prefill: Whether this batch is a prefill batch (vs decode).
    """

    batch_id: int
    requests: List[Request] = field(default_factory=list)
    is_prefill: bool = False

    @property
    def size(self) -> int:
        """Number of requests in this batch."""
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        """Total tokens in this batch for the current step."""
        if self.is_prefill:
            return sum(r.input_tokens for r in self.requests)
        # Decode: 1 token per request
        return len(self.requests)
