"""React agent implementation using the Agno framework."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from agno.agent import Agent
from agno.models.base import Model

from agents.base import BaseAgent

if TYPE_CHECKING:
    from ipw.src.telemetry.events import EventRecorder


class React(BaseAgent):
    """React agent that uses the Agno Agent framework for tool-augmented reasoning."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: Model,
        tools: List[Callable],
        instructions: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the React agent.

        Args:
            model: The model instance to use.
            tools: List of callable tools/functions for the agent to use.
            instructions: Optional custom instructions for the agent.
                Defaults to DEFAULT_INSTRUCTIONS.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to the Agent constructor.
        """
        super().__init__(event_recorder=event_recorder)

        self.model = model
        self._original_tools = tools
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

        # Instrument tools if event_recorder is provided
        if event_recorder:
            self.tools = self._instrument_tools(tools)
        else:
            self.tools = tools

        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            tool_choice="auto",
            instructions=self.instructions,
            **kwargs,
        )

    def _instrument_tools(self, tools: List[Callable]) -> List[Callable]:
        """Wrap tools to emit start/end events for energy tracking.

        Args:
            tools: List of tool functions to instrument.

        Returns:
            List of instrumented tool functions that emit events.
        """
        instrumented = []
        for tool in tools:
            tool_name = getattr(tool, "__name__", str(tool))

            @functools.wraps(tool)
            def wrapper(
                *args: Any,
                __tool: Callable = tool,
                __name: str = tool_name,
                **kwargs: Any,
            ) -> Any:
                self._record_event("tool_call_start", tool=__name)
                try:
                    return __tool(*args, **kwargs)
                finally:
                    self._record_event("tool_call_end", tool=__name)

            instrumented.append(wrapper)
        return instrumented

    def run(self, input: str, **kwargs: Any) -> Any:
        """Run the React agent.

        Args:
            input: The input message or prompt for the agent.
            **kwargs: Additional keyword arguments passed to agent.run().

        Returns:
            The output from the agent.run() call.
        """
        self._record_event("lm_inference_start", model=str(self.model))
        result = None
        try:
            result = self.agent.run(input, **kwargs)
            # Extract token metrics from the result if available
            end_metadata: dict[str, Any] = {"model": str(self.model)}
            if result is not None and hasattr(result, "metrics") and result.metrics is not None:
                metrics = result.metrics
                end_metadata["prompt_tokens"] = getattr(metrics, "input_tokens", 0)
                end_metadata["completion_tokens"] = getattr(metrics, "output_tokens", 0)
                end_metadata["total_tokens"] = getattr(metrics, "total_tokens", 0)
            self._record_event("lm_inference_end", **end_metadata)
            return result
        except Exception:
            # Record end event even on failure
            self._record_event("lm_inference_end", model=str(self.model))
            raise
