"""Instrumentation utilities for adding telemetry to agents."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar

if TYPE_CHECKING:
    from .events import EventRecorder

T = TypeVar("T")


class InstrumentedAgent:
    """Wrapper that adds energy telemetry to any agent.

    Wraps an agent's tools and run method to emit telemetry events that can
    be correlated with energy samples for per-action energy attribution.

    Example:
        >>> from ipw.telemetry.events import EventRecorder
        >>> from ipw.telemetry.instrumentation import InstrumentedAgent
        >>>
        >>> recorder = EventRecorder()
        >>> agent = MyAgent(tools=[calc_tool, search_tool])
        >>> instrumented = InstrumentedAgent(agent, recorder)
        >>>
        >>> result = instrumented.run("What is 2+2?")
        >>> events = recorder.get_events()
        >>> # events now contains tool_call_start/end and lm_inference_start/end
    """

    def __init__(
        self,
        agent: Any,
        event_recorder: "EventRecorder",
        *,
        tools_attr: str = "tools",
        instrument_run: bool = True,
    ) -> None:
        """Initialize the instrumented agent.

        Args:
            agent: The agent to instrument. Should have a run() method
                and optionally a tools attribute.
            event_recorder: EventRecorder to emit events to.
            tools_attr: Name of the attribute containing tools (default: "tools").
            instrument_run: Whether to wrap run() with lm_inference events.
        """
        self._agent = agent
        self._event_recorder = event_recorder
        self._tools_attr = tools_attr
        self._instrument_run = instrument_run
        self._original_tools: Optional[Dict[str, Any]] = None

        # Instrument tools if they exist
        if hasattr(agent, tools_attr):
            self._instrument_tools()

    def _instrument_tools(self) -> None:
        """Wrap tools to emit tool_call_start/end events."""
        tools = getattr(self._agent, self._tools_attr)
        if tools is None:
            return

        # Store original tools for potential restoration
        self._original_tools = dict(tools) if isinstance(tools, dict) else None

        if isinstance(tools, dict):
            instrumented_tools = {}
            for name, tool in tools.items():
                instrumented_tools[name] = self._wrap_tool(tool, name)
            setattr(self._agent, self._tools_attr, instrumented_tools)
        elif isinstance(tools, list):
            instrumented_tools = []
            for tool in tools:
                tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))
                instrumented_tools.append(self._wrap_tool(tool, tool_name))
            setattr(self._agent, self._tools_attr, instrumented_tools)

    def _wrap_tool(self, tool: Any, tool_name: str) -> Any:
        """Wrap a single tool to emit events.

        Args:
            tool: The tool to wrap (callable or object with __call__).
            tool_name: Name to use in event metadata.

        Returns:
            Wrapped tool that emits telemetry events.
        """
        recorder = self._event_recorder
        original_tool = tool  # Capture in closure

        if callable(tool) and not hasattr(tool, "__call__"):
            # Plain function
            @functools.wraps(tool)
            def wrapped_func(*args: Any, **kwargs: Any) -> Any:
                recorder.record("tool_call_start", tool=tool_name)
                try:
                    return original_tool(*args, **kwargs)
                finally:
                    recorder.record("tool_call_end", tool=tool_name)

            return wrapped_func

        elif hasattr(tool, "__call__"):
            # Object with __call__ method (e.g., Tool class instance)
            class WrappedTool:
                def __init__(self, original: Any, name: str) -> None:
                    self._original = original
                    self._name = name
                    # Copy attributes from original
                    for attr in dir(original):
                        if not attr.startswith("_") and attr != "__call__":
                            try:
                                setattr(self, attr, getattr(original, attr))
                            except AttributeError:
                                pass

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    recorder.record("tool_call_start", tool=self._name)
                    try:
                        return self._original(*args, **kwargs)
                    finally:
                        recorder.record("tool_call_end", tool=self._name)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._original, name)

            return WrappedTool(tool, tool_name)

        # Not callable, return as-is
        return tool

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent with optional lm_inference event wrapping.

        Args:
            *args: Positional arguments for agent.run().
            **kwargs: Keyword arguments for agent.run().

        Returns:
            Result from agent.run().
        """
        if self._instrument_run:
            self._event_recorder.record("lm_inference_start")
            try:
                return self._agent.run(*args, **kwargs)
            finally:
                self._event_recorder.record("lm_inference_end")
        else:
            return self._agent.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to wrapped agent."""
        return getattr(self._agent, name)


def instrument_agent(
    agent: T,
    event_recorder: "EventRecorder",
    *,
    tools_attr: str = "tools",
    instrument_run: bool = True,
) -> InstrumentedAgent:
    """Create an instrumented agent wrapper.

    Convenience factory function for creating InstrumentedAgent instances.

    Args:
        agent: The agent to instrument.
        event_recorder: EventRecorder to emit events to.
        tools_attr: Name of the attribute containing tools (default: "tools").
        instrument_run: Whether to wrap run() with lm_inference events.

    Returns:
        InstrumentedAgent wrapping the provided agent.

    Example:
        >>> from ipw.telemetry.events import EventRecorder
        >>> from ipw.telemetry.instrumentation import instrument_agent
        >>>
        >>> recorder = EventRecorder()
        >>> agent = create_my_agent()
        >>> instrumented = instrument_agent(agent, recorder)
        >>> result = instrumented.run("Hello!")
    """
    return InstrumentedAgent(
        agent,
        event_recorder,
        tools_attr=tools_attr,
        instrument_run=instrument_run,
    )


__all__ = ["InstrumentedAgent", "instrument_agent"]
