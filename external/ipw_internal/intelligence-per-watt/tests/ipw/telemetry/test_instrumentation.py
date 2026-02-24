"""Tests for telemetry instrumentation."""

from typing import Any, Dict, List

import pytest

from ipw.telemetry.events import EventRecorder
from ipw.telemetry.instrumentation import InstrumentedAgent, instrument_agent


class MockTool:
    """Mock tool class for testing."""

    def __init__(self, name: str, return_value: Any = "result") -> None:
        self.name = name
        self.return_value = return_value
        self.call_count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.return_value


class MockAgent:
    """Mock agent for testing instrumentation."""

    def __init__(self, tools: Any = None) -> None:
        self.tools = tools
        self.run_count = 0

    def run(self, input: str, **kwargs: Any) -> str:
        self.run_count += 1
        return f"processed: {input}"


class TestInstrumentedAgent:
    """Tests for InstrumentedAgent class."""

    def test_instrument_run_emits_events(self) -> None:
        """Test that run() emits lm_inference_start/end events."""
        recorder = EventRecorder()
        agent = MockAgent()
        instrumented = InstrumentedAgent(agent, recorder)

        result = instrumented.run("test input")

        assert result == "processed: test input"
        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "lm_inference_start"
        assert events[1].event_type == "lm_inference_end"

    def test_instrument_run_disabled(self) -> None:
        """Test that instrument_run=False skips lm_inference events."""
        recorder = EventRecorder()
        agent = MockAgent()
        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)

        result = instrumented.run("test input")

        assert result == "processed: test input"
        assert len(recorder) == 0

    def test_instrument_dict_tools(self) -> None:
        """Test instrumentation of dict-based tools."""
        recorder = EventRecorder()
        tools = {
            "calc": MockTool("calc", 42),
            "search": MockTool("search", ["result1"]),
        }
        agent = MockAgent(tools=tools)
        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)

        # Call the tools through the agent
        result1 = instrumented.tools["calc"]()
        result2 = instrumented.tools["search"]()

        assert result1 == 42
        assert result2 == ["result1"]

        events = recorder.get_events()
        assert len(events) == 4
        assert events[0].event_type == "tool_call_start"
        assert events[0].metadata["tool"] == "calc"
        assert events[1].event_type == "tool_call_end"
        assert events[1].metadata["tool"] == "calc"
        assert events[2].event_type == "tool_call_start"
        assert events[2].metadata["tool"] == "search"
        assert events[3].event_type == "tool_call_end"
        assert events[3].metadata["tool"] == "search"

    def test_instrument_list_tools(self) -> None:
        """Test instrumentation of list-based tools."""
        recorder = EventRecorder()
        tools = [MockTool("tool1"), MockTool("tool2")]
        agent = MockAgent(tools=tools)
        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)

        instrumented.tools[0]()

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[0].metadata["tool"] == "tool1"

    def test_proxy_attribute_access(self) -> None:
        """Test that attributes are proxied to wrapped agent."""
        recorder = EventRecorder()
        agent = MockAgent()
        agent.custom_attr = "custom_value"
        instrumented = InstrumentedAgent(agent, recorder)

        assert instrumented.custom_attr == "custom_value"

    def test_custom_tools_attr(self) -> None:
        """Test instrumentation with custom tools attribute name."""
        recorder = EventRecorder()

        class CustomAgent:
            def __init__(self) -> None:
                self.my_tools = {"calc": MockTool("calc")}

            def run(self, input: str) -> str:
                return input

        agent = CustomAgent()
        instrumented = InstrumentedAgent(
            agent, recorder, tools_attr="my_tools", instrument_run=False
        )

        instrumented.my_tools["calc"]()

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].metadata["tool"] == "calc"

    def test_tool_exception_still_emits_end_event(self) -> None:
        """Test that tool_call_end is emitted even on exception."""
        recorder = EventRecorder()

        def failing_tool() -> None:
            raise ValueError("Tool failed!")

        agent = MockAgent(tools={"fail": failing_tool})
        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)

        with pytest.raises(ValueError, match="Tool failed!"):
            instrumented.tools["fail"]()

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[1].event_type == "tool_call_end"

    def test_run_exception_still_emits_end_event(self) -> None:
        """Test that lm_inference_end is emitted even on exception."""
        recorder = EventRecorder()

        class FailingAgent:
            def run(self, input: str) -> str:
                raise RuntimeError("Agent failed!")

        agent = FailingAgent()
        instrumented = InstrumentedAgent(agent, recorder)

        with pytest.raises(RuntimeError, match="Agent failed!"):
            instrumented.run("test")

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "lm_inference_start"
        assert events[1].event_type == "lm_inference_end"

    def test_none_tools_handled(self) -> None:
        """Test that None tools are handled gracefully."""
        recorder = EventRecorder()
        agent = MockAgent(tools=None)
        instrumented = InstrumentedAgent(agent, recorder)

        result = instrumented.run("test")
        assert result == "processed: test"

    def test_combined_run_and_tool_events(self) -> None:
        """Test run with tool calls produces correct event sequence."""
        recorder = EventRecorder()
        tools = {"calc": MockTool("calc")}
        agent = MockAgent(tools=tools)
        instrumented = InstrumentedAgent(agent, recorder)

        # Simulate: run starts, tool is called, run ends
        instrumented.run("test")
        instrumented.tools["calc"]()

        events = recorder.get_events()
        assert len(events) == 4
        # Run events
        assert events[0].event_type == "lm_inference_start"
        assert events[1].event_type == "lm_inference_end"
        # Tool events
        assert events[2].event_type == "tool_call_start"
        assert events[3].event_type == "tool_call_end"


class TestInstrumentAgentFactory:
    """Tests for instrument_agent factory function."""

    def test_factory_creates_instrumented_agent(self) -> None:
        """Test that factory function creates InstrumentedAgent."""
        recorder = EventRecorder()
        agent = MockAgent()

        result = instrument_agent(agent, recorder)

        assert isinstance(result, InstrumentedAgent)

    def test_factory_passes_options(self) -> None:
        """Test that factory passes through options."""
        recorder = EventRecorder()
        agent = MockAgent()

        instrumented = instrument_agent(agent, recorder, instrument_run=False)
        instrumented.run("test")

        # Should have no events since instrument_run=False
        assert len(recorder) == 0

    def test_factory_custom_tools_attr(self) -> None:
        """Test factory with custom tools attribute."""
        recorder = EventRecorder()

        class CustomAgent:
            def __init__(self) -> None:
                self.actions = {"do": MockTool("do")}

            def run(self, input: str) -> str:
                return input

        agent = CustomAgent()
        instrumented = instrument_agent(
            agent, recorder, tools_attr="actions", instrument_run=False
        )

        instrumented.actions["do"]()

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].metadata["tool"] == "do"


class TestWrappedToolPreservesAttributes:
    """Test that wrapped tools preserve their original attributes."""

    def test_wrapped_tool_preserves_name(self) -> None:
        """Test that wrapped tool keeps its name attribute."""
        recorder = EventRecorder()
        tool = MockTool("calculator")
        agent = MockAgent(tools={"calc": tool})

        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)
        wrapped = instrumented.tools["calc"]

        assert wrapped.name == "calculator"

    def test_wrapped_tool_preserves_custom_attrs(self) -> None:
        """Test that wrapped tool keeps custom attributes."""
        recorder = EventRecorder()
        tool = MockTool("mytool")
        tool.description = "A useful tool"
        tool.version = "1.0"
        agent = MockAgent(tools={"t": tool})

        instrumented = InstrumentedAgent(agent, recorder, instrument_run=False)
        wrapped = instrumented.tools["t"]

        assert wrapped.description == "A useful tool"
        assert wrapped.version == "1.0"
