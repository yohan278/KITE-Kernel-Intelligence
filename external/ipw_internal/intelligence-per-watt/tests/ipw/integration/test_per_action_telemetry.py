"""Integration tests for per-action energy telemetry."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import pytest

from ipw.telemetry.events import AgentEvent, EventRecorder
from ipw.telemetry.correlation import (
    ActionEnergyBreakdown,
    compute_analysis,
    correlate_energy_to_events,
)


@dataclass
class MockReading:
    """Mock telemetry reading."""

    energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None


@dataclass
class MockSample:
    """Mock telemetry sample."""

    timestamp: float
    reading: MockReading


class TestPerActionTelemetryFlow:
    """Test the complete per-action telemetry flow."""

    def test_event_recorder_with_mock_agent(self) -> None:
        """Test EventRecorder captures events from agent-like execution."""
        recorder = EventRecorder()

        # Simulate agent execution pattern
        recorder.record("lm_inference_start", model="test-model")
        recorder.record("tool_call_start", tool="calculator")
        recorder.record("tool_call_end", tool="calculator")
        recorder.record("lm_inference_end", model="test-model")

        events = recorder.get_events()
        assert len(events) == 4
        assert events[0].event_type == "lm_inference_start"
        assert events[1].event_type == "tool_call_start"
        assert events[1].metadata["tool"] == "calculator"

    def test_full_correlation_flow(self) -> None:
        """Test events + samples -> breakdowns -> analysis."""
        # Simulate energy samples during execution
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(1.5, MockReading(energy_joules=125.0, cpu_energy_joules=12.5)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(2.5, MockReading(energy_joules=160.0, cpu_energy_joules=16.0)),
            MockSample(3.0, MockReading(energy_joules=170.0, cpu_energy_joules=17.0)),
        ]

        # Simulate agent events
        events = [
            AgentEvent("lm_inference_start", 1.0, {"model": "llama3"}),
            AgentEvent("tool_call_start", 2.0, {"tool": "calculator"}),
            AgentEvent("tool_call_end", 2.5, {"tool": "calculator"}),
            AgentEvent("lm_inference_end", 3.0, {"model": "llama3"}),
        ]

        # Run correlation
        breakdowns = correlate_energy_to_events(samples, events)
        analysis = compute_analysis(breakdowns)

        # Verify breakdowns (order depends on when pairs complete)
        assert len(breakdowns) == 2
        action_types = {b.action_type for b in breakdowns}
        assert action_types == {"lm_inference", "tool_call"}

        # Verify analysis
        assert analysis["action_counts"]["lm_inference"] == 1
        assert analysis["action_counts"]["tool_call"] == 1
        assert analysis["total_energy_joules"] > 0

    def test_result_structure_matches_benchmark_output(self) -> None:
        """Test that correlation output matches expected benchmark result structure."""
        breakdowns = [
            ActionEnergyBreakdown(
                action_type="lm_inference",
                step_number=0,
                gpu_energy_joules=50.0,
                cpu_energy_joules=5.0,
                total_energy_joules=55.0,
                duration_ms=1000.0,
                metadata={"model": "llama3"},
            ),
        ]
        analysis = compute_analysis(breakdowns)

        # These keys should match what bench.py expects
        assert "total_energy_joules" in analysis
        assert "action_counts" in analysis
        assert "energy_by_action" in analysis
        assert "total_gpu_energy_joules" in analysis
        assert "total_cpu_energy_joules" in analysis
        assert "total_duration_ms" in analysis

    def test_nested_tool_calls_within_inference(self) -> None:
        """Test tool calls nested within LM inference are correctly tracked."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(3.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
            MockSample(4.0, MockReading(energy_joules=200.0, cpu_energy_joules=20.0)),
            MockSample(5.0, MockReading(energy_joules=250.0, cpu_energy_joules=25.0)),
        ]

        # Nested events: inference contains a tool call
        events = [
            AgentEvent("lm_inference_start", 1.0, {"model": "llama3"}),
            AgentEvent("tool_call_start", 2.0, {"tool": "search"}),
            AgentEvent("tool_call_end", 3.0, {"tool": "search"}),
            AgentEvent("lm_inference_end", 5.0, {"model": "llama3"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        analysis = compute_analysis(breakdowns)

        assert len(breakdowns) == 2
        # Find each breakdown by type
        lm_breakdown = next(b for b in breakdowns if b.action_type == "lm_inference")
        tool_breakdown = next(b for b in breakdowns if b.action_type == "tool_call")

        # LM inference covers 1.0 to 5.0: 150J GPU, 15J CPU
        assert lm_breakdown.gpu_energy_joules == 150.0
        assert lm_breakdown.cpu_energy_joules == 15.0
        assert lm_breakdown.duration_ms == 4000.0

        # Tool call covers 2.0 to 3.0: 30J GPU, 3J CPU
        assert tool_breakdown.gpu_energy_joules == 30.0
        assert tool_breakdown.cpu_energy_joules == 3.0
        assert tool_breakdown.duration_ms == 1000.0

    def test_multiple_sequential_tool_calls(self) -> None:
        """Test multiple sequential tool calls are tracked independently."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=120.0, cpu_energy_joules=12.0)),
            MockSample(3.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(4.0, MockReading(energy_joules=170.0, cpu_energy_joules=17.0)),
        ]

        events = [
            AgentEvent("tool_call_start", 1.0, {"tool": "calculator"}),
            AgentEvent("tool_call_end", 2.0, {"tool": "calculator"}),
            AgentEvent("tool_call_start", 3.0, {"tool": "search"}),
            AgentEvent("tool_call_end", 4.0, {"tool": "search"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)

        assert len(breakdowns) == 2
        assert breakdowns[0].action_type == "tool_call"
        assert breakdowns[0].metadata["tool"] == "calculator"
        assert breakdowns[0].gpu_energy_joules == 20.0

        assert breakdowns[1].action_type == "tool_call"
        assert breakdowns[1].metadata["tool"] == "search"
        assert breakdowns[1].gpu_energy_joules == 20.0


class TestReactAgentInstrumentation:
    """Test React agent's tool instrumentation pattern."""

    def test_tool_wrapper_emits_events(self) -> None:
        """Test that tool wrappers emit start/end events."""
        recorder = EventRecorder()

        # Create a mock tool
        def mock_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        original_name = mock_tool.__name__

        # Simulate what React._instrument_tools does
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder.record("tool_call_start", tool=original_name)
            try:
                return mock_tool(*args, **kwargs)
            finally:
                recorder.record("tool_call_end", tool=original_name)

        result = wrapper(5)

        assert result == 10
        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[0].metadata["tool"] == "mock_tool"
        assert events[1].event_type == "tool_call_end"
        assert events[1].metadata["tool"] == "mock_tool"

    def test_tool_wrapper_handles_exceptions(self) -> None:
        """Test that tool wrappers emit end event even on exception."""
        recorder = EventRecorder()

        def failing_tool() -> None:
            """Tool that raises exception."""
            raise ValueError("Tool failed")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder.record("tool_call_start", tool="failing_tool")
            try:
                return failing_tool(*args, **kwargs)
            finally:
                recorder.record("tool_call_end", tool="failing_tool")

        with pytest.raises(ValueError):
            wrapper()

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[1].event_type == "tool_call_end"

    def test_multiple_tools_instrumented_correctly(self) -> None:
        """Test multiple tools can be instrumented and tracked independently."""
        recorder = EventRecorder()

        def create_wrapper(fn: Callable) -> Callable:
            """Create instrumented wrapper for a function."""
            name = fn.__name__

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder.record("tool_call_start", tool=name)
                try:
                    return fn(*args, **kwargs)
                finally:
                    recorder.record("tool_call_end", tool=name)

            return wrapper

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        wrapped_add = create_wrapper(add)
        wrapped_multiply = create_wrapper(multiply)

        # Execute both tools
        result1 = wrapped_add(2, 3)
        result2 = wrapped_multiply(4, 5)

        assert result1 == 5
        assert result2 == 20

        events = recorder.get_events()
        assert len(events) == 4
        assert events[0].metadata["tool"] == "add"
        assert events[1].metadata["tool"] == "add"
        assert events[2].metadata["tool"] == "multiply"
        assert events[3].metadata["tool"] == "multiply"


class TestEndToEndAgentSimulation:
    """End-to-end tests simulating full agent execution with energy tracking."""

    def test_simulated_agent_run_produces_valid_analysis(self) -> None:
        """Simulate a complete agent run and verify analysis output."""
        recorder = EventRecorder()

        # Simulate samples collected during agent run
        samples = [
            MockSample(0.0, MockReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            MockSample(0.5, MockReading(energy_joules=25.0, cpu_energy_joules=2.5)),
            MockSample(1.0, MockReading(energy_joules=50.0, cpu_energy_joules=5.0)),
            MockSample(1.5, MockReading(energy_joules=75.0, cpu_energy_joules=7.5)),
            MockSample(2.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.5, MockReading(energy_joules=110.0, cpu_energy_joules=11.0)),
            MockSample(3.0, MockReading(energy_joules=130.0, cpu_energy_joules=13.0)),
        ]

        # Simulate agent execution
        recorder.record("lm_inference_start", model="llama3")
        recorder.record("tool_call_start", tool="calculator")
        recorder.record("tool_call_end", tool="calculator", result=42)
        recorder.record("lm_inference_end", model="llama3")

        # Manually set timestamps for testing
        events = recorder.get_events()
        events[0] = AgentEvent("lm_inference_start", 0.0, events[0].metadata)
        events[1] = AgentEvent("tool_call_start", 1.0, events[1].metadata)
        events[2] = AgentEvent("tool_call_end", 2.0, events[2].metadata)
        events[3] = AgentEvent("lm_inference_end", 3.0, events[3].metadata)

        # Correlate and analyze
        breakdowns = correlate_energy_to_events(samples, events)
        analysis = compute_analysis(breakdowns)

        # Verify structure
        assert len(breakdowns) == 2
        assert analysis["total_energy_joules"] > 0
        assert "lm_inference" in analysis["action_counts"]
        assert "tool_call" in analysis["action_counts"]

        # Verify energy attribution makes sense
        # Tool call (1.0-2.0): 50J GPU
        # LM inference (0.0-3.0): 130J GPU
        tool_breakdown = next(b for b in breakdowns if b.action_type == "tool_call")
        assert tool_breakdown.gpu_energy_joules == 50.0

    def test_analysis_with_no_tool_calls(self) -> None:
        """Test analysis when agent uses no tools."""
        samples = [
            MockSample(0.0, MockReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
        ]

        events = [
            AgentEvent("lm_inference_start", 0.0, {"model": "llama3"}),
            AgentEvent("lm_inference_end", 1.0, {"model": "llama3"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        analysis = compute_analysis(breakdowns)

        assert len(breakdowns) == 1
        assert breakdowns[0].action_type == "lm_inference"
        assert analysis["action_counts"] == {"lm_inference": 1}
        assert analysis["energy_by_action"] == {"lm_inference": 110.0}

    def test_breakdown_serialization_for_json_output(self) -> None:
        """Test that breakdown can be serialized to dict for JSON output."""
        breakdown = ActionEnergyBreakdown(
            action_type="tool_call",
            step_number=0,
            gpu_energy_joules=25.0,
            cpu_energy_joules=2.5,
            total_energy_joules=27.5,
            duration_ms=500.0,
            metadata={"tool": "calculator", "result": 42},
        )

        # Convert to dict (simulating JSON serialization)
        from dataclasses import asdict

        breakdown_dict = asdict(breakdown)

        assert breakdown_dict["action_type"] == "tool_call"
        assert breakdown_dict["step_number"] == 0
        assert breakdown_dict["gpu_energy_joules"] == 25.0
        assert breakdown_dict["cpu_energy_joules"] == 2.5
        assert breakdown_dict["total_energy_joules"] == 27.5
        assert breakdown_dict["duration_ms"] == 500.0
        assert breakdown_dict["metadata"]["tool"] == "calculator"
