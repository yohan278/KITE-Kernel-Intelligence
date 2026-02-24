"""Tests for energy correlation."""

from dataclasses import dataclass
from typing import Any, Optional

import pytest

from ipw.telemetry.events import AgentEvent
from ipw.telemetry.correlation import (
    ActionEnergyBreakdown,
    compute_analysis,
    correlate_energy_to_events,
)


@dataclass
class MockReading:
    """Mock reading for testing."""

    energy_joules: Optional[float] = None
    cpu_energy_joules: Optional[float] = None
    power_watts: Optional[float] = None
    cpu_power_watts: Optional[float] = None
    gpu_compute_utilization_pct: Optional[float] = None
    gpu_memory_bandwidth_utilization_pct: Optional[float] = None
    gpu_tensor_core_utilization_pct: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None


@dataclass
class MockSample:
    """Mock telemetry sample for testing."""

    timestamp: float
    reading: MockReading


class TestActionEnergyBreakdown:
    """Tests for ActionEnergyBreakdown dataclass."""

    def test_breakdown_creation(self) -> None:
        """Test creating an energy breakdown."""
        breakdown = ActionEnergyBreakdown(
            action_type="lm_inference",
            step_number=0,
            gpu_energy_joules=50.0,
            cpu_energy_joules=5.0,
            total_energy_joules=55.0,
            duration_ms=1000.0,
            metadata={"model": "llama3"},
        )
        assert breakdown.action_type == "lm_inference"
        assert breakdown.total_energy_joules == 55.0
        assert breakdown.metadata == {"model": "llama3"}

    def test_breakdown_repr(self) -> None:
        """Test string representation."""
        breakdown = ActionEnergyBreakdown(
            action_type="tool_call",
            step_number=1,
            gpu_energy_joules=10.0,
            cpu_energy_joules=1.0,
            total_energy_joules=11.0,
            duration_ms=500.0,
        )
        repr_str = repr(breakdown)
        assert "tool_call" in repr_str
        assert "11.00J" in repr_str


class TestCorrelateEnergyToEvents:
    """Tests for correlate_energy_to_events function."""

    def test_correlate_energy_simple(self) -> None:
        """Test basic energy correlation with known values."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)

        assert len(breakdowns) == 1
        assert breakdowns[0].gpu_energy_joules == 50.0
        assert breakdowns[0].cpu_energy_joules == 5.0
        assert breakdowns[0].total_energy_joules == 55.0
        assert breakdowns[0].duration_ms == 1000.0

    def test_correlate_multiple_actions(self) -> None:
        """Test correlation with multiple actions."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(3.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
            AgentEvent("tool_call_start", 2.0, {"tool": "calc"}),
            AgentEvent("tool_call_end", 3.0, {"tool": "calc"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)

        assert len(breakdowns) == 2
        assert breakdowns[0].action_type == "lm_inference"
        assert breakdowns[1].action_type == "tool_call"
        assert breakdowns[1].gpu_energy_joules == 30.0

    def test_correlate_energy_missing_samples(self) -> None:
        """Test with single sample returns breakdown with zero delta."""
        samples = [MockSample(1.0, MockReading(energy_joules=100.0))]
        events = [
            AgentEvent("lm_inference_start", 0.5, {}),
            AgentEvent("lm_inference_end", 1.5, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        # Both nearest samples will be the same, delta = 0
        assert len(breakdowns) == 1
        assert breakdowns[0].gpu_energy_joules == 0.0

    def test_correlate_energy_unpaired_events(self) -> None:
        """Test that unpaired events are ignored."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            # Missing end event
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 0

    def test_correlate_energy_counter_reset(self) -> None:
        """Test handling of counter reset (negative delta)."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=50.0)),  # Reset!
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        # On counter reset, use end value
        assert breakdowns[0].gpu_energy_joules == 50.0

    def test_correlate_energy_none_values(self) -> None:
        """Test handling of None energy values."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=None, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=50.0, cpu_energy_joules=None)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        assert breakdowns[0].gpu_energy_joules == 50.0  # end - 0
        assert breakdowns[0].cpu_energy_joules == 0.0  # 0 - 10 = negative, so 0

    def test_correlate_empty_samples(self) -> None:
        """Test with empty samples list."""
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events([], events)
        assert breakdowns == []

    def test_correlate_empty_events(self) -> None:
        """Test with empty events list."""
        samples = [MockSample(1.0, MockReading(energy_joules=100.0))]

        breakdowns = correlate_energy_to_events(samples, [])
        assert breakdowns == []

    def test_correlate_both_empty(self) -> None:
        """Test with both empty inputs."""
        assert correlate_energy_to_events([], []) == []

    def test_correlate_preserves_metadata(self) -> None:
        """Test that event metadata is preserved in breakdown."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {"tool": "calculator", "input": "2+2"}),
            AgentEvent("tool_call_end", 2.0, {"tool": "calculator", "result": 4}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        # Metadata from both start and end events should be merged
        assert breakdowns[0].metadata["tool"] == "calculator"
        assert breakdowns[0].metadata["input"] == "2+2"
        assert breakdowns[0].metadata["result"] == 4

    def test_correlate_finds_nearest_samples(self) -> None:
        """Test that nearest samples are found correctly."""
        samples = [
            MockSample(0.5, MockReading(energy_joules=50.0)),
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(1.5, MockReading(energy_joules=125.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
            MockSample(2.5, MockReading(energy_joules=175.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.1, {}),  # Nearest to 1.0
            AgentEvent("lm_inference_end", 1.9, {}),  # Nearest to 2.0
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        # Should use samples at 1.0 (100J) and 2.0 (150J)
        assert breakdowns[0].gpu_energy_joules == 50.0


class TestComputeAnalysis:
    """Tests for compute_analysis function."""

    def test_compute_analysis(self) -> None:
        """Test aggregate analysis computation."""
        breakdowns = [
            ActionEnergyBreakdown("lm_inference", 0, 50.0, 5.0, 55.0, 1000.0, metadata={}),
            ActionEnergyBreakdown("tool_call", 1, 10.0, 1.0, 11.0, 500.0, metadata={}),
            ActionEnergyBreakdown("lm_inference", 2, 30.0, 3.0, 33.0, 800.0, metadata={}),
        ]

        analysis = compute_analysis(breakdowns)

        assert analysis["total_energy_joules"] == 99.0
        assert analysis["total_gpu_energy_joules"] == 90.0
        assert analysis["total_cpu_energy_joules"] == 9.0
        assert analysis["total_duration_ms"] == 2300.0
        assert analysis["action_counts"] == {"lm_inference": 2, "tool_call": 1}
        assert analysis["energy_by_action"]["lm_inference"] == 88.0
        assert analysis["energy_by_action"]["tool_call"] == 11.0

    def test_compute_analysis_empty(self) -> None:
        """Test analysis with empty breakdowns."""
        analysis = compute_analysis([])

        assert analysis["total_energy_joules"] == 0.0
        assert analysis["total_gpu_energy_joules"] == 0.0
        assert analysis["total_cpu_energy_joules"] == 0.0
        assert analysis["total_duration_ms"] == 0.0
        assert analysis["action_counts"] == {}
        assert analysis["energy_by_action"] == {}

    def test_compute_analysis_single_action(self) -> None:
        """Test analysis with single action."""
        breakdowns = [
            ActionEnergyBreakdown("lm_inference", 0, 100.0, 10.0, 110.0, 2000.0, metadata={}),
        ]

        analysis = compute_analysis(breakdowns)

        assert analysis["total_energy_joules"] == 110.0
        assert analysis["action_counts"] == {"lm_inference": 1}
        assert analysis["energy_by_action"] == {"lm_inference": 110.0}

    def test_compute_analysis_multiple_action_types(self) -> None:
        """Test analysis with many different action types."""
        breakdowns = [
            ActionEnergyBreakdown("lm_inference", 0, 50.0, 5.0, 55.0, 1000.0, metadata={}),
            ActionEnergyBreakdown("tool_call", 1, 10.0, 1.0, 11.0, 500.0, metadata={}),
            ActionEnergyBreakdown("prefill", 2, 20.0, 2.0, 22.0, 600.0, metadata={}),
            ActionEnergyBreakdown("decode", 3, 15.0, 1.5, 16.5, 700.0, metadata={}),
        ]

        analysis = compute_analysis(breakdowns)

        assert len(analysis["action_counts"]) == 4
        assert analysis["action_counts"]["lm_inference"] == 1
        assert analysis["action_counts"]["tool_call"] == 1
        assert analysis["action_counts"]["prefill"] == 1
        assert analysis["action_counts"]["decode"] == 1

        total = 55.0 + 11.0 + 22.0 + 16.5
        assert analysis["total_energy_joules"] == total


class TestIdleTimeTracking:
    """Tests for idle time tracking feature."""

    def test_include_idle_default_false(self) -> None:
        """Test that include_idle defaults to False."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(4.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
            MockSample(5.0, MockReading(energy_joules=200.0, cpu_energy_joules=20.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {"tool": "calc"}),
            AgentEvent("tool_call_end", 2.0, {"tool": "calc"}),
            # Gap from 2.0 to 4.0
            AgentEvent("tool_call_start", 4.0, {"tool": "search"}),
            AgentEvent("tool_call_end", 5.0, {"tool": "search"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        # Should only have 2 breakdowns (no idle)
        assert len(breakdowns) == 2
        assert all(b.action_type == "tool_call" for b in breakdowns)

    def test_include_idle_tracks_gaps(self) -> None:
        """Test that include_idle=True adds idle periods."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(4.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
            MockSample(5.0, MockReading(energy_joules=200.0, cpu_energy_joules=20.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {"tool": "calc"}),
            AgentEvent("tool_call_end", 2.0, {"tool": "calc"}),
            # Gap from 2.0 to 4.0
            AgentEvent("tool_call_start", 4.0, {"tool": "search"}),
            AgentEvent("tool_call_end", 5.0, {"tool": "search"}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        # Should have 3 breakdowns: tool_call, idle, tool_call
        assert len(breakdowns) == 3
        action_types = [b.action_type for b in breakdowns]
        assert "idle" in action_types

        # Find the idle breakdown
        idle_breakdown = next(b for b in breakdowns if b.action_type == "idle")
        assert idle_breakdown.duration_ms == 2000.0  # 4.0 - 2.0 = 2 seconds
        # Energy from 2.0 (150J) to 4.0 (180J) = 30J
        assert idle_breakdown.gpu_energy_joules == 30.0
        assert idle_breakdown.cpu_energy_joules == 3.0

    def test_idle_period_metadata(self) -> None:
        """Test that idle periods have correct metadata."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(3.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
            MockSample(4.0, MockReading(energy_joules=200.0, cpu_energy_joules=20.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
            AgentEvent("tool_call_start", 3.0, {}),
            AgentEvent("tool_call_end", 4.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        idle_breakdown = next(b for b in breakdowns if b.action_type == "idle")
        # Should have between_steps metadata
        assert "between_steps" in idle_breakdown.metadata
        assert idle_breakdown.metadata["between_steps"] == [0, 1]
        # Should have timestamps
        assert "start_timestamp" in idle_breakdown.metadata
        assert "end_timestamp" in idle_breakdown.metadata
        assert idle_breakdown.metadata["start_timestamp"] == 2.0
        assert idle_breakdown.metadata["end_timestamp"] == 3.0

    def test_no_idle_when_continuous(self) -> None:
        """Test no idle periods when actions are back-to-back."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(3.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
            AgentEvent("tool_call_start", 2.0, {}),  # Starts immediately after previous ends
            AgentEvent("tool_call_end", 3.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        # Should have 2 breakdowns, no idle (no gap)
        assert len(breakdowns) == 2
        assert all(b.action_type == "tool_call" for b in breakdowns)

    def test_idle_with_single_action(self) -> None:
        """Test that idle is not computed with single action."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        # Should only have 1 breakdown
        assert len(breakdowns) == 1
        assert breakdowns[0].action_type == "tool_call"

    def test_breakdowns_sorted_chronologically_with_idle(self) -> None:
        """Test that breakdowns are sorted by start timestamp when idle is included."""
        samples = [
            MockSample(0.0, MockReading(energy_joules=0.0, cpu_energy_joules=0.0)),
            MockSample(1.0, MockReading(energy_joules=50.0, cpu_energy_joules=5.0)),
            MockSample(2.0, MockReading(energy_joules=80.0, cpu_energy_joules=8.0)),
            MockSample(3.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(4.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 0.0, {}),
            AgentEvent("lm_inference_end", 1.0, {}),
            # Gap from 1.0 to 3.0
            AgentEvent("tool_call_start", 3.0, {}),
            AgentEvent("tool_call_end", 4.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)

        assert len(breakdowns) == 3
        # Should be in order: lm_inference, idle, tool_call
        assert breakdowns[0].action_type == "lm_inference"
        assert breakdowns[1].action_type == "idle"
        assert breakdowns[2].action_type == "tool_call"

    def test_idle_in_analysis(self) -> None:
        """Test that idle periods are included in analysis."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0, cpu_energy_joules=10.0)),
            MockSample(2.0, MockReading(energy_joules=150.0, cpu_energy_joules=15.0)),
            MockSample(4.0, MockReading(energy_joules=180.0, cpu_energy_joules=18.0)),
            MockSample(5.0, MockReading(energy_joules=200.0, cpu_energy_joules=20.0)),
        ]
        events = [
            AgentEvent("tool_call_start", 1.0, {}),
            AgentEvent("tool_call_end", 2.0, {}),
            AgentEvent("tool_call_start", 4.0, {}),
            AgentEvent("tool_call_end", 5.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events, include_idle=True)
        analysis = compute_analysis(breakdowns)

        assert "idle" in analysis["action_counts"]
        assert analysis["action_counts"]["idle"] == 1
        assert "idle" in analysis["energy_by_action"]
        assert analysis["energy_by_action"]["idle"] == 33.0  # 30J GPU + 3J CPU


class TestPowerSplit:
    """Tests for GPU/CPU power split in ActionEnergyBreakdown."""

    def test_power_split_populated(self) -> None:
        """Test that GPU/CPU power split fields are populated from samples."""
        samples = [
            MockSample(
                1.0,
                MockReading(
                    energy_joules=100.0,
                    cpu_energy_joules=10.0,
                    power_watts=200.0,
                    cpu_power_watts=50.0,
                ),
            ),
            MockSample(
                2.0,
                MockReading(
                    energy_joules=150.0,
                    cpu_energy_joules=15.0,
                    power_watts=250.0,
                    cpu_power_watts=60.0,
                ),
            ),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        b = breakdowns[0]

        # GPU power
        assert b.gpu_max_power_watts == 250.0
        assert b.gpu_avg_power_watts == 225.0  # (200+250)/2

        # CPU power
        assert b.cpu_max_power_watts == 60.0
        assert b.cpu_avg_power_watts == 55.0  # (50+60)/2

        # Combined (backward compat)
        assert b.max_power_watts == 310.0  # 250+60
        assert b.avg_power_watts == 280.0  # (250+310)/2 = 280

    def test_power_split_none_when_no_data(self) -> None:
        """Test that power fields are None when samples lack power data."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        b = breakdowns[0]
        assert b.gpu_max_power_watts is None
        assert b.gpu_avg_power_watts is None
        assert b.cpu_max_power_watts is None
        assert b.cpu_avg_power_watts is None

    def test_negative_power_filtered(self) -> None:
        """Test that negative power values (null collector) are filtered out."""
        samples = [
            MockSample(
                1.0,
                MockReading(
                    energy_joules=100.0,
                    power_watts=-1.0,
                    cpu_power_watts=-1.0,
                ),
            ),
            MockSample(
                2.0,
                MockReading(
                    energy_joules=150.0,
                    power_watts=-1.0,
                    cpu_power_watts=-1.0,
                ),
            ),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        b = breakdowns[0]
        assert b.gpu_max_power_watts is None
        assert b.cpu_max_power_watts is None


class TestUtilizationMetrics:
    """Tests for GPU utilization metrics in ActionEnergyBreakdown."""

    def test_utilization_populated(self) -> None:
        """Test GPU utilization fields are populated from samples."""
        samples = [
            MockSample(
                1.0,
                MockReading(
                    energy_joules=100.0,
                    gpu_compute_utilization_pct=80.0,
                    gpu_memory_bandwidth_utilization_pct=60.0,
                    gpu_tensor_core_utilization_pct=40.0,
                ),
            ),
            MockSample(
                2.0,
                MockReading(
                    energy_joules=150.0,
                    gpu_compute_utilization_pct=90.0,
                    gpu_memory_bandwidth_utilization_pct=70.0,
                    gpu_tensor_core_utilization_pct=50.0,
                ),
            ),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        b = breakdowns[0]

        assert b.gpu_compute_utilization_pct_avg == 85.0
        assert b.gpu_compute_utilization_pct_max == 90.0
        assert b.gpu_memory_bw_utilization_pct_avg == 65.0
        assert b.gpu_memory_bw_utilization_pct_max == 70.0
        assert b.gpu_tensor_core_utilization_pct_avg == 45.0
        assert b.gpu_tensor_core_utilization_pct_max == 50.0

    def test_utilization_none_when_no_data(self) -> None:
        """Test utilization fields are None when no utilization data."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        b = breakdowns[0]
        assert b.gpu_compute_utilization_pct_avg is None
        assert b.gpu_compute_utilization_pct_max is None

    def test_utilization_in_analysis(self) -> None:
        """Test utilization fields are aggregated in compute_analysis."""
        b1 = ActionEnergyBreakdown(
            "lm_inference", 0, 50.0, 5.0, 55.0, 1000.0,
            metadata={},
            gpu_compute_utilization_pct_avg=80.0,
            gpu_compute_utilization_pct_max=90.0,
            gpu_memory_bw_utilization_pct_avg=60.0,
            gpu_memory_bw_utilization_pct_max=70.0,
            gpu_tensor_core_utilization_pct_avg=40.0,
            gpu_tensor_core_utilization_pct_max=50.0,
        )
        b2 = ActionEnergyBreakdown(
            "lm_inference", 1, 30.0, 3.0, 33.0, 800.0,
            metadata={},
            gpu_compute_utilization_pct_avg=70.0,
            gpu_compute_utilization_pct_max=95.0,
            gpu_memory_bw_utilization_pct_avg=50.0,
            gpu_memory_bw_utilization_pct_max=80.0,
            gpu_tensor_core_utilization_pct_avg=30.0,
            gpu_tensor_core_utilization_pct_max=60.0,
        )

        analysis = compute_analysis([b1, b2])

        assert analysis["gpu_compute_utilization_pct_avg"] == 75.0  # (80+70)/2
        assert analysis["gpu_compute_utilization_pct_max"] == 95.0  # max(90,95)
        assert analysis["gpu_memory_bw_utilization_pct_avg"] == 55.0
        assert analysis["gpu_memory_bw_utilization_pct_max"] == 80.0
        assert analysis["gpu_tensor_core_utilization_pct_avg"] == 35.0
        assert analysis["gpu_tensor_core_utilization_pct_max"] == 60.0


class TestCostTracking:
    """Tests for dollar cost tracking in ActionEnergyBreakdown."""

    def test_cost_from_event_metadata(self) -> None:
        """Test cost_usd is extracted from end event metadata."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("submodel_call_start", 1.0, {"model_id": "gpt-4o"}),
            AgentEvent("submodel_call_end", 2.0, {"model_id": "gpt-4o", "cost_usd": 0.05}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert len(breakdowns) == 1
        assert breakdowns[0].cost_usd == 0.05

    def test_cost_none_when_not_present(self) -> None:
        """Test cost_usd is None when not in event metadata."""
        samples = [
            MockSample(1.0, MockReading(energy_joules=100.0)),
            MockSample(2.0, MockReading(energy_joules=150.0)),
        ]
        events = [
            AgentEvent("lm_inference_start", 1.0, {}),
            AgentEvent("lm_inference_end", 2.0, {}),
        ]

        breakdowns = correlate_energy_to_events(samples, events)
        assert breakdowns[0].cost_usd is None

    def test_cost_aggregated_in_analysis(self) -> None:
        """Test total_cost_usd and cost_by_model in compute_analysis."""
        b1 = ActionEnergyBreakdown(
            "submodel_call", 0, 10.0, 1.0, 11.0, 500.0,
            metadata={"model_id": "gpt-4o"},
            cost_usd=0.05,
        )
        b2 = ActionEnergyBreakdown(
            "submodel_call", 1, 10.0, 1.0, 11.0, 500.0,
            metadata={"model_id": "gpt-4o-mini"},
            cost_usd=0.01,
        )
        b3 = ActionEnergyBreakdown(
            "submodel_call", 2, 10.0, 1.0, 11.0, 500.0,
            metadata={"model_id": "gpt-4o"},
            cost_usd=0.03,
        )

        analysis = compute_analysis([b1, b2, b3])

        assert abs(analysis["total_cost_usd"] - 0.09) < 1e-9
        assert abs(analysis["cost_by_model"]["gpt-4o"] - 0.08) < 1e-9
        assert abs(analysis["cost_by_model"]["gpt-4o-mini"] - 0.01) < 1e-9

    def test_cost_zero_when_no_costs(self) -> None:
        """Test total_cost_usd is 0.0 when no breakdowns have cost."""
        b = ActionEnergyBreakdown(
            "lm_inference", 0, 50.0, 5.0, 55.0, 1000.0,
            metadata={},
        )
        analysis = compute_analysis([b])
        assert analysis["total_cost_usd"] == 0.0
        assert analysis["cost_by_model"] == {}


class TestPowerSplitInAnalysis:
    """Tests for GPU/CPU power split aggregation in compute_analysis."""

    def test_power_split_aggregated(self) -> None:
        """Test GPU/CPU power split is correctly aggregated."""
        b1 = ActionEnergyBreakdown(
            "lm_inference", 0, 50.0, 5.0, 55.0, 1000.0,
            metadata={},
            gpu_max_power_watts=250.0,
            gpu_avg_power_watts=200.0,
            cpu_max_power_watts=60.0,
            cpu_avg_power_watts=50.0,
        )
        b2 = ActionEnergyBreakdown(
            "lm_inference", 1, 30.0, 3.0, 33.0, 800.0,
            metadata={},
            gpu_max_power_watts=300.0,
            gpu_avg_power_watts=220.0,
            cpu_max_power_watts=55.0,
            cpu_avg_power_watts=45.0,
        )

        analysis = compute_analysis([b1, b2])

        assert analysis["gpu_max_power_watts"] == 300.0  # max of maxes
        assert analysis["gpu_avg_power_watts"] == 210.0  # mean of avgs
        assert analysis["cpu_max_power_watts"] == 60.0
        assert analysis["cpu_avg_power_watts"] == 47.5

    def test_power_split_none_in_empty_analysis(self) -> None:
        """Test power split fields are None for empty breakdowns."""
        analysis = compute_analysis([])
        assert analysis["gpu_max_power_watts"] is None
        assert analysis["gpu_avg_power_watts"] is None
        assert analysis["cpu_max_power_watts"] is None
        assert analysis["cpu_avg_power_watts"] is None


class TestBackwardCompat:
    """Tests for backward compatibility."""

    def test_breakdown_without_new_fields(self) -> None:
        """Test creating ActionEnergyBreakdown without new fields still works."""
        b = ActionEnergyBreakdown(
            action_type="lm_inference",
            step_number=0,
            gpu_energy_joules=50.0,
            cpu_energy_joules=5.0,
            total_energy_joules=55.0,
            duration_ms=1000.0,
        )
        assert b.gpu_max_power_watts is None
        assert b.cost_usd is None
        assert b.gpu_compute_utilization_pct_avg is None

    def test_analysis_has_all_new_keys(self) -> None:
        """Test compute_analysis returns all new keys even with old-style breakdowns."""
        b = ActionEnergyBreakdown(
            "lm_inference", 0, 50.0, 5.0, 55.0, 1000.0, metadata={}
        )
        analysis = compute_analysis([b])

        # All new keys should exist
        assert "gpu_max_power_watts" in analysis
        assert "gpu_avg_power_watts" in analysis
        assert "cpu_max_power_watts" in analysis
        assert "cpu_avg_power_watts" in analysis
        assert "gpu_compute_utilization_pct_avg" in analysis
        assert "gpu_compute_utilization_pct_max" in analysis
        assert "gpu_memory_bw_utilization_pct_avg" in analysis
        assert "gpu_memory_bw_utilization_pct_max" in analysis
        assert "gpu_tensor_core_utilization_pct_avg" in analysis
        assert "gpu_tensor_core_utilization_pct_max" in analysis
        assert "total_cost_usd" in analysis
        assert "cost_by_model" in analysis
