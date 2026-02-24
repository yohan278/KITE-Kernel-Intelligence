"""Tests for EventRecorder."""

import threading
import time

import pytest

from ipw.telemetry.events import AgentEvent, EventRecorder, EventType


class TestAgentEvent:
    """Tests for AgentEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test creating an event with metadata."""
        event = AgentEvent(
            event_type="tool_call_start",
            timestamp=1234567890.123,
            metadata={"tool": "calculator"},
        )
        assert event.event_type == "tool_call_start"
        assert event.timestamp == 1234567890.123
        assert event.metadata == {"tool": "calculator"}

    def test_event_default_metadata(self) -> None:
        """Test event with default empty metadata."""
        event = AgentEvent(event_type="test", timestamp=1.0)
        assert event.metadata == {}

    def test_event_repr(self) -> None:
        """Test string representation."""
        event = AgentEvent(
            event_type="test",
            timestamp=1.0,
            metadata={"key": "value"},
        )
        repr_str = repr(event)
        assert "test" in repr_str
        assert "1.000" in repr_str


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self) -> None:
        """Test that expected event types exist."""
        assert EventType.LM_INFERENCE_START == "lm_inference_start"
        assert EventType.LM_INFERENCE_END == "lm_inference_end"
        assert EventType.TOOL_CALL_START == "tool_call_start"
        assert EventType.TOOL_CALL_END == "tool_call_end"

    def test_event_types_are_strings(self) -> None:
        """Test that event types are string-compatible."""
        assert EventType.LM_INFERENCE_START.value == "lm_inference_start"
        assert EventType.TOOL_CALL_START.value == "tool_call_start"


class TestEventRecorder:
    """Tests for EventRecorder class."""

    def test_record_event(self) -> None:
        """Test basic event recording with metadata."""
        recorder = EventRecorder()
        recorder.record("lm_inference_start", tokens=100, model="llama3")

        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "lm_inference_start"
        assert events[0].metadata == {"tokens": 100, "model": "llama3"}
        assert events[0].timestamp > 0

    def test_record_multiple_events(self) -> None:
        """Test recording multiple events."""
        recorder = EventRecorder()
        recorder.record("tool_call_start", tool="calculator")
        recorder.record("tool_call_end", tool="calculator", result=42)

        events = recorder.get_events()
        assert len(events) == 2
        assert events[0].event_type == "tool_call_start"
        assert events[1].event_type == "tool_call_end"
        assert events[1].metadata["result"] == 42

    def test_clear_events(self) -> None:
        """Test clearing recorded events."""
        recorder = EventRecorder()
        recorder.record("tool_call_start", tool="calculator")
        recorder.record("tool_call_end", tool="calculator")

        assert len(recorder.get_events()) == 2
        recorder.clear()
        assert len(recorder.get_events()) == 0

    def test_len(self) -> None:
        """Test __len__ method."""
        recorder = EventRecorder()
        assert len(recorder) == 0

        recorder.record("test_event")
        assert len(recorder) == 1

        recorder.record("test_event2")
        assert len(recorder) == 2

    def test_get_events_returns_copy(self) -> None:
        """Test that get_events returns a copy, not the internal list."""
        recorder = EventRecorder()
        recorder.record("test_event")

        events1 = recorder.get_events()
        events2 = recorder.get_events()

        # Should be equal but not the same object
        assert events1 == events2
        assert events1 is not events2

        # Modifying the returned list shouldn't affect the recorder
        events1.clear()
        assert len(recorder.get_events()) == 1

    def test_thread_safety(self) -> None:
        """Test concurrent recording from multiple threads."""
        recorder = EventRecorder()
        num_threads = 10
        events_per_thread = 100

        def record_events(thread_id: int) -> None:
            for i in range(events_per_thread):
                recorder.record("test_event", thread=thread_id, index=i)

        threads = [
            threading.Thread(target=record_events, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = recorder.get_events()
        assert len(events) == num_threads * events_per_thread

        # Verify all events are present (not corrupted)
        thread_event_counts = {}
        for event in events:
            thread_id = event.metadata["thread"]
            thread_event_counts[thread_id] = thread_event_counts.get(thread_id, 0) + 1

        for thread_id in range(num_threads):
            assert thread_event_counts.get(thread_id) == events_per_thread

    def test_event_timestamps_increasing(self) -> None:
        """Test that timestamps are monotonically increasing."""
        recorder = EventRecorder()

        for _ in range(10):
            recorder.record("test_event")
            time.sleep(0.001)  # Small delay

        events = recorder.get_events()
        timestamps = [e.timestamp for e in events]

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_record_without_metadata(self) -> None:
        """Test recording event with no metadata."""
        recorder = EventRecorder()
        recorder.record("simple_event")

        events = recorder.get_events()
        assert len(events) == 1
        assert events[0].event_type == "simple_event"
        assert events[0].metadata == {}

    def test_record_with_various_metadata_types(self) -> None:
        """Test recording events with various metadata value types."""
        recorder = EventRecorder()
        recorder.record(
            "complex_event",
            string_val="test",
            int_val=42,
            float_val=3.14,
            bool_val=True,
            list_val=[1, 2, 3],
            dict_val={"nested": "value"},
            none_val=None,
        )

        events = recorder.get_events()
        assert len(events) == 1
        meta = events[0].metadata
        assert meta["string_val"] == "test"
        assert meta["int_val"] == 42
        assert meta["float_val"] == 3.14
        assert meta["bool_val"] is True
        assert meta["list_val"] == [1, 2, 3]
        assert meta["dict_val"] == {"nested": "value"}
        assert meta["none_val"] is None
