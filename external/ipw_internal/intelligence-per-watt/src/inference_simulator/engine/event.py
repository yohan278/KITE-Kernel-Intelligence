"""Event types and priority queue for discrete-event simulation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Sequence


class EventType(str, Enum):
    """Types of events in the inference simulation."""

    REQUEST_ARRIVAL = "request_arrival"
    BATCH_SCHEDULE = "batch_schedule"
    PREFILL_COMPLETE = "prefill_complete"
    DECODE_STEP = "decode_step"
    DECODE_COMPLETE = "decode_complete"
    REQUEST_COMPLETE = "request_complete"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_COMPLETE = "tool_execution_complete"
    STEP_COMPLETE = "step_complete"


@dataclass(order=False)
class Event:
    """A simulation event with a timestamp and payload.

    Attributes:
        time_ns: Event time in nanoseconds (simulation clock).
        event_type: The type of event.
        payload: Event-specific data.
        _seq: Tie-breaker for heap ordering (set by EventQueue).
    """

    time_ns: int
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    _seq: int = field(default=0, repr=False)

    def __lt__(self, other: Event) -> bool:
        if self.time_ns != other.time_ns:
            return self.time_ns < other.time_ns
        return self._seq < other._seq

    def __le__(self, other: Event) -> bool:
        if self.time_ns != other.time_ns:
            return self.time_ns < other.time_ns
        return self._seq <= other._seq


class EventQueue:
    """Min-heap priority queue for simulation events.

    Events are ordered by time_ns, with a sequence counter as tie-breaker
    to maintain insertion order for events at the same time.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter: int = 0

    def push(self, event: Event) -> None:
        """Add an event to the queue."""
        event._seq = self._counter
        self._counter += 1
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        """Remove and return the earliest event.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._heap)

    def peek(self) -> Event:
        """Return the earliest event without removing it.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("peek on empty EventQueue")
        return self._heap[0]

    def push_many(self, events: Sequence[Event]) -> None:
        """Add multiple events to the queue."""
        for event in events:
            self.push(event)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0
