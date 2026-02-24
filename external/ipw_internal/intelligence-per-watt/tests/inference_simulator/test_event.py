"""Tests for EventQueue ordering and Event dataclass."""

from inference_simulator.engine.event import Event, EventQueue, EventType


class TestEvent:
    def test_create(self):
        e = Event(time_ns=1000, event_type=EventType.REQUEST_ARRIVAL)
        assert e.time_ns == 1000
        assert e.event_type == EventType.REQUEST_ARRIVAL
        assert e.payload == {}

    def test_with_payload(self):
        e = Event(
            time_ns=500,
            event_type=EventType.DECODE_STEP,
            payload={"batch_id": 1},
        )
        assert e.payload["batch_id"] == 1

    def test_ordering_by_time(self):
        e1 = Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL, _seq=0)
        e2 = Event(time_ns=200, event_type=EventType.BATCH_SCHEDULE, _seq=1)
        assert e1 < e2
        assert not e2 < e1

    def test_ordering_tie_break_by_seq(self):
        e1 = Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL, _seq=0)
        e2 = Event(time_ns=100, event_type=EventType.BATCH_SCHEDULE, _seq=1)
        assert e1 < e2


class TestEventType:
    def test_all_types(self):
        expected = {
            "request_arrival",
            "batch_schedule",
            "prefill_complete",
            "decode_step",
            "decode_complete",
            "request_complete",
            "tool_execution_start",
            "tool_execution_complete",
            "step_complete",
        }
        actual = {et.value for et in EventType}
        assert actual == expected


class TestEventQueue:
    def test_empty_queue(self):
        q = EventQueue()
        assert len(q) == 0
        assert not q

    def test_push_pop(self):
        q = EventQueue()
        e = Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL)
        q.push(e)
        assert len(q) == 1
        assert q
        result = q.pop()
        assert result.time_ns == 100
        assert len(q) == 0

    def test_ordering(self):
        q = EventQueue()
        q.push(Event(time_ns=300, event_type=EventType.DECODE_STEP))
        q.push(Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL))
        q.push(Event(time_ns=200, event_type=EventType.BATCH_SCHEDULE))

        assert q.pop().time_ns == 100
        assert q.pop().time_ns == 200
        assert q.pop().time_ns == 300

    def test_fifo_for_same_time(self):
        q = EventQueue()
        q.push(Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL))
        q.push(Event(time_ns=100, event_type=EventType.BATCH_SCHEDULE))
        q.push(Event(time_ns=100, event_type=EventType.DECODE_STEP))

        # Should come out in insertion order
        e1 = q.pop()
        e2 = q.pop()
        e3 = q.pop()
        assert e1._seq < e2._seq < e3._seq

    def test_peek(self):
        q = EventQueue()
        q.push(Event(time_ns=200, event_type=EventType.DECODE_STEP))
        q.push(Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL))

        peeked = q.peek()
        assert peeked.time_ns == 100
        assert len(q) == 2  # peek doesn't remove

    def test_pop_empty_raises(self):
        q = EventQueue()
        try:
            q.pop()
            assert False, "Should have raised IndexError"
        except IndexError:
            pass

    def test_peek_empty_raises(self):
        q = EventQueue()
        try:
            q.peek()
            assert False, "Should have raised IndexError"
        except IndexError:
            pass

    def test_push_many(self):
        q = EventQueue()
        events = [
            Event(time_ns=300, event_type=EventType.DECODE_STEP),
            Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL),
        ]
        q.push_many(events)
        assert len(q) == 2
        assert q.pop().time_ns == 100

    def test_bool(self):
        q = EventQueue()
        assert not q
        q.push(Event(time_ns=100, event_type=EventType.REQUEST_ARRIVAL))
        assert q
