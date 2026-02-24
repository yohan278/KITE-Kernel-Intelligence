"""Telemetry session helpers for profiling runs."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Optional

from ipw.core.types import TelemetryReading
from ipw.telemetry import EnergyMonitorCollector


@dataclass
class TelemetrySample:
    timestamp: float
    reading: TelemetryReading


class TelemetrySession(AbstractContextManager["TelemetrySession"]):
    """Capture telemetry readings in a background thread."""

    # Default timeout waiting for first telemetry sample
    DEFAULT_READY_TIMEOUT = 10.0

    def __init__(
        self,
        collector: EnergyMonitorCollector,
        *,
        buffer_seconds: float = 30.0,
        max_samples: int = 10_000,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
    ) -> None:
        self._collector = collector
        self._buffer_seconds = buffer_seconds
        self._max_samples = max_samples
        self._ready_timeout = ready_timeout
        self._samples: Deque[TelemetrySample] = deque(maxlen=max_samples)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # Signals when first sample arrives
        self._collector_ctx = None

    def __enter__(self) -> "TelemetrySession":
        self._collector_ctx = self._collector.start()
        self._collector_ctx.__enter__()
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Wait for first sample to ensure telemetry is streaming before benchmark starts
        if not self._ready_event.wait(timeout=self._ready_timeout):
            import warnings
            warnings.warn(
                f"Telemetry stream not ready after {self._ready_timeout}s. "
                "Early benchmark actions may have inaccurate energy measurements.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._collector_ctx is not None:
            self._collector_ctx.__exit__(None, None, None)

    def _run(self) -> None:
        try:
            first_sample = True
            for reading in self._collector.stream_readings():
                timestamp = (
                    float(reading.timestamp_nanos) / 1_000_000_000.0
                    if reading.timestamp_nanos is not None
                    else time.time()
                )
                self._samples.append(
                    TelemetrySample(timestamp=timestamp, reading=reading)
                )
                self._trim(timestamp)

                # Signal that telemetry is ready after first sample arrives
                if first_sample:
                    self._ready_event.set()
                    first_sample = False

                if self._stop_event.is_set():
                    break
        except Exception:  # pragma: no cover - surface to caller on access
            self._stop_event.set()
            raise

    def _trim(self, current_time: float) -> None:
        cutoff = current_time - self._buffer_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def readings(self) -> Iterable[TelemetrySample]:
        return list(self._samples)

    def window(self, start_time: float, end_time: float) -> list[TelemetrySample]:
        """Return samples within the given time window (thread-safe)."""
        return [
            sample
            for sample in list(self._samples)
            if start_time <= sample.timestamp <= end_time
        ]

    @property
    def sample_count(self) -> int:
        """Number of telemetry samples currently buffered."""
        return len(self._samples)

    @property
    def last_sample_timestamp(self) -> Optional[float]:
        """Timestamp of the most recent telemetry sample, or None if empty."""
        if self._samples:
            return self._samples[-1].timestamp
        return None

    @property
    def is_ready(self) -> bool:
        """Whether the telemetry stream has started producing samples."""
        return self._ready_event.is_set()
