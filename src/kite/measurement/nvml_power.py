"""Power sampling helpers.

This module prefers NVML via ``pynvml``. When NVML is unavailable, it provides a
deterministic fallback sampler so measurement pipelines remain runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional


@dataclass(slots=True)
class PowerSample:
    timestamp_s: float
    power_w: float


def _now() -> float:
    return time.perf_counter()


class NvmlPowerSampler:
    """Sample GPU power at a fixed interval."""

    def __init__(
        self,
        device_index: int = 0,
        sampling_interval_ms: float = 50.0,
        fallback_power_w: float = 300.0,
    ) -> None:
        self.device_index = device_index
        self.sampling_interval_ms = sampling_interval_ms
        self.fallback_power_w = fallback_power_w
        self.samples: list[PowerSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._nvml = None
        self._nvml_handle = None
        self._init_nvml()

    @property
    def using_nvml(self) -> bool:
        return self._nvml is not None and self._nvml_handle is not None

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml = pynvml
            self._nvml_handle = handle
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def read_power_w(self) -> float:
        if self.using_nvml:
            try:
                assert self._nvml is not None
                assert self._nvml_handle is not None
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                return float(power_mw) / 1000.0
            except Exception:
                pass
        return float(self.fallback_power_w)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return list(self.samples)

    def _run(self) -> None:
        sleep_s = max(0.001, self.sampling_interval_ms / 1000.0)
        start = _now()
        while not self._stop.is_set():
            self.samples.append(
                PowerSample(timestamp_s=_now() - start, power_w=self.read_power_w())
            )
            time.sleep(sleep_s)

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

