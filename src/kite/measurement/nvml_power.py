"""Power and GPU telemetry sampling helpers.

This module prefers NVML via ``pynvml``. When NVML is unavailable, it provides a
deterministic fallback sampler so measurement pipelines remain runnable.

``NvmlPowerSampler`` captures power only (backward compat).
``NvmlRichSampler`` captures power, SM utilization, memory-controller
utilization, temperature, clock frequencies, and memory usage -- the full
set of signals needed for energy-vs-compute analysis per kernel type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Optional


@dataclass(slots=True)
class PowerSample:
    timestamp_s: float
    power_w: float


@dataclass(slots=True)
class GpuSample:
    """Rich GPU telemetry sample with all NVML-accessible signals."""
    timestamp_s: float
    power_w: float
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    temp_c: float = 0.0
    sm_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    mem_used_mb: float = 0.0


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


class NvmlRichSampler:
    """Sample power + utilization + temperature + clocks + memory from NVML.

    Falls back to synthetic values when NVML is unavailable so that
    measurement pipelines remain runnable off-GPU.
    """

    def __init__(
        self,
        device_index: int = 0,
        sampling_interval_ms: float = 50.0,
        fallback_power_w: float = 300.0,
    ) -> None:
        self.device_index = device_index
        self.sampling_interval_ms = sampling_interval_ms
        self.fallback_power_w = fallback_power_w
        self.samples: list[GpuSample] = []
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

    def read_sample(self) -> GpuSample:
        """Read all available GPU signals in a single tick."""
        nvml = self._nvml
        handle = self._nvml_handle
        if nvml is None or handle is None:
            return GpuSample(timestamp_s=0.0, power_w=self.fallback_power_w)

        power_w = self.fallback_power_w
        gpu_util = 0.0
        mem_util = 0.0
        temp_c = 0.0
        sm_clock = 0.0
        mem_clock = 0.0
        mem_used_mb = 0.0

        try:
            power_w = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            pass
        try:
            rates = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(rates.gpu)
            mem_util = float(rates.memory)
        except Exception:
            pass
        try:
            temp_c = float(nvml.nvmlDeviceGetTemperature(
                handle, nvml.NVML_TEMPERATURE_GPU
            ))
        except Exception:
            pass
        try:
            sm_clock = float(nvml.nvmlDeviceGetClockInfo(
                handle, nvml.NVML_CLOCK_SM
            ))
        except Exception:
            pass
        try:
            mem_clock = float(nvml.nvmlDeviceGetClockInfo(
                handle, nvml.NVML_CLOCK_MEM
            ))
        except Exception:
            pass
        try:
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = float(mem_info.used) / (1024.0 * 1024.0)
        except Exception:
            pass

        return GpuSample(
            timestamp_s=0.0,
            power_w=power_w,
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
            temp_c=temp_c,
            sm_clock_mhz=sm_clock,
            mem_clock_mhz=mem_clock,
            mem_used_mb=mem_used_mb,
        )

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[GpuSample]:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return list(self.samples)

    def _run(self) -> None:
        sleep_s = max(0.001, self.sampling_interval_ms / 1000.0)
        start = _now()
        while not self._stop.is_set():
            sample = self.read_sample()
            sample.timestamp_s = _now() - start
            self.samples.append(sample)
            time.sleep(sleep_s)

    def read_power_w(self) -> float:
        """Convenience: read just power (compatible with NvmlPowerSampler API)."""
        s = self.read_sample()
        return s.power_w

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

