"""Timing protocol with warmup and synchronized iteration boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from kite.measurement.energy_integrate import EnergyWindow, integrate_energy
from kite.measurement.nvml_power import NvmlPowerSampler, PowerSample


@dataclass(slots=True)
class TimedRun:
    runtime_ms: float
    window: EnergyWindow
    samples: list[PowerSample]


def _synchronize_if_cuda() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def timed_runs(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    measure_iters: int = 20,
    sampling_interval_ms: float = 50.0,
    device_index: int = 0,
) -> TimedRun:
    """Execute a workload under a fixed protocol and capture power samples."""
    for _ in range(max(0, warmup_iters)):
        fn()
    _synchronize_if_cuda()

    sampler = NvmlPowerSampler(
        device_index=device_index,
        sampling_interval_ms=sampling_interval_ms,
    )
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(max(1, measure_iters)):
        fn()
    _synchronize_if_cuda()
    t1 = time.perf_counter()
    samples = sampler.stop()
    duration_s = max(0.0, t1 - t0)
    if len(samples) < 2:
        power = sampler.read_power_w()
        samples = [
            PowerSample(timestamp_s=0.0, power_w=power),
            PowerSample(timestamp_s=duration_s, power_w=power),
        ]
    sampler.close()

    runtime_ms = (t1 - t0) * 1000.0 / max(1, measure_iters)
    window = integrate_energy(samples)
    return TimedRun(runtime_ms=runtime_ms, window=window, samples=samples)
