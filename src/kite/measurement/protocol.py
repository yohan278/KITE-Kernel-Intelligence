"""Reusable measurement protocol for baseline and candidate runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import subprocess
from typing import Callable, Iterable

from kite.measurement.timing_protocol import TimedRun, timed_runs


@dataclass(slots=True)
class MeasurementConfig:
    warmup_iters: int = 5
    measure_iters: int = 20
    repeats: int = 5
    sampling_interval_ms: float = 50.0
    device_index: int = 0
    fixed_power_cap_w: int | None = None
    fixed_clock_profile: str | None = None


@dataclass(slots=True)
class MeasurementResult:
    runtime_ms_mean: float
    runtime_ms_std: float
    runtime_ms_cv: float
    avg_power_w_mean: float
    avg_power_w_std: float
    energy_j_mean: float
    energy_j_std: float
    energy_j_cv: float
    repeats: int
    runs: list[TimedRun]


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _std(values: Iterable[float], mean: float) -> float:
    rows = list(values)
    if len(rows) <= 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in rows) / (len(rows) - 1)
    return math.sqrt(max(0.0, var))


class MeasurementProtocol:
    def __init__(self, config: MeasurementConfig | None = None) -> None:
        self.config = config or MeasurementConfig()

    def measure(self, fn: Callable[[], object]) -> MeasurementResult:
        self._maybe_apply_runtime_controls()
        runs: list[TimedRun] = []
        for _ in range(max(1, self.config.repeats)):
            runs.append(
                timed_runs(
                    fn,
                    warmup_iters=self.config.warmup_iters,
                    measure_iters=self.config.measure_iters,
                    sampling_interval_ms=self.config.sampling_interval_ms,
                    device_index=self.config.device_index,
                )
            )

        runtimes = [r.runtime_ms for r in runs]
        powers = [r.window.avg_power_w for r in runs]
        energies = [r.window.energy_j for r in runs]

        runtime_mean = _mean(runtimes)
        runtime_std = _std(runtimes, runtime_mean)
        runtime_cv = runtime_std / runtime_mean if runtime_mean > 0 else 0.0

        power_mean = _mean(powers)
        power_std = _std(powers, power_mean)

        energy_mean = _mean(energies)
        energy_std = _std(energies, energy_mean)
        energy_cv = energy_std / energy_mean if energy_mean > 0 else 0.0

        return MeasurementResult(
            runtime_ms_mean=runtime_mean,
            runtime_ms_std=runtime_std,
            runtime_ms_cv=runtime_cv,
            avg_power_w_mean=power_mean,
            avg_power_w_std=power_std,
            energy_j_mean=energy_mean,
            energy_j_std=energy_std,
            energy_j_cv=energy_cv,
            repeats=len(runs),
            runs=runs,
        )

    def _maybe_apply_runtime_controls(self) -> None:
        if self.config.fixed_power_cap_w is None:
            return
        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.config.device_index),
                    "-pl",
                    str(self.config.fixed_power_cap_w),
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return
