"""Energy integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kite.measurement.nvml_power import GpuSample, PowerSample


@dataclass(slots=True)
class EnergyWindow:
    duration_s: float
    avg_power_w: float
    energy_j: float
    num_samples: int


@dataclass(slots=True)
class RichEnergyWindow:
    """Energy window with all GPU telemetry signals averaged."""
    duration_s: float
    avg_power_w: float
    energy_j: float
    num_samples: int
    avg_gpu_util_pct: float = 0.0
    avg_mem_util_pct: float = 0.0
    avg_temp_c: float = 0.0
    avg_sm_clock_mhz: float = 0.0
    avg_mem_clock_mhz: float = 0.0
    avg_mem_used_mb: float = 0.0


def integrate_energy(samples: Iterable[PowerSample]) -> EnergyWindow:
    rows = list(samples)
    if len(rows) == 0:
        return EnergyWindow(duration_s=0.0, avg_power_w=0.0, energy_j=0.0, num_samples=0)
    if len(rows) == 1:
        return EnergyWindow(
            duration_s=0.0,
            avg_power_w=rows[0].power_w,
            energy_j=0.0,
            num_samples=1,
        )

    energy = 0.0
    total_dt = 0.0
    for prev, nxt in zip(rows[:-1], rows[1:]):
        dt = max(0.0, nxt.timestamp_s - prev.timestamp_s)
        total_dt += dt
        energy += 0.5 * (prev.power_w + nxt.power_w) * dt

    avg_power = energy / total_dt if total_dt > 0 else (sum(r.power_w for r in rows) / len(rows))
    return EnergyWindow(
        duration_s=total_dt,
        avg_power_w=avg_power,
        energy_j=energy,
        num_samples=len(rows),
    )


def integrate_rich_energy(samples: Iterable[GpuSample]) -> RichEnergyWindow:
    """Integrate a sequence of GpuSample into a RichEnergyWindow."""
    rows = list(samples)
    n = len(rows)
    if n == 0:
        return RichEnergyWindow(duration_s=0.0, avg_power_w=0.0, energy_j=0.0, num_samples=0)

    def _mean(attr: str) -> float:
        vals = [getattr(r, attr, 0.0) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    if n == 1:
        r = rows[0]
        return RichEnergyWindow(
            duration_s=0.0,
            avg_power_w=r.power_w,
            energy_j=0.0,
            num_samples=1,
            avg_gpu_util_pct=r.gpu_util_pct,
            avg_mem_util_pct=r.mem_util_pct,
            avg_temp_c=r.temp_c,
            avg_sm_clock_mhz=r.sm_clock_mhz,
            avg_mem_clock_mhz=r.mem_clock_mhz,
            avg_mem_used_mb=r.mem_used_mb,
        )

    energy = 0.0
    total_dt = 0.0
    for prev, nxt in zip(rows[:-1], rows[1:]):
        dt = max(0.0, nxt.timestamp_s - prev.timestamp_s)
        total_dt += dt
        energy += 0.5 * (prev.power_w + nxt.power_w) * dt

    avg_power = energy / total_dt if total_dt > 0 else _mean("power_w")
    return RichEnergyWindow(
        duration_s=total_dt,
        avg_power_w=avg_power,
        energy_j=energy,
        num_samples=n,
        avg_gpu_util_pct=_mean("gpu_util_pct"),
        avg_mem_util_pct=_mean("mem_util_pct"),
        avg_temp_c=_mean("temp_c"),
        avg_sm_clock_mhz=_mean("sm_clock_mhz"),
        avg_mem_clock_mhz=_mean("mem_clock_mhz"),
        avg_mem_used_mb=_mean("mem_used_mb"),
    )

