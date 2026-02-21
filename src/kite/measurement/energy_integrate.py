"""Energy integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kite.measurement.nvml_power import PowerSample


@dataclass(slots=True)
class EnergyWindow:
    duration_s: float
    avg_power_w: float
    energy_j: float
    num_samples: int


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

