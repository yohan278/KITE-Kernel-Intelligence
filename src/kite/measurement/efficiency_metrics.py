"""Derived energy-efficiency metrics from rich GPU telemetry.

These metrics capture the *character* of energy usage, enabling analysis
like: "two kernels at the same runtime but one uses 60% more energy because
it's memory-bound (89% mem_bw_util) while the other is compute-bound (82%
SM util)."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kite.measurement.energy_integrate import RichEnergyWindow


@dataclass(slots=True)
class EfficiencyMetrics:
    """Derived metrics from a RichEnergyWindow + runtime."""
    energy_per_ms: Optional[float] = None
    power_efficiency_ms_per_j: Optional[float] = None
    avg_sm_util: float = 0.0
    avg_mem_bw_util: float = 0.0
    compute_to_mem_ratio: Optional[float] = None
    thermal_headroom_c: Optional[float] = None
    clock_efficiency: Optional[float] = None


def compute_efficiency_metrics(
    window: RichEnergyWindow,
    runtime_ms: Optional[float] = None,
    max_temp_c: float = 83.0,
    max_sm_clock_mhz: float = 2520.0,
) -> EfficiencyMetrics:
    """Compute derived efficiency metrics.

    Args:
        window: Rich energy window from NvmlRichSampler integration.
        runtime_ms: Kernel runtime in milliseconds (from KernelBench eval).
        max_temp_c: GPU thermal limit for headroom calculation (L40S default).
        max_sm_clock_mhz: Max SM clock for clock efficiency (L40S default).
    """
    energy_per_ms = None
    power_eff = None
    if runtime_ms is not None and runtime_ms > 0 and window.energy_j > 0:
        energy_per_ms = window.energy_j / runtime_ms
        power_eff = runtime_ms / window.energy_j

    compute_to_mem = None
    if window.avg_mem_util_pct > 0:
        compute_to_mem = window.avg_gpu_util_pct / window.avg_mem_util_pct

    thermal_headroom = None
    if window.avg_temp_c > 0:
        thermal_headroom = max_temp_c - window.avg_temp_c

    clock_eff = None
    if max_sm_clock_mhz > 0 and window.avg_sm_clock_mhz > 0:
        clock_eff = window.avg_sm_clock_mhz / max_sm_clock_mhz

    return EfficiencyMetrics(
        energy_per_ms=energy_per_ms,
        power_efficiency_ms_per_j=power_eff,
        avg_sm_util=window.avg_gpu_util_pct,
        avg_mem_bw_util=window.avg_mem_util_pct,
        compute_to_mem_ratio=compute_to_mem,
        thermal_headroom_c=thermal_headroom,
        clock_efficiency=clock_eff,
    )
