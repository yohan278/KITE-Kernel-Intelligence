"""Phase attribution for prefill/decode metrics.

When the IPW analysis stack is available, delegates to IPW's ``PhasedAnalysis``
for data from profiling runs.  For raw ``EnergyTrace`` objects (e.g. from
real-time capture), uses interpolation-based segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from kite.types import EnergyTrace, PhaseSegment
from kite.utils.logging import get_logger

logger = get_logger(__name__)


def attribute_prefill_decode(
    trace: EnergyTrace,
    ttft_s: Optional[float] = None,
) -> EnergyTrace:
    """Split an EnergyTrace into prefill and decode phase segments.

    Uses ``ttft_s`` (time-to-first-token) as the boundary between phases.
    If not provided, defaults to 30% of the total trace duration.
    """
    if not trace.timestamps:
        return trace

    t_start = trace.timestamps[0]
    t_end = trace.timestamps[-1]
    if ttft_s is None:
        split = t_start + 0.3 * max(0.0, t_end - t_start)
    else:
        split = min(max(t_start, t_start + ttft_s), t_end)

    prefill_energy = _energy_between(trace, t_start, split)
    decode_energy = _energy_between(trace, split, t_end)

    prefill_power = _avg_power_between(trace, t_start, split)
    decode_power = _avg_power_between(trace, split, t_end)

    trace.phase_segments = [
        PhaseSegment(name="prefill", start_s=t_start, end_s=split, energy_j=prefill_energy),
        PhaseSegment(name="decode", start_s=split, end_s=t_end, energy_j=decode_energy),
    ]
    return trace


def attribute_from_ipw_profile(
    results_dir: Path,
    model_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Run IPW's PhasedAnalysis on a profiling results directory.

    Returns the phased summary dict or ``None`` if IPW is not available.
    """
    try:
        from ipw.analysis.phased import PhasedAnalysis  # type: ignore
        from ipw.analysis.base import AnalysisContext, AnalysisResult  # type: ignore
    except ImportError:
        logger.debug("IPW analysis not available; skipping phased attribution")
        return None

    options: dict[str, object] = {}
    if model_name:
        options["model_name"] = model_name

    context = AnalysisContext(results_dir=results_dir, options=options)
    analysis = PhasedAnalysis()
    result: AnalysisResult = analysis.run(context)

    summary: dict[str, Any] = {}
    if hasattr(result, "artifacts"):
        for artifact in result.artifacts:
            if hasattr(artifact, "data") and isinstance(artifact.data, Mapping):
                summary.update(artifact.data)
    return summary or None


def trace_from_ipw_phased_summary(
    summary: Mapping[str, Any],
    total_duration_s: float = 1.0,
) -> EnergyTrace:
    """Build an EnergyTrace from an IPW phased summary dict."""
    prefill_energy = _safe_float(summary.get("mean_prefill_energy_j"))
    decode_energy = _safe_float(summary.get("mean_decode_energy_j"))
    prefill_dur_ms = _safe_float(summary.get("mean_prefill_duration_ms"))
    decode_dur_ms = _safe_float(summary.get("mean_decode_duration_ms"))
    prefill_power = _safe_float(summary.get("mean_prefill_power_avg_w"))
    decode_power = _safe_float(summary.get("mean_decode_power_avg_w"))

    total_energy = (prefill_energy or 0.0) + (decode_energy or 0.0)
    if prefill_dur_ms is not None and decode_dur_ms is not None:
        total_duration_s = (prefill_dur_ms + decode_dur_ms) / 1000.0
    if total_duration_s <= 0:
        total_duration_s = 1.0

    avg_power = total_energy / total_duration_s if total_duration_s > 0 else 0.0
    prefill_end = (prefill_dur_ms / 1000.0) if prefill_dur_ms else total_duration_s * 0.3

    trace = EnergyTrace(
        timestamps=[0.0, total_duration_s],
        power_w=[avg_power, avg_power],
        energy_j=[0.0, total_energy],
        phase_segments=[
            PhaseSegment(name="prefill", start_s=0.0, end_s=prefill_end, energy_j=prefill_energy),
            PhaseSegment(name="decode", start_s=prefill_end, end_s=total_duration_s, energy_j=decode_energy),
        ],
    )
    return trace


# ---- internal helpers ----

def _energy_between(trace: EnergyTrace, t0: float, t1: float) -> float:
    if not trace.timestamps or not trace.energy_j or t1 <= t0:
        return 0.0

    e0 = _interp_energy(trace, t0)
    e1 = _interp_energy(trace, t1)
    return max(0.0, e1 - e0)


def _avg_power_between(trace: EnergyTrace, t0: float, t1: float) -> float:
    if not trace.timestamps or not trace.power_w or t1 <= t0:
        return 0.0

    in_range = [
        p for t, p in zip(trace.timestamps, trace.power_w)
        if t0 <= t <= t1
    ]
    return sum(in_range) / len(in_range) if in_range else 0.0


def _interp_energy(trace: EnergyTrace, t: float) -> float:
    ts = trace.timestamps
    es = trace.energy_j
    if t <= ts[0]:
        return es[0]
    if t >= ts[-1]:
        return es[-1]

    for i in range(1, len(ts)):
        if ts[i] >= t:
            t0, t1 = ts[i - 1], ts[i]
            e0, e1 = es[i - 1], es[i]
            alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return e0 + alpha * (e1 - e0)
    return es[-1]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
