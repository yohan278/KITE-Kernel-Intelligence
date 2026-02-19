"""Phase attribution for prefill/decode metrics."""

from __future__ import annotations

from typing import Optional

from kite.types import EnergyTrace, PhaseSegment


def attribute_prefill_decode(
    trace: EnergyTrace,
    ttft_s: Optional[float] = None,
) -> EnergyTrace:
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

    trace.phase_segments = [
        PhaseSegment(name="prefill", start_s=t_start, end_s=split, energy_j=prefill_energy),
        PhaseSegment(name="decode", start_s=split, end_s=t_end, energy_j=decode_energy),
    ]
    return trace


def _energy_between(trace: EnergyTrace, t0: float, t1: float) -> float:
    if not trace.timestamps or not trace.energy_j or t1 <= t0:
        return 0.0

    e0 = _interp_energy(trace, t0)
    e1 = _interp_energy(trace, t1)
    return max(0.0, e1 - e0)


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
