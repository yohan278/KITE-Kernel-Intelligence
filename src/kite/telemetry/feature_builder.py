"""Feature extraction for runtime policy state."""

from __future__ import annotations

from typing import Optional

from kite.types import EnergyTrace, RuntimeState


def build_runtime_features(
    state: RuntimeState,
    trace: Optional[EnergyTrace] = None,
) -> dict[str, float]:
    """Build a feature dict from runtime state and optional energy trace."""
    features = {
        "queue_depth": float(state.queue_depth),
        "phase_ratio": state.phase_ratio,
        "batch_size": float(state.batch_size),
        "concurrency": float(state.concurrency),
        "power_cap": float(state.power_cap),
        "ttft_p95": state.ttft_p95,
        "e2e_p95": state.e2e_p95,
        "throughput_tps": state.throughput_tps,
        "avg_power_w": state.avg_power_w,
    }

    if trace is not None:
        avg_power = sum(trace.power_w) / len(trace.power_w) if trace.power_w else 0.0
        total_energy = (trace.energy_j[-1] - trace.energy_j[0]) if len(trace.energy_j) >= 2 else 0.0
        avg_gpu_util = sum(trace.gpu_util) / len(trace.gpu_util) if trace.gpu_util else 0.0

        features["avg_power_w"] = avg_power
        features["total_energy_j"] = total_energy
        features["avg_gpu_util"] = avg_gpu_util

        for seg in trace.phase_segments:
            prefix = seg.name.lower()
            features[f"{prefix}_duration_s"] = seg.end_s - seg.start_s
            if seg.energy_j is not None:
                features[f"{prefix}_energy_j"] = seg.energy_j

    return features
