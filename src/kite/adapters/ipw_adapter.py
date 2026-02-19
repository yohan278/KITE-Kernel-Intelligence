"""IPW metric adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from kite.types import EnergyTrace, PhaseSegment


@dataclass(slots=True)
class IPWSummary:
    total_energy_j: float
    avg_power_w: float
    energy_per_output_token_j: Optional[float]
    prefill_energy_per_input_token_j: Optional[float]
    decode_energy_per_output_token_j: Optional[float]


class IPWAdapter:
    """Adapter for telemetry structures used by IPW-style measurements."""

    def parse_trace(self, payload: Mapping[str, Any]) -> EnergyTrace:
        timestamps = [float(x) for x in payload.get("timestamps", [])]
        power_w = [float(x) for x in payload.get("power_w", payload.get("power", []))]
        energy_j = [float(x) for x in payload.get("energy_j", payload.get("energy", []))]
        gpu_util = [float(x) for x in payload.get("gpu_util", [])]
        temp_c = [float(x) for x in payload.get("temp_c", [])]

        phase_segments: list[PhaseSegment] = []
        raw_segments = payload.get("phase_segments", [])
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if not isinstance(seg, Mapping):
                    continue
                phase_segments.append(
                    PhaseSegment(
                        name=str(seg.get("name", "unknown")),
                        start_s=float(seg.get("start_s", 0.0)),
                        end_s=float(seg.get("end_s", 0.0)),
                        energy_j=float(seg["energy_j"]) if seg.get("energy_j") is not None else None,
                    )
                )

        return EnergyTrace(
            timestamps=timestamps,
            power_w=power_w,
            energy_j=energy_j,
            gpu_util=gpu_util,
            temp_c=temp_c,
            phase_segments=phase_segments,
        )

    def summarize(
        self,
        trace: EnergyTrace,
        input_tokens: int,
        output_tokens: int,
    ) -> IPWSummary:
        total_energy = 0.0
        if trace.energy_j:
            total_energy = max(0.0, trace.energy_j[-1] - trace.energy_j[0])

        avg_power = sum(trace.power_w) / len(trace.power_w) if trace.power_w else 0.0

        etok = total_energy / output_tokens if output_tokens > 0 else None

        prefill_energy = None
        decode_energy = None
        for seg in trace.phase_segments:
            if seg.energy_j is None:
                continue
            if seg.name.lower() == "prefill":
                prefill_energy = seg.energy_j if prefill_energy is None else prefill_energy + seg.energy_j
            elif seg.name.lower() == "decode":
                decode_energy = seg.energy_j if decode_energy is None else decode_energy + seg.energy_j

        prefill_etok = prefill_energy / input_tokens if prefill_energy is not None and input_tokens > 0 else None
        decode_etok = decode_energy / output_tokens if decode_energy is not None and output_tokens > 0 else None

        return IPWSummary(
            total_energy_j=total_energy,
            avg_power_w=avg_power,
            energy_per_output_token_j=etok,
            prefill_energy_per_input_token_j=prefill_etok,
            decode_energy_per_output_token_j=decode_etok,
        )
