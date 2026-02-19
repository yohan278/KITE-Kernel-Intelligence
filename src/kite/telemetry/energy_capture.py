"""Energy capture and ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from kite.adapters.ipw_adapter import IPWAdapter
from kite.types import EnergyTrace, PhaseSegment
from kite.utils.serialization import load_json, save_json


class EnergyCapture:
    def __init__(self) -> None:
        self.adapter = IPWAdapter()

    def load_trace(self, path: Path) -> EnergyTrace:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Trace file must be a JSON object: {path}")
        return self.trace_from_payload(payload)

    def trace_from_payload(self, payload: Mapping[str, Any]) -> EnergyTrace:
        # Native payload format expected by IPWAdapter.
        if "timestamps" in payload or "power_w" in payload or "energy_j" in payload:
            return self.adapter.parse_trace(payload)

        # Ground-truth benchmark style payload:
        # {"power_samples": [{"timestamp_s": .., "power_w": ..}], "total_energy_j": ...}
        if "power_samples" in payload:
            power_samples = payload.get("power_samples", [])
            if isinstance(power_samples, list) and power_samples:
                timestamps: list[float] = []
                power: list[float] = []
                energy: list[float] = []
                running = 0.0
                last_t: Optional[float] = None

                for sample in power_samples:
                    if not isinstance(sample, Mapping):
                        continue
                    t = float(sample.get("timestamp_s", 0.0))
                    p = float(sample.get("power_w", 0.0))
                    if last_t is not None and t >= last_t:
                        running += p * (t - last_t)
                    timestamps.append(t)
                    power.append(p)
                    energy.append(running)
                    last_t = t

                trace = EnergyTrace(timestamps=timestamps, power_w=power, energy_j=energy)
                return self._add_phase_segments_if_available(trace, payload)

        # Compact metric payload fallback.
        total_energy_j = self._to_float(payload.get("total_energy_j")) or self._to_float(
            payload.get("energy_j")
        )
        avg_power_w = self._to_float(payload.get("avg_power_w")) or self._to_float(
            payload.get("power_avg_w")
        )
        latency_ms = self._to_float(payload.get("latency_ms"))
        if latency_ms is None:
            latency_s = self._to_float(payload.get("total_query_seconds"))
        else:
            latency_s = latency_ms / 1000.0

        if total_energy_j is None and avg_power_w is not None and latency_s is not None:
            total_energy_j = avg_power_w * latency_s

        if total_energy_j is None:
            raise ValueError("Unsupported telemetry payload format; cannot infer energy trace.")

        if latency_s is None or latency_s <= 0.0:
            latency_s = 1.0
        if avg_power_w is None:
            avg_power_w = total_energy_j / latency_s

        trace = EnergyTrace(
            timestamps=[0.0, latency_s],
            power_w=[avg_power_w, avg_power_w],
            energy_j=[0.0, total_energy_j],
        )
        return self._add_phase_segments_if_available(trace, payload)

    def save_trace(self, path: Path, payload: Mapping[str, Any]) -> None:
        save_json(path, dict(payload))

    def load_latest_trace(self, trace_dir: Path) -> Optional[EnergyTrace]:
        traces = self.load_traces_from_dir(trace_dir)
        if not traces:
            return None
        return traces[-1]

    def load_traces_from_dir(
        self,
        trace_dir: Path,
        recursive: bool = True,
    ) -> list[EnergyTrace]:
        if not trace_dir.exists():
            return []
        iterator: Iterable[Path]
        if recursive:
            iterator = trace_dir.rglob("*.json")
        else:
            iterator = trace_dir.glob("*.json")

        traces: list[EnergyTrace] = []
        for path in sorted(iterator):
            try:
                traces.append(self.load_trace(path))
            except Exception:
                continue
        return traces

    def load_traces_from_ipw_profile(
        self,
        profile_dir: Path,
        max_rows: int = 256,
    ) -> list[EnergyTrace]:
        """Load approximate traces from `ipw profile` output directory."""
        try:
            from datasets import Dataset, DatasetDict, load_from_disk  # type: ignore
        except Exception:
            return []

        if not profile_dir.exists():
            return []
        try:
            data = load_from_disk(str(profile_dir))
        except Exception:
            return []

        if isinstance(data, DatasetDict):
            if not data:
                return []
            dataset = next(iter(data.values()))
        else:
            dataset = data

        if not isinstance(dataset, Dataset):
            return []

        traces: list[EnergyTrace] = []
        for i, row in enumerate(dataset):
            if i >= max_rows:
                break
            if not isinstance(row, Mapping):
                continue
            try:
                trace = self.trace_from_payload(row)
                traces.append(trace)
            except Exception:
                continue
        return traces

    def load_trace_corpus(
        self,
        trace_dir: Optional[Path] = None,
        ipw_profile_dir: Optional[Path] = None,
        allow_synthetic_fallback: bool = True,
    ) -> list[EnergyTrace]:
        traces: list[EnergyTrace] = []
        if trace_dir is not None:
            traces.extend(self.load_traces_from_dir(trace_dir))
        if ipw_profile_dir is not None:
            traces.extend(self.load_traces_from_ipw_profile(ipw_profile_dir))
        if traces:
            return traces
        if allow_synthetic_fallback:
            return [self.synthetic_trace(steps=120)]
        return []

    def synthetic_trace(self, steps: int = 100, base_power: float = 300.0) -> EnergyTrace:
        timestamps = [i * 0.05 for i in range(steps)]
        power = [base_power + (15.0 if i % 10 < 5 else -10.0) for i in range(steps)]
        energy = []
        running = 0.0
        for i, p in enumerate(power):
            if i > 0:
                dt = timestamps[i] - timestamps[i - 1]
                running += p * dt
            energy.append(running)
        return EnergyTrace(timestamps=timestamps, power_w=power, energy_j=energy)

    def _add_phase_segments_if_available(
        self, trace: EnergyTrace, payload: Mapping[str, Any]
    ) -> EnergyTrace:
        prefill_duration_ms = self._to_float(payload.get("prefill_duration_ms"))
        prefill_energy_j = self._to_float(payload.get("prefill_energy_j"))
        decode_energy_j = self._to_float(payload.get("decode_energy_j"))

        if prefill_duration_ms is None:
            return trace

        total_time = trace.timestamps[-1] if trace.timestamps else 0.0
        prefill_end = min(total_time, max(0.0, prefill_duration_ms / 1000.0))
        decode_end = total_time

        trace.phase_segments = [
            PhaseSegment(
                name="prefill",
                start_s=0.0,
                end_s=prefill_end,
                energy_j=prefill_energy_j,
            ),
            PhaseSegment(
                name="decode",
                start_s=prefill_end,
                end_s=decode_end,
                energy_j=decode_energy_j,
            ),
        ]
        return trace

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
