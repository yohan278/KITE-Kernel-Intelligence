"""Energy capture and ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from kite.adapters.ipw_adapter import IPWAdapter
from kite.types import EnergyTrace
from kite.utils.serialization import load_json, save_json


class EnergyCapture:
    def __init__(self) -> None:
        self.adapter = IPWAdapter()

    def load_trace(self, path: Path) -> EnergyTrace:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Trace file must be a JSON object: {path}")
        return self.adapter.parse_trace(payload)

    def save_trace(self, path: Path, payload: Mapping[str, Any]) -> None:
        save_json(path, dict(payload))

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
