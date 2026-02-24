"""Phase-aware analysis summary for profiling runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ipw.analysis.base import AnalysisContext, AnalysisProvider, AnalysisResult
from ipw.analysis.helpers import iter_model_entries, load_metrics_dataset, resolve_model_name
from ipw.core.registry import AnalysisRegistry


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_values(
    entries: Sequence[Mapping[str, object]],
    *keys: str,
) -> list[float]:
    values: list[float] = []
    for entry in entries:
        cursor: Any = entry
        for key in keys:
            if not isinstance(cursor, Mapping):
                cursor = None
                break
            cursor = cursor.get(key)
        numeric = _to_float(cursor)
        if numeric is not None:
            values.append(numeric)
    return values


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _collect_flat_values(dataset, key: str) -> list[float]:
    values: list[float] = []
    for row in dataset:
        if not isinstance(row, Mapping):
            continue
        numeric = _to_float(row.get(key))
        if numeric is not None:
            values.append(numeric)
    return values


def _load_run_metadata(results_dir: Path) -> Mapping[str, Any]:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}
    metadata = data.get("run_metadata")
    if isinstance(metadata, Mapping):
        return metadata
    return {}


@AnalysisRegistry.register("phased")
class PhasedAnalysis(AnalysisProvider):
    """Phase-aware summary analysis."""

    analysis_id = "phased"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        results_dir = context.results_dir
        options = dict(context.options)
        requested_model = options.get("model")

        dataset = load_metrics_dataset(results_dir)
        active_model = resolve_model_name(dataset, requested_model, results_dir)

        entries = list(iter_model_entries(dataset, active_model))
        if not entries:
            raise RuntimeError(
                f"No usable metrics found for model '{active_model}' in dataset at '{results_dir}'."
            )

        prefill_energy = _collect_flat_values(dataset, "prefill_energy_j")
        decode_energy = _collect_flat_values(dataset, "decode_energy_j")
        prefill_power = _collect_flat_values(dataset, "prefill_power_avg_w")
        decode_power = _collect_flat_values(dataset, "decode_power_avg_w")
        prefill_duration = _collect_flat_values(dataset, "prefill_duration_ms")
        decode_duration = _collect_flat_values(dataset, "decode_duration_ms")
        prefill_energy_per_token = _collect_flat_values(
            dataset, "prefill_energy_per_input_token_j"
        )
        decode_energy_per_token = _collect_flat_values(
            dataset, "decode_energy_per_output_token_j"
        )

        if not any(
            (
                prefill_energy,
                decode_energy,
                prefill_power,
                decode_power,
                prefill_duration,
                decode_duration,
                prefill_energy_per_token,
                decode_energy_per_token,
            )
        ):
            phase_entries = [
                entry.get("phase_metrics")
                for entry in entries
                if isinstance(entry.get("phase_metrics"), Mapping)
            ]
            prefill_energy = _collect_values(phase_entries, "prefill_energy_j")
            decode_energy = _collect_values(phase_entries, "decode_energy_j")
            prefill_power = _collect_values(phase_entries, "prefill_power_avg_w")
            decode_power = _collect_values(phase_entries, "decode_power_avg_w")
            prefill_duration = _collect_values(phase_entries, "prefill_duration_ms")
            decode_duration = _collect_values(phase_entries, "decode_duration_ms")
            prefill_energy_per_token = _collect_values(
                phase_entries, "prefill_energy_per_input_token_j"
            )
            decode_energy_per_token = _collect_values(
                phase_entries, "decode_energy_per_output_token_j"
            )

        if not any((prefill_energy, decode_energy, prefill_power, decode_power)):
            run_metadata = _load_run_metadata(results_dir)
            phased_enabled = run_metadata.get("phased_profiling")
            warning_suffix = (
                " (phased profiling disabled)" if phased_enabled is False else ""
            )
            warning_msg = (
                "No phase-aware metrics found. Run profiling with --phased to enable "
                f"phase attribution{warning_suffix}."
            )
            artifact_dir = results_dir / "analysis"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "phased_summary.json"
            artifact_payload = {
                "analysis": self.analysis_id,
                "summary": {"total_samples": len(entries)},
                "warnings": [warning_msg],
                "data": {},
            }
            artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))
            return AnalysisResult(
                analysis=self.analysis_id,
                summary={"total_samples": len(entries)},
                warnings=(warning_msg,),
                artifacts={"report": artifact_path},
            )

        prefill_total = sum(prefill_energy) if prefill_energy else None
        decode_total = sum(decode_energy) if decode_energy else None
        combined_total = (
            (prefill_total or 0.0) + (decode_total or 0.0)
            if prefill_total is not None or decode_total is not None
            else None
        )

        prefill_fraction = (
            prefill_total / combined_total
            if prefill_total is not None
            and combined_total is not None
            and combined_total > 0.0
            else None
        )
        decode_fraction = (
            decode_total / combined_total
            if decode_total is not None
            and combined_total is not None
            and combined_total > 0.0
            else None
        )

        efficiency_ratio = None
        prefill_mean_token = _mean(prefill_energy_per_token)
        decode_mean_token = _mean(decode_energy_per_token)
        if (
            prefill_mean_token is not None
            and decode_mean_token is not None
            and prefill_mean_token > 0.0
        ):
            efficiency_ratio = decode_mean_token / prefill_mean_token

        summary_payload = {
            "total_samples": len(entries),
            "prefill_total_energy_j": prefill_total,
            "decode_total_energy_j": decode_total,
            "prefill_energy_fraction": prefill_fraction,
            "decode_energy_fraction": decode_fraction,
        }

        data_payload = {
            "prefill": {
                "total_energy_j": prefill_total,
                "energy_fraction": prefill_fraction,
                "mean_power_w": _mean(prefill_power),
                "mean_duration_ms": _mean(prefill_duration),
                "mean_energy_per_input_token_j": prefill_mean_token,
            },
            "decode": {
                "total_energy_j": decode_total,
                "energy_fraction": decode_fraction,
                "mean_power_w": _mean(decode_power),
                "mean_duration_ms": _mean(decode_duration),
                "mean_energy_per_output_token_j": decode_mean_token,
            },
            "energy_per_token_ratio": efficiency_ratio,
        }

        artifact_payload = {
            "analysis": self.analysis_id,
            "summary": summary_payload,
            "data": data_payload,
        }

        artifact_dir = results_dir / "analysis"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "phased_summary.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))

        return AnalysisResult(
            analysis=self.analysis_id,
            summary=summary_payload,
            data=data_payload,
            artifacts={"report": artifact_path},
        )


__all__ = ["PhasedAnalysis"]
