"""Energy capture and ingestion helpers with real IPW integration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from kite.adapters.ipw_adapter import IPWAdapter
from kite.types import EnergyTrace, PhaseSegment
from kite.utils.logging import get_logger
from kite.utils.serialization import load_json, save_json

logger = get_logger(__name__)


def _ipw_available() -> bool:
    try:
        from ipw.telemetry import EnergyMonitorCollector  # type: ignore  # noqa: F401
        from ipw.execution.telemetry_session import TelemetrySession  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except ImportError:
        return False


class EnergyCapture:
    def __init__(self) -> None:
        self.adapter = IPWAdapter()

    # ---- Real GPU telemetry capture ----

    def capture_kernel_trace(
        self,
        kernel_fn: Callable,
        inputs: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
        warmup_iters: int = 3,
        measure_iters: int = 10,
        sampling_interval_ms: float = 50.0,
    ) -> EnergyTrace:
        """Run kernel_fn under energy monitoring and return an EnergyTrace.

        When the IPW telemetry stack is available on a CUDA-capable machine,
        uses the real ``EnergyMonitorCollector`` + ``TelemetrySession`` to
        capture per-sample power readings.  Falls back to nvidia-smi polling
        if IPW is not installed, and to a synthetic trace if no GPU is present.
        """
        if _ipw_available() and _torch_cuda_available():
            return self._capture_with_ipw(
                kernel_fn, inputs, warmup_iters, measure_iters, sampling_interval_ms
            )
        if _torch_cuda_available():
            return self._capture_with_nvml(
                kernel_fn, inputs, warmup_iters, measure_iters
            )
        logger.debug("No GPU; returning synthetic trace")
        return self.synthetic_trace(steps=measure_iters * 10)

    def _capture_with_ipw(
        self,
        kernel_fn: Callable,
        inputs: Any,
        warmup_iters: int,
        measure_iters: int,
        sampling_interval_ms: float,
    ) -> EnergyTrace:
        """Capture via IPW's EnergyMonitorCollector + TelemetrySession."""
        import torch  # type: ignore
        from ipw.telemetry import EnergyMonitorCollector  # type: ignore
        from ipw.execution.telemetry_session import TelemetrySession, TelemetrySample  # type: ignore

        collector = EnergyMonitorCollector()

        for _ in range(warmup_iters):
            kernel_fn(inputs)
        torch.cuda.synchronize()

        session = TelemetrySession(
            collector,
            buffer_seconds=60.0,
            max_samples=50_000,
        )
        with session:
            t_start = time.monotonic()
            for _ in range(measure_iters):
                kernel_fn(inputs)
            torch.cuda.synchronize()
            t_end = time.monotonic()
            samples = list(session.samples_between(t_start, t_end))

        return self._samples_to_trace(samples, t_start, t_end)

    def _capture_with_nvml(
        self,
        kernel_fn: Callable,
        inputs: Any,
        warmup_iters: int,
        measure_iters: int,
    ) -> EnergyTrace:
        """Fallback: poll GPU power via PyTorch/pynvml."""
        import torch  # type: ignore

        for _ in range(warmup_iters):
            kernel_fn(inputs)
        torch.cuda.synchronize()

        timestamps: list[float] = []
        power_readings: list[float] = []
        energy_cumulative: list[float] = []
        running_energy = 0.0

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            use_nvml = True
        except Exception:
            use_nvml = False

        t0 = time.monotonic()
        for i in range(measure_iters):
            kernel_fn(inputs)
            torch.cuda.synchronize()
            t_now = time.monotonic() - t0

            if use_nvml:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                except Exception:
                    power_w = 300.0
            else:
                power_w = 300.0

            if timestamps:
                dt = t_now - timestamps[-1]
                running_energy += power_w * dt

            timestamps.append(t_now)
            power_readings.append(power_w)
            energy_cumulative.append(running_energy)

        return EnergyTrace(
            timestamps=timestamps,
            power_w=power_readings,
            energy_j=energy_cumulative,
        )

    @staticmethod
    def _samples_to_trace(samples, t_start: float, t_end: float) -> EnergyTrace:
        """Convert IPW TelemetrySample list into EnergyTrace."""
        timestamps: list[float] = []
        power_w: list[float] = []
        gpu_util: list[float] = []
        temp_c: list[float] = []
        energy_j: list[float] = []
        running = 0.0

        for sample in samples:
            t = sample.timestamp - t_start
            reading = sample.reading

            p = getattr(reading, "gpu_power_w", None)
            if p is None:
                p = getattr(reading, "power_w", 300.0)
            p = float(p) if p is not None else 300.0

            if timestamps:
                dt = t - timestamps[-1]
                if dt > 0:
                    running += p * dt

            timestamps.append(t)
            power_w.append(p)
            energy_j.append(running)

            util = getattr(reading, "gpu_utilization", None)
            if util is not None:
                gpu_util.append(float(util))

            temp = getattr(reading, "temperature", None)
            if temp is not None:
                temp_c.append(float(temp))

        return EnergyTrace(
            timestamps=timestamps,
            power_w=power_w,
            energy_j=energy_j,
            gpu_util=gpu_util,
            temp_c=temp_c,
        )

    # ---- Trace ingestion from files / IPW profile output ----

    def load_trace(self, path: Path) -> EnergyTrace:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Trace file must be a JSON object: {path}")
        return self.trace_from_payload(payload)

    def trace_from_payload(self, payload: Mapping[str, Any]) -> EnergyTrace:
        if "timestamps" in payload or "power_w" in payload or "energy_j" in payload:
            return self.adapter.parse_trace(payload)

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

    def trace_from_ipw_model_metrics(self, model_metrics: Any) -> EnergyTrace:
        """Build an EnergyTrace from an IPW ModelMetrics or PhaseMetrics object."""
        energy = getattr(model_metrics, "energy_metrics", None)
        latency = getattr(model_metrics, "latency_metrics", None)
        power = getattr(model_metrics, "power_metrics", None)
        phase = getattr(model_metrics, "phase_metrics", None)

        total_energy_j = getattr(energy, "per_query_joules", None) if energy else None
        total_seconds = getattr(latency, "total_query_seconds", None) if latency else None

        gpu_power = getattr(power, "gpu", None) if power else None
        avg_power_w = getattr(gpu_power, "per_query_watts", None) if gpu_power else None
        if avg_power_w is not None and hasattr(avg_power_w, "avg"):
            avg_power_w = avg_power_w.avg

        if total_energy_j is None and avg_power_w is not None and total_seconds:
            total_energy_j = avg_power_w * total_seconds
        if total_energy_j is None:
            total_energy_j = 0.0
        if total_seconds is None or total_seconds <= 0:
            total_seconds = 1.0
        if avg_power_w is None:
            avg_power_w = total_energy_j / total_seconds

        trace = EnergyTrace(
            timestamps=[0.0, total_seconds],
            power_w=[avg_power_w, avg_power_w],
            energy_j=[0.0, total_energy_j],
        )

        if phase is not None:
            segments = []
            prefill_energy = getattr(phase, "prefill_energy_j", None)
            decode_energy = getattr(phase, "decode_energy_j", None)
            prefill_dur_ms = getattr(phase, "prefill_duration_ms", None)
            decode_dur_ms = getattr(phase, "decode_duration_ms", None)

            prefill_end = 0.0
            if prefill_dur_ms is not None:
                prefill_end = min(total_seconds, prefill_dur_ms / 1000.0)
            elif total_seconds > 0:
                prefill_end = total_seconds * 0.3

            segments.append(PhaseSegment(
                name="prefill", start_s=0.0, end_s=prefill_end, energy_j=prefill_energy,
            ))
            segments.append(PhaseSegment(
                name="decode", start_s=prefill_end, end_s=total_seconds, energy_j=decode_energy,
            ))
            trace.phase_segments = segments

        return trace

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
        """Load approximate traces from ``ipw profile`` output directory."""
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
        energy: list[float] = []
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
