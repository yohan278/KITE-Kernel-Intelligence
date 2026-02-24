"""Measurement harness for GPU timing and energy telemetry."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement

logger = logging.getLogger(__name__)


class MeasurementHarness:
    """Measures operator execution time using CUDA events or wall-clock fallback.

    Uses torch.cuda.Event for GPU timing when available, falls back to
    time.perf_counter for CPU-only environments.
    """

    def __init__(
        self,
        warmup: int = 5,
        iterations: int = 20,
        use_energy: bool = False,
    ) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self.use_energy = use_energy
        self._has_cuda = self._check_cuda()

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def measure(
        self,
        op_fn: Callable[..., Any],
        *args: Any,
        operator_name: str = "unknown",
        category: OperatorCategory = OperatorCategory.LINEAR,
        batch_size: int = 1,
        seq_len: int = 1,
        flops: Optional[int] = None,
        bytes_accessed: Optional[int] = None,
        **kwargs: Any,
    ) -> OperatorMeasurement:
        """Run an operator function and measure execution time.

        Args:
            op_fn: Callable to benchmark.
            *args: Positional args passed to op_fn.
            operator_name: Name for the measurement record.
            category: Operator category.
            batch_size: Batch size for this measurement.
            seq_len: Sequence length for this measurement.
            flops: Expected FLOPs (for throughput calculation).
            bytes_accessed: Expected bytes accessed (for bandwidth calculation).
            **kwargs: Keyword args passed to op_fn.

        Returns:
            OperatorMeasurement with timing and optional energy data.
        """
        if self._has_cuda:
            time_s, energy_j, power_w = self._measure_cuda(op_fn, *args, **kwargs)
        else:
            time_s, energy_j, power_w = self._measure_cpu(op_fn, *args, **kwargs)

        bandwidth_gb_s = None
        if bytes_accessed is not None and time_s > 0:
            bandwidth_gb_s = bytes_accessed / time_s / 1e9

        return OperatorMeasurement(
            operator_name=operator_name,
            category=category,
            batch_size=batch_size,
            seq_len=seq_len,
            time_s=time_s,
            energy_j=energy_j,
            power_w=power_w,
            flops=flops,
            bytes_accessed=bytes_accessed,
            bandwidth_gb_s=bandwidth_gb_s,
        )

    def _measure_cuda(
        self, op_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[float, Optional[float], Optional[float]]:
        """Measure using CUDA events for precise GPU timing."""
        import torch

        # Warmup
        for _ in range(self.warmup):
            op_fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times_ms = []
        for _ in range(self.iterations):
            start_event.record()
            op_fn(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        mean_time_s = sum(times_ms) / len(times_ms) / 1000.0

        # Energy measurement (optional, requires telemetry infrastructure)
        energy_j = None
        power_w = None
        if self.use_energy:
            try:
                from ipw.telemetry import EnergyMonitorCollector
                from ipw.execution.telemetry_session import TelemetrySession

                collector = EnergyMonitorCollector()
                with TelemetrySession(collector) as session:
                    # Wait for telemetry to start sampling
                    time.sleep(0.3)
                    for _ in range(self.iterations):
                        op_fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    # Wait for telemetry to capture post-op readings
                    time.sleep(0.3)

                    samples = list(session.readings())
                    if len(samples) >= 2:
                        # Energy is a cumulative counter — compute delta
                        first = samples[0].reading
                        last = samples[-1].reading
                        if (
                            first.energy_joules is not None
                            and last.energy_joules is not None
                        ):
                            delta_energy = last.energy_joules - first.energy_joules
                            # Handle counter reset (negative delta)
                            if delta_energy >= 0:
                                energy_j = delta_energy / self.iterations

                        # Power is instantaneous — average across samples
                        power_readings = [
                            s.reading.power_watts
                            for s in samples
                            if s.reading.power_watts is not None
                        ]
                        if power_readings:
                            power_w = sum(power_readings) / len(power_readings)

                        # Fallback: approximate energy from average power when
                        # counter delta is zero (window too short for counter resolution)
                        if energy_j is None and power_w is not None and mean_time_s > 0:
                            energy_j = power_w * mean_time_s
            except (ImportError, RuntimeError):
                pass

        return mean_time_s, energy_j, power_w

    def _measure_cpu(
        self, op_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[float, Optional[float], Optional[float]]:
        """Fallback: measure using wall-clock time, with optional CPU energy via RAPL."""
        # Warmup
        for _ in range(self.warmup):
            op_fn(*args, **kwargs)

        # Measurement
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            op_fn(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        mean_time_s = sum(times) / len(times)

        energy_j = None
        power_w = None
        if self.use_energy:
            energy_j, power_w = self._measure_cpu_energy(
                op_fn, mean_time_s, *args, **kwargs
            )

        return mean_time_s, energy_j, power_w

    def _measure_cpu_energy(
        self,
        op_fn: Callable[..., Any],
        mean_time_s: float,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Optional[float], Optional[float]]:
        """Collect CPU energy/power via RAPL telemetry from the energy-monitor daemon."""
        try:
            from ipw.telemetry import EnergyMonitorCollector, ensure_monitor
            from ipw.execution.telemetry_session import TelemetrySession

            with ensure_monitor():
                collector = EnergyMonitorCollector()
                with TelemetrySession(collector) as session:
                    time.sleep(0.3)
                    for _ in range(self.iterations):
                        op_fn(*args, **kwargs)
                    time.sleep(0.3)

                    samples = list(session.readings())
                    if len(samples) >= 2:
                        first = samples[0].reading
                        last = samples[-1].reading

                        # CPU energy is a cumulative counter — compute delta
                        energy_j = None
                        if (
                            first.cpu_energy_joules is not None
                            and last.cpu_energy_joules is not None
                        ):
                            delta = last.cpu_energy_joules - first.cpu_energy_joules
                            if delta >= 0:
                                energy_j = delta / self.iterations

                        # CPU power is instantaneous — average across samples
                        power_readings = [
                            s.reading.cpu_power_watts
                            for s in samples
                            if s.reading.cpu_power_watts is not None
                        ]
                        power_w = (
                            sum(power_readings) / len(power_readings)
                            if power_readings
                            else None
                        )

                        # Fallback: approximate energy from average power
                        if energy_j is None and power_w is not None and mean_time_s > 0:
                            energy_j = power_w * mean_time_s

                        return energy_j, power_w
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("CPU telemetry unavailable: %s", exc)

        return None, None
