"""Metrics collector for inference simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from inference_simulator.energy.power_model import EnergyBreakdown, OperatorEvent
from inference_simulator.request.request import Request
from inference_simulator.types.operators import OperatorCategory


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregated metrics from a simulation run.

    All latency values are in seconds. Throughput is in requests/s and tokens/s.
    """

    # Time to first token percentiles (seconds)
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    # Time between tokens percentiles (seconds)
    tbt_p50: float = 0.0
    tbt_p90: float = 0.0
    tbt_p95: float = 0.0
    tbt_p99: float = 0.0

    # End-to-end latency percentiles (seconds)
    e2e_p50: float = 0.0
    e2e_p90: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0

    # Throughput
    throughput_rps: float = 0.0
    throughput_tps: float = 0.0

    # Energy
    total_energy_j: float = 0.0
    avg_power_w: float = 0.0

    # Counts
    total_requests: int = 0
    total_tokens_generated: int = 0

    # Efficiency metrics (v4)
    accuracy_score: float = 0.0
    ipw: float = 0.0
    ipj: float = 0.0
    cost_per_query_usd: float = 0.0

    # Per-step timing aggregates (v4, multi-step requests)
    total_prefill_time_s: float = 0.0
    total_decode_time_s: float = 0.0
    total_tool_time_s: float = 0.0
    avg_num_steps: float = 0.0

    # GPU utilization (fraction of time GPUs are actively computing)
    gpu_utilization: float = 0.0

    # Per-category energy breakdown (IrEne-inspired)
    energy_breakdown: Optional[EnergyBreakdown] = None


class MetricsCollector:
    """Collects per-request metrics and computes aggregate statistics."""

    def __init__(self, warmup_requests: int = 0) -> None:
        self._warmup_requests = warmup_requests
        self._completed_requests: List[Request] = []
        self._decode_step_durations_ns: List[int] = []
        self._total_energy_j: float = 0.0
        self._total_time_s: float = 0.0
        # Per-step timing accumulators (for multi-step requests)
        self._total_prefill_time_ns: int = 0
        self._total_decode_time_ns: int = 0
        self._total_tool_time_ns: int = 0
        self._total_steps: int = 0
        # GPU active time tracking
        self._gpu_active_time_ns: int = 0
        # Per-operator event tracking
        self._operator_events: List[OperatorEvent] = []
        # Energy breakdown (set externally by simulator)
        self._energy_breakdown: Optional[EnergyBreakdown] = None

    def record_request(self, request: Request) -> None:
        """Record a completed request for metrics computation."""
        self._completed_requests.append(request)

    def record_decode_step(self, duration_ns: int) -> None:
        """Record a single decode step duration for TBT calculation."""
        self._decode_step_durations_ns.append(duration_ns)

    def set_energy(self, energy_j: float) -> None:
        """Set total energy consumed during the simulation."""
        self._total_energy_j = energy_j

    def set_total_time(self, time_s: float) -> None:
        """Set total simulation wall-clock time."""
        self._total_time_s = time_s

    def record_step_timing(
        self,
        prefill_ns: int = 0,
        decode_ns: int = 0,
        tool_ns: int = 0,
    ) -> None:
        """Record timing for one step of a multi-step request."""
        self._total_prefill_time_ns += prefill_ns
        self._total_decode_time_ns += decode_ns
        self._total_tool_time_ns += tool_ns
        self._total_steps += 1

    def record_gpu_active_time(self, duration_ns: int) -> None:
        """Record GPU-active time (prefill + decode, not tool execution)."""
        self._gpu_active_time_ns += duration_ns

    def record_operator_event(
        self,
        category: OperatorCategory,
        duration_ns: int,
        batch_size: int,
        seq_len: int,
        start_time_ns: int = 0,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Record an operator execution event for energy modeling.

        Args:
            category: Operator category.
            duration_ns: Duration in nanoseconds.
            batch_size: Batch size.
            seq_len: Sequence length.
            start_time_ns: Event start time in nanoseconds.
            layer_idx: Optional layer index for per-layer breakdown.
        """
        self._operator_events.append(
            OperatorEvent(
                category=category,
                duration_ns=duration_ns,
                batch_size=batch_size,
                seq_len=seq_len,
                start_time_ns=start_time_ns,
                layer_idx=layer_idx,
            )
        )

    def set_energy_breakdown(self, breakdown: EnergyBreakdown) -> None:
        """Store the energy breakdown for inclusion in SimulationMetrics."""
        self._energy_breakdown = breakdown

    @property
    def operator_events(self) -> List[OperatorEvent]:
        """Return the list of recorded operator events."""
        return self._operator_events

    def compute(
        self,
        accuracy_score: float = 0.0,
        price_per_gpu_hour_usd: float = 0.0,
        num_gpus: int = 1,
    ) -> SimulationMetrics:
        """Compute aggregate metrics from all recorded requests.

        Args:
            accuracy_score: Model accuracy score for IPW/IPJ calculation.
            price_per_gpu_hour_usd: Cost per GPU-hour for cost calculation.
            num_gpus: Number of GPUs used.

        Returns:
            SimulationMetrics with percentiles, throughput, energy, and efficiency.
        """
        if not self._completed_requests:
            return SimulationMetrics()

        # Exclude warmup and drain requests for steady-state metrics
        # Require at least 3*warmup completed so that after stripping warmup
        # from both ends, at least warmup requests remain in steady-state.
        if (
            self._warmup_requests > 0
            and len(self._completed_requests) >= 3 * self._warmup_requests
        ):
            steady_requests = self._completed_requests[
                self._warmup_requests : -self._warmup_requests
            ]
        else:
            steady_requests = self._completed_requests

        # Collect TTFT values from steady-state requests
        ttft_values_ns = []
        for req in steady_requests:
            ttft = req.ttft_ns
            if ttft is not None:
                ttft_values_ns.append(ttft)

        # Collect E2E latency values from steady-state requests
        e2e_values_ns = []
        for req in steady_requests:
            e2e = req.e2e_latency_ns
            if e2e is not None:
                e2e_values_ns.append(e2e)

        # Compute TBT from recorded decode step durations
        tbt_values_ns = self._decode_step_durations_ns

        # Convert to seconds for output
        ttft_s = np.array(ttft_values_ns, dtype=np.float64) / 1e9 if ttft_values_ns else np.array([])
        e2e_s = np.array(e2e_values_ns, dtype=np.float64) / 1e9 if e2e_values_ns else np.array([])
        tbt_s = np.array(tbt_values_ns, dtype=np.float64) / 1e9 if tbt_values_ns else np.array([])

        # Throughput from steady-state requests
        total_tokens = sum(r.tokens_generated for r in steady_requests)
        num_requests = len(steady_requests)

        # Compute steady-state time window from request timestamps
        # (avoids dividing steady-state tokens by total sim time, which
        # inflates throughput when warmup strips most requests)
        if len(steady_requests) >= 2:
            first_arrival_ns = steady_requests[0].arrival_time_ns
            completions = [
                r.completion_ns
                for r in steady_requests
                if r.completion_ns is not None
            ]
            last_completion_ns = max(completions) if completions else None
            if (
                last_completion_ns is not None
                and last_completion_ns > first_arrival_ns
            ):
                steady_duration_s = (last_completion_ns - first_arrival_ns) / 1e9
            else:
                steady_duration_s = self._total_time_s
        else:
            steady_duration_s = self._total_time_s

        if steady_duration_s > 0:
            throughput_rps = num_requests / steady_duration_s
            throughput_tps = total_tokens / steady_duration_s
        else:
            throughput_rps = 0.0
            throughput_tps = 0.0

        avg_power_w = 0.0
        if self._total_time_s > 0 and self._total_energy_j > 0:
            avg_power_w = self._total_energy_j / self._total_time_s

        # IPW: Intelligence Per Watt = accuracy_score / avg_power_w
        ipw = 0.0
        if avg_power_w > 0 and accuracy_score > 0:
            ipw = accuracy_score / avg_power_w

        # IPJ: Intelligence Per Joule = accuracy_score / energy_per_query
        ipj = 0.0
        if self._total_energy_j > 0 and num_requests > 0 and accuracy_score > 0:
            energy_per_query = self._total_energy_j / num_requests
            ipj = accuracy_score / energy_per_query

        # Cost per query (use steady-state duration for consistency)
        cost_per_query_usd = 0.0
        if price_per_gpu_hour_usd > 0 and steady_duration_s > 0 and num_requests > 0:
            total_cost = price_per_gpu_hour_usd * num_gpus * (steady_duration_s / 3600.0)
            cost_per_query_usd = total_cost / num_requests

        # Per-step timing
        total_prefill_s = self._total_prefill_time_ns / 1e9
        total_decode_s = self._total_decode_time_ns / 1e9
        total_tool_s = self._total_tool_time_ns / 1e9
        # Count steps only from steady-state completed requests (not global counter)
        steady_steps = sum(max(len(r.step_prefill_times_ns), 1) for r in steady_requests)
        avg_num_steps = steady_steps / num_requests if num_requests > 0 else 0.0

        # GPU utilization
        gpu_utilization = 0.0
        if self._total_time_s > 0:
            total_time_ns = int(self._total_time_s * 1e9)
            if total_time_ns > 0:
                gpu_utilization = min(self._gpu_active_time_ns / total_time_ns, 1.0)

        return SimulationMetrics(
            ttft_p50=_percentile(ttft_s, 50),
            ttft_p90=_percentile(ttft_s, 90),
            ttft_p95=_percentile(ttft_s, 95),
            ttft_p99=_percentile(ttft_s, 99),
            tbt_p50=_percentile(tbt_s, 50),
            tbt_p90=_percentile(tbt_s, 90),
            tbt_p95=_percentile(tbt_s, 95),
            tbt_p99=_percentile(tbt_s, 99),
            e2e_p50=_percentile(e2e_s, 50),
            e2e_p90=_percentile(e2e_s, 90),
            e2e_p95=_percentile(e2e_s, 95),
            e2e_p99=_percentile(e2e_s, 99),
            throughput_rps=throughput_rps,
            throughput_tps=throughput_tps,
            total_energy_j=self._total_energy_j,
            avg_power_w=avg_power_w,
            total_requests=len(self._completed_requests),
            total_tokens_generated=sum(
                r.tokens_generated for r in self._completed_requests
            ),
            accuracy_score=accuracy_score,
            ipw=ipw,
            ipj=ipj,
            cost_per_query_usd=cost_per_query_usd,
            total_prefill_time_s=total_prefill_s,
            total_decode_time_s=total_decode_s,
            total_tool_time_s=total_tool_s,
            avg_num_steps=avg_num_steps,
            gpu_utilization=gpu_utilization,
            energy_breakdown=self._energy_breakdown,
        )


def _percentile(values: np.ndarray, pct: float) -> float:
    """Compute a percentile, returning 0.0 for empty arrays."""
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, pct))
