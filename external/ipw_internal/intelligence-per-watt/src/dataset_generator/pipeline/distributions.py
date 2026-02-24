"""Workload distribution analysis from agent run results."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dataset_generator.pipeline.agent_runner import AgentRunResult


def _percentile(data: List[float], p: float) -> float:
    """Compute percentile using linear interpolation on sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@dataclass
class DistributionStats:
    """Summary statistics for a numeric distribution."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float


@dataclass
class WorkloadDistribution:
    """Distribution summaries for a workload type."""

    workload_type: str
    num_samples: int
    prefill_tokens: DistributionStats
    decode_tokens: DistributionStats
    num_steps: DistributionStats
    tool_calls_per_query: DistributionStats
    latency_s: DistributionStats
    tool_type_counts: Dict[str, int] = field(default_factory=dict)


def compute_distribution_stats(values: List[float]) -> DistributionStats:
    """Compute summary statistics for a list of numeric values."""
    if not values:
        zero = DistributionStats(
            mean=0.0, median=0.0, std=0.0,
            min=0.0, max=0.0,
            p50=0.0, p90=0.0, p95=0.0, p99=0.0,
        )
        return zero

    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    return DistributionStats(
        mean=mean,
        median=median,
        std=std,
        min=min(values),
        max=max(values),
        p50=_percentile(values, 50),
        p90=_percentile(values, 90),
        p95=_percentile(values, 95),
        p99=_percentile(values, 99),
    )


def compute_distributions(results: List[AgentRunResult]) -> WorkloadDistribution:
    """Compute workload distribution from a list of AgentRunResults."""
    if not results:
        empty = compute_distribution_stats([])
        return WorkloadDistribution(
            workload_type="",
            num_samples=0,
            prefill_tokens=empty,
            decode_tokens=empty,
            num_steps=empty,
            tool_calls_per_query=empty,
            latency_s=empty,
        )

    workload_type = results[0].workload_type

    prefill_vals = [float(r.prefill_tokens) for r in results]
    decode_vals = [float(r.decode_tokens) for r in results]
    steps_vals = [float(r.num_steps) for r in results]
    tool_counts_vals = [float(len(r.tool_calls)) for r in results]
    latency_vals = [r.total_latency_s for r in results]

    # Count tool types
    tool_type_counts: Dict[str, int] = {}
    for r in results:
        for tc in r.tool_calls:
            tool_type_counts[tc] = tool_type_counts.get(tc, 0) + 1

    return WorkloadDistribution(
        workload_type=workload_type,
        num_samples=len(results),
        prefill_tokens=compute_distribution_stats(prefill_vals),
        decode_tokens=compute_distribution_stats(decode_vals),
        num_steps=compute_distribution_stats(steps_vals),
        tool_calls_per_query=compute_distribution_stats(tool_counts_vals),
        latency_s=compute_distribution_stats(latency_vals),
        tool_type_counts=tool_type_counts,
    )


def distributions_to_csv(dist: WorkloadDistribution, path: Path) -> None:
    """Write a distribution summary to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p50",
        "p90",
        "p95",
        "p99",
    ]

    metrics = {
        "prefill_tokens": dist.prefill_tokens,
        "decode_tokens": dist.decode_tokens,
        "num_steps": dist.num_steps,
        "tool_calls_per_query": dist.tool_calls_per_query,
        "latency_s": dist.latency_s,
    }

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name, stats in metrics.items():
            writer.writerow({
                "metric": metric_name,
                "mean": stats.mean,
                "median": stats.median,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "p50": stats.p50,
                "p90": stats.p90,
                "p95": stats.p95,
                "p99": stats.p99,
            })
