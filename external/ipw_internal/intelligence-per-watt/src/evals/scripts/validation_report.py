#!/usr/bin/env python3
"""Compare real vs simulated benchmark results and generate error report.

Loads per-QPS results from run_validation_benchmark.py (real) and
run_validation_simulation.py (simulated), computes percent error per
metric per QPS level, and outputs a markdown report.

Usage:
    python validation_report.py \
        --real-path data/validation/qwen3-8b_a100_80gb_vllm_real.jsonl \
        --sim-path data/validation/qwen3-8b_a100_80gb_vllm_simulated.jsonl \
        --output data/validation/baseline_error_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation_report")


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load per-QPS results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _pct_error(real: float, sim: float) -> Optional[float]:
    """Compute percent error: (sim - real) / real * 100."""
    if real == 0:
        return None
    return (sim - real) / real * 100.0


def _get_metric(result: Dict[str, Any], metric_path: str) -> float:
    """Extract a nested metric value like 'ttft.p50'."""
    parts = metric_path.split(".")
    val: Any = result
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part, 0.0)
        else:
            return 0.0
    return float(val) if val is not None else 0.0


# Metrics to compare: (display_name, metric_path)
COMPARISON_METRICS: List[Tuple[str, str]] = [
    ("TTFT P50 (s)", "ttft.p50"),
    ("TTFT P90 (s)", "ttft.p90"),
    ("TTFT P99 (s)", "ttft.p99"),
    ("TBT P50 (s)", "tbt.p50"),
    ("TBT P90 (s)", "tbt.p90"),
    ("TBT P99 (s)", "tbt.p99"),
    ("E2E P50 (s)", "e2e.p50"),
    ("E2E P90 (s)", "e2e.p90"),
    ("E2E P99 (s)", "e2e.p99"),
    ("Throughput RPS", "throughput_rps"),
    ("Throughput TPS", "throughput_tps"),
    ("Avg Power (W)", "avg_power_w"),
]


def build_comparison_table(
    real_results: List[Dict[str, Any]],
    sim_results: List[Dict[str, Any]],
) -> str:
    """Build a markdown comparison table.

    Returns a markdown string with per-metric, per-QPS comparison.
    """
    # Match results by QPS target
    real_by_qps = {r["qps_target"]: r for r in real_results}
    sim_by_qps = {r["qps_target"]: r for r in sim_results}
    all_qps = sorted(set(list(real_by_qps.keys()) + list(sim_by_qps.keys())))

    lines: List[str] = []
    lines.append("# Validation Error Report")
    lines.append("")
    lines.append("Comparison of real vLLM serving benchmark vs EventDrivenSimulator.")
    lines.append("")

    # Per-metric tables
    for metric_name, metric_path in COMPARISON_METRICS:
        lines.append(f"## {metric_name}")
        lines.append("")
        lines.append("| QPS | Real | Simulated | % Error |")
        lines.append("|----:|-----:|----------:|--------:|")

        for qps in all_qps:
            real_val = _get_metric(real_by_qps.get(qps, {}), metric_path)
            sim_val = _get_metric(sim_by_qps.get(qps, {}), metric_path)
            pct_err = _pct_error(real_val, sim_val)

            if pct_err is not None:
                lines.append(
                    f"| {qps:5.1f} | {real_val:10.4f} | {sim_val:10.4f} | {pct_err:+7.1f}% |"
                )
            else:
                lines.append(
                    f"| {qps:5.1f} | {real_val:10.4f} | {sim_val:10.4f} | N/A |"
                )

        lines.append("")

    # Summary table: mean absolute % error per metric
    lines.append("## Summary: Mean Absolute % Error by Metric")
    lines.append("")
    lines.append("| Metric | Mean |% Error| |")
    lines.append("|--------|------:|")

    for metric_name, metric_path in COMPARISON_METRICS:
        errors = []
        for qps in all_qps:
            real_val = _get_metric(real_by_qps.get(qps, {}), metric_path)
            sim_val = _get_metric(sim_by_qps.get(qps, {}), metric_path)
            pct_err = _pct_error(real_val, sim_val)
            if pct_err is not None:
                errors.append(abs(pct_err))

        mean_err = sum(errors) / len(errors) if errors else 0.0
        lines.append(f"| {metric_name} | {mean_err:6.1f}% |")

    lines.append("")

    # Max sustainable QPS comparison
    lines.append("## Max Sustainable QPS")
    lines.append("")
    lines.append("Defined as highest QPS where TTFT P99 < 5s and E2E P99 < 30s.")
    lines.append("")

    def _max_sustainable_qps(results: List[Dict[str, Any]]) -> float:
        max_qps = 0.0
        for r in results:
            ttft_p99 = _get_metric(r, "ttft.p99")
            e2e_p99 = _get_metric(r, "e2e.p99")
            if ttft_p99 < 5.0 and e2e_p99 < 30.0:
                max_qps = max(max_qps, r["qps_target"])
        return max_qps

    real_max = _max_sustainable_qps(real_results)
    sim_max = _max_sustainable_qps(sim_results)
    pct_err = _pct_error(real_max, sim_max)

    lines.append(f"- Real: {real_max:.1f} QPS")
    lines.append(f"- Simulated: {sim_max:.1f} QPS")
    if pct_err is not None:
        lines.append(f"- Error: {pct_err:+.1f}%")
    lines.append("")

    # Energy per query comparison
    lines.append("## Energy per Query")
    lines.append("")
    lines.append("| QPS | Real (J/q) | Simulated (J/q) | % Error |")
    lines.append("|----:|-----------:|----------------:|--------:|")

    for qps in all_qps:
        real_r = real_by_qps.get(qps, {})
        sim_r = sim_by_qps.get(qps, {})

        real_power = _get_metric(real_r, "avg_power_w")
        real_rps = _get_metric(real_r, "throughput_rps")
        real_epq = real_power / real_rps if real_rps > 0 else 0.0

        sim_power = _get_metric(sim_r, "avg_power_w")
        sim_rps = _get_metric(sim_r, "throughput_rps")
        sim_epq = sim_power / sim_rps if sim_rps > 0 else 0.0

        # Also check for total_energy_j / total_requests in sim results
        if "total_energy_j" in sim_r and sim_r.get("total_requests", 0) > 0:
            sim_epq = sim_r["total_energy_j"] / sim_r["total_requests"]

        pct_err = _pct_error(real_epq, sim_epq)
        if pct_err is not None:
            lines.append(
                f"| {qps:5.1f} | {real_epq:10.2f} | {sim_epq:10.2f} | {pct_err:+7.1f}% |"
            )
        else:
            lines.append(
                f"| {qps:5.1f} | {real_epq:10.2f} | {sim_epq:10.2f} | N/A |"
            )

    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare real vs simulated benchmark results"
    )
    parser.add_argument(
        "--real-path", required=True,
        help="Path to real benchmark JSONL (from run_validation_benchmark.py)",
    )
    parser.add_argument(
        "--sim-path", required=True,
        help="Path to simulated JSONL (from run_validation_simulation.py)",
    )
    parser.add_argument(
        "--output", default="data/validation/baseline_error_report.md",
        help="Output markdown report path (default: data/validation/baseline_error_report.md)",
    )

    args = parser.parse_args()

    real_path = Path(args.real_path)
    sim_path = Path(args.sim_path)
    output_path = Path(args.output)

    if not real_path.exists():
        logger.error("Real results file not found: %s", real_path)
        return
    if not sim_path.exists():
        logger.error("Simulated results file not found: %s", sim_path)
        return

    logger.info("Loading real results from %s", real_path)
    real_results = load_results(real_path)
    logger.info("  Loaded %d QPS levels", len(real_results))

    logger.info("Loading simulated results from %s", sim_path)
    sim_results = load_results(sim_path)
    logger.info("  Loaded %d QPS levels", len(sim_results))

    report = build_comparison_table(real_results, sim_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", output_path)

    # Print report to stdout
    print(report)


if __name__ == "__main__":
    main()
