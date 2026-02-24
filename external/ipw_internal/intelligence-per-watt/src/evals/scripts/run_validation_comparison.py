#!/usr/bin/env python3
"""Compare Phase 0 (baseline) vs Phase 2 (post-fix) validation results.

Loads baseline and post-fix simulation results, computes per-metric
improvement, and generates a side-by-side comparison report showing
which fixes had the biggest impact.

Usage:
    python run_validation_comparison.py \
        --baseline-dir data/validation/phase0 \
        --postfix-dir data/validation/phase2 \
        --output data/validation/phase0_vs_phase2_comparison.md
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
logger = logging.getLogger("validation_comparison")


def _find_jsonl_files(directory: Path, suffix: str = "_simulated.jsonl") -> List[Path]:
    """Find simulation result JSONL files in a directory."""
    return sorted(directory.glob(f"*{suffix}"))


def _load_results(path: Path) -> List[Dict[str, Any]]:
    """Load per-QPS results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


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


def _pct_error(real: float, sim: float) -> Optional[float]:
    """Compute percent error: (sim - real) / real * 100."""
    if real == 0:
        return None
    return (sim - real) / real * 100.0


# Metrics to compare
COMPARISON_METRICS: List[Tuple[str, str]] = [
    ("TTFT P50", "ttft.p50"),
    ("TTFT P90", "ttft.p90"),
    ("TTFT P99", "ttft.p99"),
    ("TBT P50", "tbt.p50"),
    ("TBT P90", "tbt.p90"),
    ("TBT P99", "tbt.p99"),
    ("E2E P50", "e2e.p50"),
    ("E2E P90", "e2e.p90"),
    ("E2E P99", "e2e.p99"),
    ("Throughput RPS", "throughput_rps"),
    ("Throughput TPS", "throughput_tps"),
    ("Avg Power (W)", "avg_power_w"),
]


def compute_errors_vs_real(
    real_results: List[Dict[str, Any]],
    sim_results: List[Dict[str, Any]],
) -> Dict[str, Dict[float, Optional[float]]]:
    """Compute per-metric, per-QPS percent error of sim vs real.

    Returns: {metric_path: {qps: pct_error}}
    """
    real_by_qps = {r["qps_target"]: r for r in real_results}
    sim_by_qps = {r["qps_target"]: r for r in sim_results}
    all_qps = sorted(set(list(real_by_qps.keys()) + list(sim_by_qps.keys())))

    errors: Dict[str, Dict[float, Optional[float]]] = {}
    for _, metric_path in COMPARISON_METRICS:
        errors[metric_path] = {}
        for qps in all_qps:
            real_val = _get_metric(real_by_qps.get(qps, {}), metric_path)
            sim_val = _get_metric(sim_by_qps.get(qps, {}), metric_path)
            errors[metric_path][qps] = _pct_error(real_val, sim_val)

    return errors


def build_comparison_report(
    baseline_real: List[Dict[str, Any]],
    baseline_sim: List[Dict[str, Any]],
    postfix_real: List[Dict[str, Any]],
    postfix_sim: List[Dict[str, Any]],
) -> str:
    """Build a markdown comparison report.

    Shows Phase 0 errors, Phase 2 errors, and the improvement.
    """
    baseline_errors = compute_errors_vs_real(baseline_real, baseline_sim)
    postfix_errors = compute_errors_vs_real(postfix_real, postfix_sim)

    all_qps = sorted(
        set(
            [r["qps_target"] for r in baseline_real]
            + [r["qps_target"] for r in postfix_real]
        )
    )

    lines: List[str] = []
    lines.append("# Phase 0 vs Phase 2 Validation Comparison")
    lines.append("")
    lines.append("Side-by-side comparison of simulation accuracy before and after fixes.")
    lines.append("")

    # Per-metric comparison tables
    for metric_name, metric_path in COMPARISON_METRICS:
        lines.append(f"## {metric_name}")
        lines.append("")
        lines.append("| QPS | Phase 0 % Error | Phase 2 % Error | Improvement |")
        lines.append("|----:|----------------:|----------------:|------------:|")

        for qps in all_qps:
            p0_err = baseline_errors.get(metric_path, {}).get(qps)
            p2_err = postfix_errors.get(metric_path, {}).get(qps)

            p0_str = f"{p0_err:+7.1f}%" if p0_err is not None else "N/A"
            p2_str = f"{p2_err:+7.1f}%" if p2_err is not None else "N/A"

            if p0_err is not None and p2_err is not None:
                improvement = abs(p0_err) - abs(p2_err)
                imp_str = f"{improvement:+7.1f}pp"
            else:
                imp_str = "N/A"

            lines.append(f"| {qps:5.1f} | {p0_str:>15} | {p2_str:>15} | {imp_str:>11} |")

        lines.append("")

    # Summary: mean absolute error comparison
    lines.append("## Summary: Mean Absolute % Error")
    lines.append("")
    lines.append("| Metric | Phase 0 | Phase 2 | Improvement |")
    lines.append("|--------|--------:|--------:|------------:|")

    biggest_improvements: List[Tuple[str, float]] = []

    for metric_name, metric_path in COMPARISON_METRICS:
        p0_errors = [
            abs(e) for e in baseline_errors.get(metric_path, {}).values()
            if e is not None
        ]
        p2_errors = [
            abs(e) for e in postfix_errors.get(metric_path, {}).values()
            if e is not None
        ]

        p0_mean = sum(p0_errors) / len(p0_errors) if p0_errors else 0.0
        p2_mean = sum(p2_errors) / len(p2_errors) if p2_errors else 0.0
        improvement = p0_mean - p2_mean

        biggest_improvements.append((metric_name, improvement))

        lines.append(
            f"| {metric_name} | {p0_mean:6.1f}% | {p2_mean:6.1f}% | {improvement:+6.1f}pp |"
        )

    lines.append("")

    # Biggest improvements
    lines.append("## Biggest Improvements")
    lines.append("")
    biggest_improvements.sort(key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(biggest_improvements[:5], 1):
        lines.append(f"{i}. **{name}**: {imp:+.1f} percentage points reduction in error")
    lines.append("")

    # Overall accuracy
    all_p0 = [
        abs(e)
        for errors in baseline_errors.values()
        for e in errors.values()
        if e is not None
    ]
    all_p2 = [
        abs(e)
        for errors in postfix_errors.values()
        for e in errors.values()
        if e is not None
    ]

    overall_p0 = sum(all_p0) / len(all_p0) if all_p0 else 0.0
    overall_p2 = sum(all_p2) / len(all_p2) if all_p2 else 0.0

    lines.append("## Overall Accuracy")
    lines.append("")
    lines.append(f"- Phase 0 mean absolute error: **{overall_p0:.1f}%**")
    lines.append(f"- Phase 2 mean absolute error: **{overall_p2:.1f}%**")
    lines.append(f"- Overall improvement: **{overall_p0 - overall_p2:+.1f} percentage points**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase 0 vs Phase 2 validation results"
    )
    parser.add_argument(
        "--baseline-dir", required=True,
        help="Directory containing Phase 0 (baseline) validation results",
    )
    parser.add_argument(
        "--postfix-dir", required=True,
        help="Directory containing Phase 2 (post-fix) validation results",
    )
    parser.add_argument(
        "--output", default="data/validation/phase0_vs_phase2_comparison.md",
        help="Output markdown report path",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    postfix_dir = Path(args.postfix_dir)
    output_path = Path(args.output)

    if not baseline_dir.exists():
        logger.error("Baseline directory not found: %s", baseline_dir)
        return
    if not postfix_dir.exists():
        logger.error("Post-fix directory not found: %s", postfix_dir)
        return

    # Load real and simulated results for both phases
    baseline_real_files = _find_jsonl_files(baseline_dir, "_real.jsonl")
    baseline_sim_files = _find_jsonl_files(baseline_dir, "_simulated.jsonl")
    postfix_real_files = _find_jsonl_files(postfix_dir, "_real.jsonl")
    postfix_sim_files = _find_jsonl_files(postfix_dir, "_simulated.jsonl")

    if not baseline_sim_files:
        logger.error("No simulated result files found in %s", baseline_dir)
        return
    if not postfix_sim_files:
        logger.error("No simulated result files found in %s", postfix_dir)
        return

    # Use the first matching pair for each phase
    # Real results can be shared between phases (same hardware benchmark)
    baseline_real = _load_results(baseline_real_files[0]) if baseline_real_files else []
    baseline_sim = _load_results(baseline_sim_files[0])

    # For post-fix, use its own real results if available, otherwise reuse baseline
    if postfix_real_files:
        postfix_real = _load_results(postfix_real_files[0])
    elif baseline_real:
        postfix_real = baseline_real
        logger.info("Using baseline real results for post-fix comparison")
    else:
        postfix_real = []
    postfix_sim = _load_results(postfix_sim_files[0])

    if not baseline_real and not postfix_real:
        logger.warning("No real benchmark results found; comparing simulations only")
        # Fall back to comparing simulations against each other
        baseline_real = baseline_sim
        postfix_real = baseline_sim

    logger.info("Baseline: %d real, %d simulated QPS levels", len(baseline_real), len(baseline_sim))
    logger.info("Post-fix: %d real, %d simulated QPS levels", len(postfix_real), len(postfix_sim))

    report = build_comparison_report(
        baseline_real=baseline_real,
        baseline_sim=baseline_sim,
        postfix_real=postfix_real,
        postfix_sim=postfix_sim,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info("Comparison report saved to %s", output_path)

    print(report)


if __name__ == "__main__":
    main()
