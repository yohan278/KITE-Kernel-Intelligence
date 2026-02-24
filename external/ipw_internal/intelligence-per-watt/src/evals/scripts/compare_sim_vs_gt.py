#!/usr/bin/env python3
"""Compare simulation results against ground-truth benchmark results.

Walks both directory trees, matches (model, config, workload, qps) tuples,
and computes percent error for key metrics.

Usage:
    python compare_sim_vs_gt.py \
        --sim-dir data/e2e_v4/simulation \
        --gt-dir data/ground_truth \
        --output data/e2e_v4/reports/sim_vs_gt.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sim_vs_gt")


def load_gt_results(gt_dir: Path) -> Dict[Tuple[str, str, str, float], Dict]:
    """Load all GT results keyed by (model, config, workload, qps)."""
    results = {}
    for model_dir in sorted(gt_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "vllm_logs":
            continue
        model_key = model_dir.name
        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_id = config_dir.name
            for json_file in sorted(config_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text())
                    workload = data.get("workload", json_file.stem.rsplit("_", 1)[0])
                    qps = float(data.get("qps_target", json_file.stem.rsplit("_", 1)[-1]))
                    results[(model_key, config_id, workload, qps)] = data
                except (json.JSONDecodeError, ValueError, IndexError):
                    continue
    return results


def load_sim_results(sim_dir: Path) -> Dict[Tuple[str, str, str, float], Dict]:
    """Load all sim results keyed by (model, config, workload, qps)."""
    results = {}
    for model_dir in sorted(sim_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        for workload_dir in sorted(model_dir.iterdir()):
            if not workload_dir.is_dir():
                continue
            workload = workload_dir.name
            for jsonl_file in sorted(workload_dir.glob("*.jsonl")):
                config_id = jsonl_file.stem
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            qps = float(data.get("qps", 0))
                            results[(model_key, config_id, workload, qps)] = data
                        except (json.JSONDecodeError, ValueError):
                            continue
    return results


def pct_error(sim_val: float, gt_val: float) -> float:
    """Percent error: (sim - gt) / gt * 100."""
    if gt_val == 0:
        return float("nan")
    return (sim_val - gt_val) / gt_val * 100.0


def extract_metric(
    gt: Dict, sim: Dict, gt_path: str, sim_path: str,
) -> Optional[Tuple[float, float]]:
    """Extract a metric from both GT and sim, return (sim_val, gt_val) or None."""
    # GT: nested dict like gt['ttft']['p50']
    gt_val = gt
    for part in gt_path.split("."):
        if isinstance(gt_val, dict):
            gt_val = gt_val.get(part)
        else:
            gt_val = None
            break

    # Sim: flat metrics dict like sim['metrics']['ttft_p50']
    sim_val = sim
    for part in sim_path.split("."):
        if isinstance(sim_val, dict):
            sim_val = sim_val.get(part)
        else:
            sim_val = None
            break

    if gt_val is not None and sim_val is not None:
        try:
            return (float(sim_val), float(gt_val))
        except (ValueError, TypeError):
            pass
    return None


def _derive_energy_per_token(data: Dict, side: str) -> Optional[float]:
    """Derive energy_per_token_j for sim or gt side.

    For GT: ``energy_per_token_j`` is a direct field.
    For Sim: ``metrics.total_energy_j / metrics.total_tokens_generated``.
    """
    if side == "gt":
        val = data.get("energy_per_token_j")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return None

    # Sim side: derive from total energy / total tokens
    metrics = data.get("metrics", {})
    total_energy = metrics.get("total_energy_j")
    total_tokens = metrics.get("total_tokens_generated")
    if total_energy is not None and total_tokens is not None and total_tokens > 0:
        try:
            return float(total_energy) / int(total_tokens)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sim vs ground truth")
    parser.add_argument("--sim-dir", required=True, help="Simulation results directory")
    parser.add_argument("--gt-dir", required=True, help="Ground truth results directory")
    parser.add_argument("--output", default=None, help="Output markdown report path")
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    gt_dir = Path(args.gt_dir)

    gt_results = load_gt_results(gt_dir)
    sim_results = load_sim_results(sim_dir)

    logger.info("Loaded %d GT results, %d sim results", len(gt_results), len(sim_results))

    # Find matching keys
    common_keys = sorted(set(gt_results.keys()) & set(sim_results.keys()))
    logger.info("Matched %d (model, config, workload, qps) tuples", len(common_keys))

    if not common_keys:
        logger.warning("No matching results found!")
        # Show what's available
        gt_models = sorted(set(k[0] for k in gt_results))
        sim_models = sorted(set(k[0] for k in sim_results))
        gt_configs = sorted(set(k[1] for k in gt_results))
        sim_configs = sorted(set(k[1] for k in sim_results))
        logger.info("GT models: %s", gt_models)
        logger.info("Sim models: %s", sim_models)
        logger.info("GT configs: %s", gt_configs)
        logger.info("Sim configs: %s", sim_configs)
        return

    # Define metrics to compare (metric_name, gt_path, sim_path, is_derived)
    metrics = [
        ("TTFT p50", "ttft.p50", "metrics.ttft_p50", False),
        ("TTFT p95", "ttft.p95", "metrics.ttft_p95", False),
        ("TBT p50", "tbt.p50", "metrics.tbt_p50", False),
        ("Throughput (tok/s)", "throughput_tps", "metrics.throughput_tps", False),
        ("Avg Power (W)", "avg_power_w", "metrics.avg_power_w", False),
        ("Total Energy (J)", "total_energy_j", "metrics.total_energy_j", False),
        ("Energy/Token (J)", None, None, True),  # derived
    ]

    # Compute errors
    rows: List[Dict[str, Any]] = []
    metric_errors: Dict[str, List[float]] = {m[0]: [] for m in metrics}

    for key in common_keys:
        model, config, workload, qps = key
        gt = gt_results[key]
        sim = sim_results[key]

        row = {"model": model, "config": config, "workload": workload, "qps": qps}

        for metric_name, gt_path, sim_path, is_derived in metrics:
            if is_derived and metric_name == "Energy/Token (J)":
                # Derive energy_per_token from both sides
                sim_val = _derive_energy_per_token(sim, "sim")
                gt_val = _derive_energy_per_token(gt, "gt")
                if sim_val is not None and gt_val is not None and gt_val > 0:
                    err = pct_error(sim_val, gt_val)
                    row[f"{metric_name}_sim"] = sim_val
                    row[f"{metric_name}_gt"] = gt_val
                    row[f"{metric_name}_err%"] = err
                    if not np.isnan(err):
                        metric_errors[metric_name].append(err)
                else:
                    if sim_val is None or gt_val is None:
                        logger.debug(
                            "Energy/Token missing for %s (sim=%s, gt=%s)", key, sim_val, gt_val
                        )
                continue

            pair = extract_metric(gt, sim, gt_path, sim_path)
            if pair is not None:
                sim_val, gt_val = pair
                err = pct_error(sim_val, gt_val)
                row[f"{metric_name}_sim"] = sim_val
                row[f"{metric_name}_gt"] = gt_val
                row[f"{metric_name}_err%"] = err
                if not np.isnan(err):
                    metric_errors[metric_name].append(err)

        rows.append(row)

    # Generate report
    lines = []
    lines.append("# Simulation vs Ground Truth Comparison\n")
    lines.append(f"**Matched runs**: {len(common_keys)}\n")

    # Summary table
    lines.append("\n## Error Summary\n")
    lines.append("| Metric | Median Error | Mean Error | P5 | P95 | Abs Median | N |")
    lines.append("|--------|-------------|------------|-----|-----|------------|---|")

    for metric_name, _, _, _ in metrics:
        errs = metric_errors[metric_name]
        if errs:
            arr = np.array(errs)
            abs_arr = np.abs(arr)
            lines.append(
                f"| {metric_name} | {np.median(arr):+.1f}% | {np.mean(arr):+.1f}% | "
                f"{np.percentile(arr, 5):+.1f}% | {np.percentile(arr, 95):+.1f}% | "
                f"{np.median(abs_arr):.1f}% | {len(errs)} |"
            )

    # Per-model breakdown
    models = sorted(set(r["model"] for r in rows))
    lines.append("\n## Per-Model Error (TTFT p50)\n")
    lines.append("| Model | Median Error | Abs Median | N |")
    lines.append("|-------|-------------|------------|---|")

    for model in models:
        errs = [r["TTFT p50_err%"] for r in rows if r["model"] == model and "TTFT p50_err%" in r]
        if errs:
            arr = np.array(errs)
            lines.append(
                f"| {model} | {np.median(arr):+.1f}% | {np.median(np.abs(arr)):.1f}% | {len(errs)} |"
            )

    # Per-model breakdown for TBT p50
    lines.append("\n## Per-Model Error (TBT p50)\n")
    lines.append("| Model | Median Error | Abs Median | N |")
    lines.append("|-------|-------------|------------|---|")

    for model in models:
        errs = [r["TBT p50_err%"] for r in rows if r["model"] == model and "TBT p50_err%" in r]
        if errs:
            arr = np.array(errs)
            lines.append(
                f"| {model} | {np.median(arr):+.1f}% | {np.median(np.abs(arr)):.1f}% | {len(errs)} |"
            )

    # Per-model breakdown for throughput
    lines.append("\n## Per-Model Error (Throughput)\n")
    lines.append("| Model | Median Error | Abs Median | N |")
    lines.append("|-------|-------------|------------|---|")

    for model in models:
        errs = [r["Throughput (tok/s)_err%"] for r in rows
                if r["model"] == model and "Throughput (tok/s)_err%" in r]
        if errs:
            arr = np.array(errs)
            lines.append(
                f"| {model} | {np.median(arr):+.1f}% | {np.median(np.abs(arr)):.1f}% | {len(errs)} |"
            )

    # Per-model breakdown for energy/token
    lines.append("\n## Per-Model Error (Energy/Token)\n")
    lines.append("| Model | Median Error | Abs Median | N |")
    lines.append("|-------|-------------|------------|---|")

    for model in models:
        errs = [r["Energy/Token (J)_err%"] for r in rows if r["model"] == model and "Energy/Token (J)_err%" in r]
        if errs:
            arr = np.array(errs)
            lines.append(
                f"| {model} | {np.median(arr):+.1f}% | {np.median(np.abs(arr)):.1f}% | {len(errs)} |"
            )

    # Per-workload breakdown
    workloads = sorted(set(r["workload"] for r in rows))
    lines.append("\n## Per-Workload Error\n")
    lines.append("| Workload | TTFT p50 | TBT p50 | Throughput | Energy/Tok | N |")
    lines.append("|----------|----------|---------|------------|------------|---|")

    for wl in workloads:
        ttft_errs = [r["TTFT p50_err%"] for r in rows if r["workload"] == wl and "TTFT p50_err%" in r]
        tbt_errs = [r["TBT p50_err%"] for r in rows if r["workload"] == wl and "TBT p50_err%" in r]
        tps_errs = [r["Throughput (tok/s)_err%"] for r in rows if r["workload"] == wl and "Throughput (tok/s)_err%" in r]
        etok_errs = [r["Energy/Token (J)_err%"] for r in rows if r["workload"] == wl and "Energy/Token (J)_err%" in r]
        ttft_str = f"{np.median(ttft_errs):+.1f}%" if ttft_errs else "N/A"
        tbt_str = f"{np.median(tbt_errs):+.1f}%" if tbt_errs else "N/A"
        tps_str = f"{np.median(tps_errs):+.1f}%" if tps_errs else "N/A"
        etok_str = f"{np.median(etok_errs):+.1f}%" if etok_errs else "N/A"
        n = max(len(ttft_errs), len(tbt_errs), len(tps_errs), len(etok_errs))
        lines.append(f"| {wl} | {ttft_str} | {tbt_str} | {tps_str} | {etok_str} | {n} |")

    # Detailed per-run table (TTFT + TBT + Throughput + Energy)
    lines.append("\n## Detailed Results\n")
    lines.append(
        "| Model | Config | Workload | QPS | "
        "TTFT Sim | TTFT GT | TTFT Err | "
        "TBT Sim | TBT GT | TBT Err | "
        "TPS Sim | TPS GT | TPS Err | "
        "E/Tok Sim | E/Tok GT | E/Tok Err |"
    )
    lines.append(
        "|-------|--------|----------|-----|"
        "---------|---------|----------|"
        "---------|---------|----------|"
        "---------|---------|----------|"
        "----------|----------|-----------|"
    )

    for r in rows:
        def _fmt(key: str, precision: int = 4) -> tuple[str, str, str]:
            sim_v = r.get(f"{key}_sim", None)
            gt_v = r.get(f"{key}_gt", None)
            err = r.get(f"{key}_err%", float("nan"))
            s = f"{sim_v:.{precision}f}" if isinstance(sim_v, (int, float)) else "N/A"
            g = f"{gt_v:.{precision}f}" if isinstance(gt_v, (int, float)) else "N/A"
            e = f"{err:+.1f}%" if not np.isnan(err) else "N/A"
            return s, g, e

        ts, tg, te = _fmt("TTFT p50")
        bs, bg, be = _fmt("TBT p50")
        ps, pg, pe = _fmt("Throughput (tok/s)", 1)
        es, eg, ee = _fmt("Energy/Token (J)", 4)
        lines.append(
            f"| {r['model']} | {r['config']} | {r['workload']} | {r['qps']:.0f} | "
            f"{ts} | {tg} | {te} | "
            f"{bs} | {bg} | {be} | "
            f"{ps} | {pg} | {pe} | "
            f"{es} | {eg} | {ee} |"
        )

    report = "\n".join(lines)

    # Write or print
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info("Report written to %s", output_path)
    else:
        print(report)


if __name__ == "__main__":
    main()
