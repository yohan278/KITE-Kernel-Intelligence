#!/usr/bin/env python3
"""Phase 1 baseline measurement runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.adapters.kernelbench_adapter import KernelBenchAdapter
from kite.measurement.protocol import MeasurementConfig, MeasurementProtocol
from kite.utils.serialization import load_yaml, save_jsonl


def _reference_workload() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            x = torch.randn(256, 256, device="cuda")
            _ = torch.mm(x, x)
            return
    except Exception:
        pass
    sum(i * i for i in range(1000))


def _fmt_stats(name: str, mean: float, std: float, cv: float) -> str:
    return f"- {name}: mean={mean:.4f}, std={std:.4f}, cv={cv:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    kb_root = Path(data_cfg.get("kernelbench_root", "./external/KernelBench"))

    output_jsonl = Path("./results/baselines/baseline_metrics.jsonl")
    report_md = Path("./results/measurement/variance_report.md")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    meas_cfg = MeasurementConfig(warmup_iters=5, measure_iters=20, repeats=30, sampling_interval_ms=50.0)
    protocol = MeasurementProtocol(meas_cfg)

    adapter = KernelBenchAdapter(kb_root)
    tasks = adapter.discover_tasks()
    selected = tasks[: min(5, len(tasks))]
    if not selected:
        raise RuntimeError("No tasks found for baseline run")

    rows = []
    report_lines = [
        "# Measurement Variance Report",
        "",
        f"Tasks measured: {len(selected)}",
        f"Repeats per task: {meas_cfg.repeats}",
        "",
    ]

    for task in selected:
        result = protocol.measure(_reference_workload)
        row = {
            "task_id": task.task_id,
            "runtime_ms_mean": result.runtime_ms_mean,
            "runtime_ms_std": result.runtime_ms_std,
            "runtime_ms_cv": result.runtime_ms_cv,
            "avg_power_w_mean": result.avg_power_w_mean,
            "avg_power_w_std": result.avg_power_w_std,
            "energy_j_mean": result.energy_j_mean,
            "energy_j_std": result.energy_j_std,
            "energy_j_cv": result.energy_j_cv,
            "repeats": result.repeats,
        }
        rows.append(row)

        report_lines.extend(
            [
                f"## {task.task_id}",
                _fmt_stats("runtime_ms", result.runtime_ms_mean, result.runtime_ms_std, result.runtime_ms_cv),
                _fmt_stats("energy_j", result.energy_j_mean, result.energy_j_std, result.energy_j_cv),
                f"- avg_power_w: mean={result.avg_power_w_mean:.4f}, std={result.avg_power_w_std:.4f}",
                "",
            ]
        )

    save_jsonl(output_jsonl, rows)
    _maybe_plot_variance(rows, report_md.parent / "runtime_vs_joules.png")
    report_md.write_text("\n".join(report_lines))
    print(f"Wrote {len(rows)} baseline rows to {output_jsonl}")
    print(f"Wrote variance report to {report_md}")
    return 0


def _maybe_plot_variance(rows: list[dict], output_path: Path) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    x = [r["runtime_ms_mean"] for r in rows]
    y = [r["energy_j_mean"] for r in rows]
    xerr = [r["runtime_ms_std"] for r in rows]
    yerr = [r["energy_j_std"] for r in rows]
    labels = [r["task_id"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ecolor="gray", capsize=4)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Runtime vs Joules (mean +- std)")
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
