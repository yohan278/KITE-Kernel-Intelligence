"""Reporting helpers for KITE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from kite.utils.serialization import save_json


def pareto_frontier(points: Iterable[Dict[str, float]], x_key: str, y_key: str) -> List[Dict[str, float]]:
    rows = list(points)
    frontier: list[dict[str, float]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            if other.get(x_key, 0.0) >= candidate.get(x_key, 0.0) and other.get(y_key, 0.0) >= candidate.get(y_key, 0.0):
                if other.get(x_key, 0.0) > candidate.get(x_key, 0.0) or other.get(y_key, 0.0) > candidate.get(y_key, 0.0):
                    dominated = True
                    break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p.get(x_key, 0.0), reverse=True)


def write_markdown_report(output_path: Path, title: str, sections: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for name, value in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(str(value))
        lines.append("")
    output_path.write_text("\n".join(lines))


def save_suite_artifacts(output_dir: Path, suite: Dict[str, Any]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "suite_results.json"
    save_json(json_path, suite)

    points = []
    for row in suite.get("results", []):
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            points.append({"id": row.get("id", "unknown"), "apj": float(metrics.get("apj", 0.0)), "apw": float(metrics.get("apw", 0.0))})

    frontier = pareto_frontier(points, "apj", "apw")
    frontier_path = output_dir / "pareto_frontier.json"
    save_json(frontier_path, {"frontier": frontier})

    md_path = output_dir / "report.md"
    write_markdown_report(
        md_path,
        title="KITE Evaluation Report",
        sections={
            "Summary": f"Evaluated {len(suite.get('results', []))} experiments.",
            "Pareto Frontier": [f"{p['id']}: apj={p['apj']:.4f}, apw={p['apw']:.4f}" for p in frontier],
        },
    )

    return {
        "suite_json": json_path,
        "pareto_json": frontier_path,
        "report_md": md_path,
    }
