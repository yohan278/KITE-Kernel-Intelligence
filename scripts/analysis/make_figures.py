#!/usr/bin/env python3
"""Generate figure artifacts from suite results."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.eval.reports import plot_pareto_frontiers
from kite.utils.serialization import load_json, save_json


def main() -> int:
    suite_path = Path("./outputs/eval/suite_results.json")
    if not suite_path.exists():
        raise FileNotFoundError("Run evaluation first: outputs/eval/suite_results.json missing")

    suite = load_json(suite_path)
    results = suite.get("results", [])
    output_dir = Path("./outputs/eval/figures")
    plots = plot_pareto_frontiers(results, output_dir)

    manifest = {"figure_paths": {k: str(v) for k, v in plots.items()}}
    save_json(output_dir / "figures_manifest.json", manifest)
    print(f"Wrote figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
