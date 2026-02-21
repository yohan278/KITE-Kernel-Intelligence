#!/usr/bin/env python3
"""Generate Pareto plots from suite results."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.eval.reports import plot_pareto_frontiers
from kite.utils.serialization import load_json


def main() -> int:
    suite_path = Path("./outputs/eval/suite_results.json")
    if not suite_path.exists():
        raise FileNotFoundError(f"Missing suite results: {suite_path}")
    suite = load_json(suite_path)
    results = suite.get("results", [])
    out = Path("./outputs/eval")
    plots = plot_pareto_frontiers(results, out)
    print(f"Generated plots: {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
