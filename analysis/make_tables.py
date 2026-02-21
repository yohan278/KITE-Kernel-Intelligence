#!/usr/bin/env python3
"""Build paper-style summary tables from experiment artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.utils.serialization import load_json


def _format_row(exp: dict) -> str:
    m = exp.get("metrics", {})
    return (
        f"| {exp.get('id')} | {m.get('correctness', 0):.3f} | {m.get('speedup', 0):.3f} | "
        f"{m.get('energy_per_token_j', 0):.4f} | {m.get('apj', 0):.4f} | {m.get('apw', 0):.4f} | "
        f"{m.get('ttft_p95_s', 0):.3f} |"
    )


def main() -> int:
    suite_path = Path("./outputs/eval/suite_results.json")
    if not suite_path.exists():
        raise FileNotFoundError("Run evaluation first: outputs/eval/suite_results.json missing")
    suite = load_json(suite_path)
    rows = suite.get("results", [])

    lines = [
        "# KITE Results Table",
        "",
        "| Exp | Correctness | Speedup | Energy/token (J) | APJ | APW | TTFT p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_format_row(r) for r in rows)

    out = Path("./outputs/eval/results_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
