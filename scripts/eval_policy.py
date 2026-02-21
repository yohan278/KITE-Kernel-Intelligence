#!/usr/bin/env python3
"""Evaluate policies/checkpoints via KITE benchmark suite runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.eval.benchmark_runner import BenchmarkRunner
from kite.utils.serialization import save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/policy_eval"))
    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output, checkpoints_root=args.checkpoints_root)
    suite = runner.run()
    save_json(args.output / "policy_eval.json", suite)
    print(f"Saved evaluation to {args.output / 'policy_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
