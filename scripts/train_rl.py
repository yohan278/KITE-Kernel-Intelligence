#!/usr/bin/env python3
"""Phase 4 training wrapper for throughput or energy-aware kernel GRPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite.cli import main as kite_main
from kite.utils.serialization import load_yaml, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kernelbench-root", type=Path, default=Path("./external/KernelBench"))
    parser.add_argument("--output", type=Path, default=Path("./checkpoints/kernel_grpo"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "sweep" in cfg:
        sweep_cfg = cfg["sweep"]
        combos = sweep_cfg.get("alpha_beta", [])
        epochs = int(sweep_cfg.get("epochs", 3))
        out_root = Path(sweep_cfg.get("output_dir", "./outputs/sweeps"))
        records = []
        for idx, combo in enumerate(combos):
            alpha, beta = float(combo[0]), float(combo[1])
            out_dir = out_root / f"alpha_{alpha}_beta_{beta}"
            cmd = [
                "train",
                "kernel-grpo",
                "--kernelbench-root",
                str(args.kernelbench_root),
                "--output",
                str(out_dir),
                "--epochs",
                str(epochs),
                "--energy-aware",
            ]
            rc = kite_main(cmd)
            records.append({"alpha": alpha, "beta": beta, "return_code": rc, "output_dir": str(out_dir)})
        save_json(out_root / "sweep_manifest.json", {"runs": records})
        return 0

    train = cfg.get("train", {})
    energy_aware = bool(train.get("energy_aware", False))
    epochs = int(train.get("epochs", 3))
    cmd = [
        "train",
        "kernel-grpo",
        "--kernelbench-root",
        str(args.kernelbench_root),
        "--output",
        str(args.output),
        "--epochs",
        str(epochs),
    ]
    if energy_aware:
        cmd.append("--energy-aware")
    return kite_main(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
