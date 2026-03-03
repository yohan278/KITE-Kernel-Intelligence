#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    root = args.root
    ckpt_root = root / "checkpoints" / "exp"
    out_dir = root / "outputs" / "agent_queue"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parsed_metrics.jsonl"

    rows: list[dict[str, Any]] = []
    for cp in sorted(ckpt_root.rglob("checkpoint.json")):
        payload = _load_json(cp)
        if payload is None:
            continue
        row = {
            "checkpoint_path": str(cp),
            "stage": payload.get("stage"),
            "mode": payload.get("mode"),
            "epochs": payload.get("epochs"),
            "num_tasks": payload.get("num_tasks") or payload.get("num_records"),
            "avg_reward": payload.get("avg_reward"),
            "train_loss": payload.get("train_loss"),
            "batch_size": payload.get("batch_size"),
            "group_size": payload.get("group_size"),
        }
        rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[pa0] wrote {len(rows)} row(s) to {out_path}")
    (out_dir / "state").mkdir(parents=True, exist_ok=True)
    (out_dir / "state" / "pa0.done").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
