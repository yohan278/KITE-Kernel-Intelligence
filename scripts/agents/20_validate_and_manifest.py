#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def _check_required_train(cfg: dict[str, Any], path: Path) -> None:
    if "train" not in cfg or not isinstance(cfg["train"], dict):
        raise ValueError(f"Missing train block: {path}")
    t = cfg["train"]
    required = ["epochs", "energy_aware", "group_size", "batch_size", "max_completion_length", "reward"]
    for k in required:
        if k not in t:
            raise ValueError(f"Missing train.{k}: {path}")


def _worker_hint(run_id: str, workers: int = 6) -> int:
    h = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % workers


def _infer_output_dir(root: Path, rel_cfg: Path) -> Path:
    stem = rel_cfg.stem
    if "throughput" in str(rel_cfg):
        return root / "checkpoints" / "exp" / "throughput" / stem
    if "energy" in str(rel_cfg) and "ipw_blend" not in str(rel_cfg):
        return root / "checkpoints" / "exp" / "energy" / stem
    if "ipw_blend" in str(rel_cfg):
        return root / "checkpoints" / "exp" / "ipw_blend" / stem
    if "abl_reward" in str(rel_cfg):
        return root / "checkpoints" / "exp" / "abl_reward" / stem
    if "abl_scale" in str(rel_cfg):
        return root / "checkpoints" / "exp" / "abl_scale" / stem
    if "abl_budget" in str(rel_cfg):
        return root / "checkpoints" / "exp" / "abl_budget" / stem
    return root / "checkpoints" / "exp" / "misc" / stem


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()

    root = args.root
    exp_root = root / "configs" / "exp"
    queue_dir = root / "outputs" / "agent_queue"
    state_dir = queue_dir / "state"
    queue_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    expected_done = [state_dir / f"sg{i}.done" for i in range(8)]
    missing = [str(p) for p in expected_done if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing generator completion files: {missing}")

    cfg_paths = sorted(exp_root.rglob("*.yaml"))
    train_cfgs: list[Path] = []
    for path in cfg_paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "train" in cfg:
            _check_required_train(cfg, path)
            train_cfgs.append(path)

    rows: list[dict[str, Any]] = []
    for cfg_path in train_cfgs:
        rel = cfg_path.relative_to(root)
        run_id = f"train::{rel.as_posix()}"
        out_dir = _infer_output_dir(root, rel)
        rows.append(
            {
                "run_id": run_id,
                "kind": "train_rl",
                "seed": None,
                "config_path": str(cfg_path),
                "output_dir": str(out_dir),
                "worker_hint": _worker_hint(run_id),
            }
        )

    for seed in (11, 22, 33):
        run_id = f"runtime_ppo::seed{seed}"
        rows.append(
            {
                "run_id": run_id,
                "kind": "runtime_ppo",
                "seed": seed,
                "config_path": "",
                "output_dir": str(root / "checkpoints" / "exp" / "runtime_ppo" / f"seed{seed}"),
                "worker_hint": _worker_hint(run_id),
            }
        )
    for seed in (11, 22, 33):
        run_id = f"hrl::seed{seed}"
        rows.append(
            {
                "run_id": run_id,
                "kind": "hrl",
                "seed": seed,
                "config_path": "",
                "output_dir": str(root / "checkpoints" / "exp" / "hrl" / f"seed{seed}"),
                "worker_hint": _worker_hint(run_id),
            }
        )

    manifest_csv = queue_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_id", "kind", "seed", "config_path", "output_dir", "worker_hint"],
        )
        w.writeheader()
        w.writerows(rows)

    db_path = queue_dir / "queue.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT UNIQUE NOT NULL,
          kind TEXT NOT NULL,
          seed INTEGER,
          config_path TEXT,
          output_dir TEXT NOT NULL,
          worker_hint INTEGER NOT NULL,
          status TEXT NOT NULL,
          return_code INTEGER,
          log_path TEXT,
          started_at TEXT,
          finished_at TEXT
        )
        """
    )
    cur.executemany(
        """
        INSERT INTO runs(run_id, kind, seed, config_path, output_dir, worker_hint, status)
        VALUES(?, ?, ?, ?, ?, ?, 'queued')
        """,
        [
            (
                r["run_id"],
                r["kind"],
                r["seed"],
                r["config_path"],
                r["output_dir"],
                r["worker_hint"],
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()

    (state_dir / "sv0.done").write_text("ok\n", encoding="utf-8")
    print(f"[sv0] validated {len(train_cfgs)} train config(s)")
    print(f"[sv0] manifest: {manifest_csv}")
    print(f"[sv0] sqlite queue: {db_path}")
    print(f"[sv0] queued runs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
