#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--until-finished", action="store_true", default=True)
    args = p.parse_args()

    db = args.root / "outputs" / "agent_queue" / "queue.db"
    if not db.exists():
        raise SystemExit(f"queue db not found: {db}")

    while True:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM runs GROUP BY status")
        counts = {k: v for k, v in cur.fetchall()}
        conn.close()

        queued = counts.get("queued", 0)
        prepared = counts.get("prepared", 0)
        running = counts.get("running", 0)
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        total = sum(counts.values())

        print(
            f"[mn0] total={total} queued={queued} prepared={prepared} "
            f"running={running} done={done} failed={failed}",
            flush=True,
        )

        if args.until_finished and queued == 0 and running == 0:
            break
        time.sleep(max(1.0, args.interval))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
