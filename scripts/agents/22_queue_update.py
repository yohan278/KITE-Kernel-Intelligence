#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone


VALID_STATUSES = {"prepared", "submitted", "running", "done", "failed"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--return-code", type=int, default=0)
    p.add_argument("--log-path", default="")
    args = p.parse_args()

    if args.status not in VALID_STATUSES:
        raise SystemExit(f"invalid status: {args.status} (valid: {sorted(VALID_STATUSES)})")

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    finished_at = now if args.status in {"done", "failed"} else None
    cur.execute(
        """
        UPDATE runs
        SET status=?, return_code=?, log_path=?, finished_at=COALESCE(?, finished_at)
        WHERE run_id=?
        """,
        (args.status, args.return_code, args.log_path, finished_at, args.run_id),
    )
    conn.commit()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
