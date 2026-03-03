#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--worker", type=int, required=True)
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    conn.isolation_level = None
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")

    cur.execute(
        """
        SELECT id, run_id, kind, COALESCE(seed, ''), COALESCE(config_path, ''), output_dir
        FROM runs
        WHERE status='queued' AND worker_hint=?
        ORDER BY id ASC
        LIMIT 1
        """,
        (args.worker,),
    )
    row = cur.fetchone()

    if row is None:
        cur.execute(
            """
            SELECT id, run_id, kind, COALESCE(seed, ''), COALESCE(config_path, ''), output_dir
            FROM runs
            WHERE status='queued'
            ORDER BY id ASC
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if row is None:
        cur.execute("COMMIT")
        conn.close()
        return 2

    run_db_id = row[0]
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "UPDATE runs SET status='running', started_at=? WHERE id=?",
        (now, run_db_id),
    )
    cur.execute("COMMIT")
    conn.close()

    # TSV output for shell parsing.
    print("\t".join(str(x) for x in row[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
