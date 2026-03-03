#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_env.sh"

GPU_ID="${1:?usage: 93_run_prepared_jobs_gpu.sh <gpu_id>}"

DB="$ROOT/outputs/agent_queue/queue.db"
if [ ! -f "$DB" ]; then
  echo "queue db not found: $DB"
  exit 1
fi

python - <<PY
import sqlite3
from pathlib import Path
root = Path("$ROOT")
db = root / "outputs/agent_queue/queue.db"
conn = sqlite3.connect(db)
cur = conn.cursor()
rows = cur.execute("SELECT run_id, log_path FROM runs WHERE status='prepared' ORDER BY id").fetchall()
conn.close()
for run_id, log_path in rows:
    print(f"{run_id}\t{log_path}")
PY

echo "Use the prepared script paths above to submit on GPU ${GPU_ID}."
echo "Example: CUDA_VISIBLE_DEVICES=${GPU_ID} bash <job_script>"
