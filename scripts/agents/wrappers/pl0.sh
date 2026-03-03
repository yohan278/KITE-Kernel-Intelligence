#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REQ_STATE="$ROOT/outputs/agent_queue/state/st0.done"
REQ_STATS="$ROOT/outputs/agent_queue/stats_summary.json"

missing=0
if [ ! -f "$REQ_STATE" ]; then
  echo "[pl0] missing prerequisite: $REQ_STATE" >&2
  missing=1
fi
if [ ! -f "$REQ_STATS" ]; then
  echo "[pl0] missing prerequisite: $REQ_STATS" >&2
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  echo "[pl0] run st0 first (or complete pa0 -> st0) before pl0" >&2
  exit 1
fi

"$SCRIPT_DIR/90_run_agent.sh" "pl0"
