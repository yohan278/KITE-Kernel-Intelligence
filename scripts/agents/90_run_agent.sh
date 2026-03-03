#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${1:?usage: 90_run_agent.sh <agent_id>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_env.sh"

STATE_DIR="$ROOT/outputs/agent_queue/state"
mkdir -p "$STATE_DIR"

wait_for_file() {
  local f="$1"
  local timeout="${2:-1800}"
  local waited=0
  while [ ! -f "$f" ]; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$timeout" ]; then
      echo "Timeout waiting for $f"
      return 1
    fi
  done
}

case "$AGENT_ID" in
  sg0|sg1|sg2|sg3|sg4|sg5|sg6|sg7)
    python "$SCRIPT_DIR/10_generate_configs.py" --agent "$AGENT_ID" --root "$ROOT"
    ;;

  sv0)
    for i in 0 1 2 3 4 5 6 7; do
      wait_for_file "$STATE_DIR/sg${i}.done"
    done
    python "$SCRIPT_DIR/20_validate_and_manifest.py" --root "$ROOT"
    ;;

  ex0|ex1|ex2|ex3|ex4|ex5)
    wait_for_file "$STATE_DIR/sv0.done"
    idx="${AGENT_ID#ex}"
    "$SCRIPT_DIR/30_executor_worker.sh" "$idx" "$idx"
    ;;

  mn0)
    wait_for_file "$STATE_DIR/sv0.done"
    python "$SCRIPT_DIR/35_monitor_queue.py" --root "$ROOT" --interval 10.0 || true
    echo "ok" > "$STATE_DIR/mn0.done"
    ;;

  pa0)
    # Parse can run after prepare or after done runs.
    for i in 0 1 2 3 4 5; do
      wait_for_file "$STATE_DIR/ex${i}.done" 3600
    done
    python "$SCRIPT_DIR/40_parse_results.py" --root "$ROOT"
    ;;

  st0)
    wait_for_file "$STATE_DIR/pa0.done"
    python "$SCRIPT_DIR/50_compute_stats.py" --root "$ROOT"
    ;;

  pl0)
    wait_for_file "$STATE_DIR/st0.done"
    wait_for_file "$ROOT/outputs/agent_queue/stats_summary.json"
    python "$SCRIPT_DIR/60_make_plots.py" --root "$ROOT"
    ;;

  tb0)
    wait_for_file "$STATE_DIR/st0.done"
    python "$SCRIPT_DIR/70_make_tables.py" --root "$ROOT"
    ;;

  *)
    echo "Unknown agent id: $AGENT_ID"
    exit 2
    ;;
esac
