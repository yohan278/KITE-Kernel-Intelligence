#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_env.sh"

mkdir -p "$ROOT/outputs/agent_queue/logs"

# Generate wrappers if missing.
"$SCRIPT_DIR/91_make_agent_wrappers.sh"

launch_bg() {
  local agent="$1"
  "$SCRIPT_DIR/wrappers/${agent}.sh" > "$ROOT/outputs/agent_queue/logs/${agent}.log" 2>&1 &
  echo "$!" > "$ROOT/outputs/agent_queue/logs/${agent}.pid"
  echo "launched $agent pid=$(cat "$ROOT/outputs/agent_queue/logs/${agent}.pid")"
}

# CPU-only prep pipeline.
for a in sg0 sg1 sg2 sg3 sg4 sg5 sg6 sg7; do launch_bg "$a"; done
launch_bg sv0
for a in ex0 ex1 ex2 ex3 ex4 ex5; do launch_bg "$a"; done
launch_bg mn0
launch_bg pa0
launch_bg st0
launch_bg pl0
launch_bg tb0

echo "All agents launched in background. Logs: $ROOT/outputs/agent_queue/logs"
