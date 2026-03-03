#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common_env.sh"

WORKER_IDX="${1:?usage: 30_executor_worker.sh <worker_idx> <gpu_id>}"
GPU_ID="${2:?usage: 30_executor_worker.sh <worker_idx> <gpu_id>}"

DB="$ROOT/outputs/agent_queue/queue.db"
JOBS_DIR="$ROOT/outputs/agent_queue/jobs"
LOG_DIR="$ROOT/outputs/agent_queue/logs"
STATE_DIR="$ROOT/outputs/agent_queue/state"
mkdir -p "$JOBS_DIR" "$LOG_DIR" "$STATE_DIR"

if [ ! -f "$DB" ]; then
  echo "queue db not found: $DB"
  exit 1
fi

slugify() {
  echo "$1" | sed 's#[^A-Za-z0-9_.-]#_#g'
}

build_cmd() {
  local kind="$1" seed="$2" config_path="$3" output_dir="$4"
  case "$kind" in
    train_rl)
      cat <<EOF
conda run --no-capture-output -n "$KITE_CONDA_ENV" python "$ROOT/scripts/train_rl.py" \\
  --config "$config_path" \\
  --output "$output_dir" \\
  --hf-cache-dir "$KITE_HF_CACHE" \\
  --local-files-only \\
  --heartbeat-seconds 20
EOF
      ;;
    runtime_ppo)
      cat <<EOF
conda run -n "$KITE_CONDA_ENV" python -m kite.cli --seed "$seed" train runtime-ppo \\
  --output "$output_dir"
EOF
      ;;
    hrl)
      cat <<EOF
conda run -n "$KITE_CONDA_ENV" python -m kite.cli --seed "$seed" train hrl \\
  --kernelbench-root "$ROOT/external/KernelBench" \\
  --output "$output_dir" \\
  --rounds 2
EOF
      ;;
    *)
      echo "Unsupported kind: $kind" >&2
      return 2
      ;;
  esac
}

while true; do
  set +e
  CLAIM_OUT="$(python "$SCRIPT_DIR/21_queue_claim.py" --db "$DB" --worker "$WORKER_IDX" 2>/dev/null)"
  CLAIM_RC=$?
  set -e

  if [ "$CLAIM_RC" -eq 2 ]; then
    echo "[ex${WORKER_IDX}] queue empty"
    break
  fi
  if [ "$CLAIM_RC" -ne 0 ]; then
    echo "[ex${WORKER_IDX}] claim failed rc=$CLAIM_RC"
    sleep 2
    continue
  fi

  IFS=$'\t' read -r RUN_ID KIND SEED CONFIG_PATH OUTPUT_DIR <<< "$CLAIM_OUT"
  SAFE_RUN_ID="$(slugify "$RUN_ID")"
  JOB_SCRIPT="$JOBS_DIR/${SAFE_RUN_ID}.sh"
  RUN_LOG="$LOG_DIR/${SAFE_RUN_ID}.log"

  mkdir -p "$OUTPUT_DIR"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export ROOT=\"$ROOT\""
    echo "export KITE_HF_CACHE=\"$KITE_HF_CACHE\""
    echo "export KITE_HF_LOCAL_FILES_ONLY=\"$KITE_HF_LOCAL_FILES_ONLY\""
    echo "export CUDA_VISIBLE_DEVICES=\"$GPU_ID\""
    build_cmd "$KIND" "$SEED" "$CONFIG_PATH" "$OUTPUT_DIR"
  } > "$JOB_SCRIPT"
  chmod +x "$JOB_SCRIPT"

  # Hard-locked prep mode: do not execute experiments from executor agents.
  RC=0
  STATUS="prepared"
  python "$SCRIPT_DIR/22_queue_update.py" \
    --db "$DB" --run-id "$RUN_ID" --status "$STATUS" --return-code "$RC" --log-path "$JOB_SCRIPT"
  echo "[ex${WORKER_IDX}] prepared run_id=$RUN_ID job_script=$JOB_SCRIPT"
done

( echo "ok" > "$STATE_DIR/ex${WORKER_IDX}.done" )
