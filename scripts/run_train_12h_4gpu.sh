#!/usr/bin/env bash
# 12-hour kernel GRPO training on 4 GPUs (one node).
# Usage: run from project root, or pass ROOT; optionally set KITE_HF_CACHE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$ROOT"

echo "=== KITE 12h training (4 GPUs) ==="
echo "ROOT: $ROOT"
echo "Start: $(date -Iseconds 2>/dev/null || date)"
echo ""

# Cluster modules (if available)
if command -v module &>/dev/null; then
  module load cuda/12.9.0 2>/dev/null || true
  module load python/3.13.5 2>/dev/null || true
fi

# Prefer venv; fallback to conda kite-train
PYTHON=""
if [ -f "$ROOT/.venv/bin/python" ]; then
  source "$ROOT/.venv/bin/activate" 2>/dev/null || true
  PYTHON="$ROOT/.venv/bin/python"
fi
if [ -z "$PYTHON" ] && command -v conda &>/dev/null; then
  conda activate kite-train 2>/dev/null || true
  PYTHON="$(which python)"
fi
if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi
echo "Python: $PYTHON"

CONFIG="$ROOT/configs/train_12h_l40.yaml"
OUTPUT="$ROOT/checkpoints/grpo_12h_l40"
KB_ROOT="$ROOT/external/KernelBench"

EXTRA=()
[ -n "${KITE_HF_CACHE:-}" ] && EXTRA+=(--hf-cache-dir "$KITE_HF_CACHE")
[ "${KITE_HF_LOCAL_FILES_ONLY:-0}" = "1" ] && EXTRA+=(--local-files-only)

echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo "KernelBench: $KB_ROOT"
echo ""

srun --partition=gpu \
     --gres=gpu:4 \
     --mem=128G \
     --cpus-per-task=16 \
     --time=12:00:00 \
     --job-name=kite-12h \
     "$PYTHON" "$ROOT/scripts/train_rl.py" \
       --config "$CONFIG" \
       --kernelbench-root "$KB_ROOT" \
       --output "$OUTPUT" \
       --heartbeat-seconds 120 \
       "${EXTRA[@]}"

echo ""
echo "End: $(date -Iseconds 2>/dev/null || date)"
echo "Checkpoints: $OUTPUT"
