#!/usr/bin/env bash
# Run smoke_test_one_task.py from project root (works with or without GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# Use venv if present
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
# Optional: load cluster modules (uncomment if on cluster)
# module load cuda/12.9.0 python/3.13.5

CONFIG="${1:-configs/smoke.yaml}"
python scripts/smoke_test_one_task.py --config "$CONFIG"
