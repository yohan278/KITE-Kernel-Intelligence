#!/usr/bin/env bash
# Launch the 5-step online RL smoke test on 2x L40S GPUs via srun.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Online RL Smoke Test Launcher ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Load cluster modules if available
if command -v module &>/dev/null; then
    module load cuda/12.9.0 2>/dev/null || true
    module load python/3.13.5 2>/dev/null || true
fi

# Activate venv
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Activated venv: $(which python)"
fi

# Verify GPU allocation
echo ""
echo "Requesting 2x L40S GPUs via srun..."
echo ""

srun --partition=gpu \
     --gres=gpu:2 \
     --mem=64G \
     --cpus-per-task=8 \
     --time=02:00:00 \
     --job-name=kite-smoke-rl \
     python "$PROJECT_ROOT/scripts/smoke_rl_online.py"
