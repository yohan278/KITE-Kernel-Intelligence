#!/usr/bin/env bash
# Allocate a single L40 (or one GPU) and run NVML signal check.
#
# Usage from login node (SLURM):
#   bash scripts/run_check_nvml_l40.sh
#
# Or if already inside an srun GPU allocation with one GPU:
#   bash scripts/run_check_nvml_l40.sh --local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Prefer same env as smoke L40s
if command -v module &>/dev/null; then
  module load cuda/12.9.0 python/3.13.5 2>/dev/null || true
fi
if [[ -d "$PROJECT_ROOT/.venv" ]]; then
  source .venv/bin/activate
fi

if [[ "${1:-}" == "--local" ]]; then
  echo "Running NVML check on current GPU (use CUDA_VISIBLE_DEVICES=0 for single GPU)..."
  python3 scripts/check_nvml_signals_l40.py
else
  echo "Allocating 1x GPU and running NVML check..."
  srun --partition=gpu --gres=gpu:1 --time=00:05:00 \
    bash -c "
      source /etc/profile 2>/dev/null || true
      command -v module &>/dev/null && module load cuda/12.9.0 python/3.13.5 2>/dev/null || true
      cd $PROJECT_ROOT
      [[ -d .venv ]] && source .venv/bin/activate
      python3 scripts/check_nvml_signals_l40.py
    "
fi
