#!/usr/bin/env bash
# Smoke-test GRPO training on 4x L40S GPUs via SLURM.
#
# Usage from login node:
#   bash scripts/run_smoke_l40s.sh
#
# Or if already inside an srun GPU allocation:
#   bash scripts/run_smoke_l40s.sh --local
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

module load cuda/12.9.0 python/3.13.5

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

if [[ "${1:-}" == "--local" ]]; then
    echo "Running locally (assuming GPU access)..."
    python scripts/smoke_train_l40s.py
else
    echo "Submitting via srun (4x GPU)..."
    srun --partition=gpu --gres=gpu:4 --time=00:30:00 \
        bash -c "
            source /etc/profile
            module load cuda/12.9.0 python/3.13.5
            cd $PROJECT_ROOT
            source .venv/bin/activate
            export TOKENIZERS_PARALLELISM=false
            export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
            python scripts/smoke_train_l40s.py
        "
fi
