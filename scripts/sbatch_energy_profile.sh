#!/usr/bin/env bash
#SBATCH --job-name=kite-eprofile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/energy_profile_%j.out
#SBATCH --error=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/energy_profile_%j.err

set -euo pipefail

ROOT="/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/results"

echo "=== KITE Energy Profiling Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo ""

if command -v module &>/dev/null; then
    module load cuda/12.9.0 2>/dev/null || true
fi

source "$ROOT/.venv/bin/activate"
PYTHON="$ROOT/.venv/bin/python"
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

$PYTHON "$ROOT/scripts/energy_profiling_experiment.py" \
    --kernelbench-root "$ROOT/external/KernelBench" \
    --output-dir "$ROOT/results/energy_profiling" \
    --levels "1,2" \
    --num-trials 3 \
    --timeout 60

echo ""
echo "=== Done ==="
echo "End: $(date)"
