#!/usr/bin/env bash
#SBATCH --job-name=kite-l1l2-energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/kite_l1l2_%j.out
#SBATCH --error=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/kite_l1l2_%j.err

#SBATCH --signal=B:SIGTERM@60

set -euo pipefail

cleanup() {
    echo "$(date): Received signal, job ending. Checkpoints saved by trainer."
    exit 0
}
trap cleanup SIGTERM

ROOT="/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence"
cd "$ROOT"
mkdir -p "$ROOT/logs"

echo "=== KITE 12h L1+L2 Energy-Aware GRPO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $SLURM_GPUS_ON_NODE"
echo "Start:  $(date)"
echo "ROOT:   $ROOT"
echo ""

if command -v module &>/dev/null; then
    module load cuda/12.9.0 2>/dev/null || true
fi

source "$ROOT/.venv/bin/activate"
PYTHON="$ROOT/.venv/bin/python"
echo "Python: $PYTHON"
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA avail: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs for distributed training"
echo ""

CONFIG="$ROOT/configs/train_12h_l1l2_energy.yaml"
OUTPUT="$ROOT/checkpoints/grpo_12h_l1l2_energy_$(date +%Y%m%d_%H%M)"
KB_ROOT="$ROOT/external/KernelBench"

echo "Config:  $CONFIG"
echo "Output:  $OUTPUT"
echo "KB Root: $KB_ROOT"
echo ""

accelerate launch \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=bf16 \
    "$ROOT/scripts/train_rl.py" \
    --config "$CONFIG" \
    --kernelbench-root "$KB_ROOT" \
    --output "$OUTPUT" \
    --heartbeat-seconds 120

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo "Checkpoints: $OUTPUT"
