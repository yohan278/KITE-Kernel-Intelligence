#!/usr/bin/env bash
#SBATCH --job-name=kite-full-exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/full_experiment_%j.out
#SBATCH --error=/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence/logs/full_experiment_%j.err

set -uo pipefail
ROOT="/home/users/yaklilu/Desktop/cs234/KITE-Kernel-Intelligence"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/results"

echo "=== KITE Full Energy Experiment ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"

if command -v module &>/dev/null; then module load cuda/12.9.0 2>/dev/null || true; fi
source "$ROOT/.venv/bin/activate"
PYTHON="$ROOT/.venv/bin/python"
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== STEP 1: Energy Profiling (all 200 kernels) ==="
$PYTHON "$ROOT/scripts/energy_profiling_experiment.py" \
    --kernelbench-root "$ROOT/external/KernelBench" \
    --output-dir "$ROOT/results/energy_profiling" \
    --levels "1,2" --num-trials 3 --timeout 60

echo ""
echo "=== STEP 2: Input Size Scaling (3 per type x 4 scales) ==="
$PYTHON "$ROOT/scripts/input_size_scaling_experiment.py" \
    --kernelbench-root "$ROOT/external/KernelBench" \
    --output-dir "$ROOT/results/input_scaling" \
    --levels "1,2" --num-trials 3 --timeout 60 --max-per-type 3 \
    || echo "WARNING: Scaling experiment had errors (some scales may OOM)"

echo ""
echo "=== STEP 3: Generate Charts ==="
$PYTHON "$ROOT/scripts/generate_charts.py" \
    --profile-dir "$ROOT/results/energy_profiling" \
    --scaling-dir "$ROOT/results/input_scaling" \
    --output-dir "$ROOT/results/charts"

echo ""
echo "=== Done! End: $(date) ==="
