#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== KITE source sync ==="

# ---- KernelBench ----
KB_DIR="$ROOT/external/KernelBench"
if [ -d "$KB_DIR/src/kernelbench" ]; then
    echo "[OK] KernelBench source tree present"
    KB_COMMIT=$(cd "$KB_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "     commit: $KB_COMMIT"
else
    echo "[MISSING] external/KernelBench – clone it:"
    echo "  git clone https://github.com/ScalingIntelligence/KernelBench.git external/KernelBench"
    exit 1
fi

# ---- IPW internal ----
IPW_DIR="$ROOT/external/ipw_internal"
IPW_SRC=""
if [ -d "$IPW_DIR" ] || [ -L "$IPW_DIR" ]; then
    echo "[OK] IPW path exists"

    # Support both direct and nested symlink layouts:
    # 1) external/ipw_internal/intelligence-per-watt
    # 2) external/ipw_internal/ipw_internal/intelligence-per-watt
    for candidate in \
      "$IPW_DIR/intelligence-per-watt/src" \
      "$IPW_DIR/ipw_internal/intelligence-per-watt/src"; do
      if [ -f "$candidate/ipw/execution/runner.py" ]; then
        IPW_SRC="$candidate"
        break
      fi
    done

    if [ -n "$IPW_SRC" ]; then
        echo "     ProfilerRunner found"
    elif [ -f "$IPW_DIR/README.md" ]; then
        echo "     README found (symlink target may be incomplete)"
    fi
else
    echo "[MISSING] external/ipw_internal – create a symlink:"
    echo "  ln -s /path/to/ipw_internal external/ipw_internal"
    exit 1
fi

# ---- Python package ----
if python -c "import kite" 2>/dev/null; then
    echo "[OK] kite package importable"
else
    echo "[WARN] kite not importable; run: pip install -e ."
fi

# ---- GPU availability ----
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "[OK] CUDA available: $GPU_NAME"
else
    echo "[INFO] No CUDA GPU detected (stub mode will be used)"
fi

# ---- IPW telemetry smoke check ----
if [ -n "$IPW_SRC" ] && python -c "
import sys, os
sys.path.insert(0, '$IPW_SRC')
from ipw.execution.runner import ProfilerRunner
print('IPW ProfilerRunner importable')
" 2>/dev/null; then
    echo "[OK] IPW profiler importable"
else
    echo "[INFO] IPW profiler not importable (will use synthetic traces)"
fi

echo ""
echo "=== Sync complete ==="
