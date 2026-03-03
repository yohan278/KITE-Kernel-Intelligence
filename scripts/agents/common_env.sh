#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export ROOT

: "${KITE_CONDA_ENV:=kite-train}"
export KITE_CONDA_ENV

: "${KITE_HF_CACHE:=$HOME/.cache/kite-hf}"
export KITE_HF_CACHE

: "${KITE_HF_LOCAL_FILES_ONLY:=1}"
export KITE_HF_LOCAL_FILES_ONLY

mkdir -p "$ROOT/configs/exp" "$ROOT/outputs/agent_queue" "$ROOT/checkpoints/exp"
