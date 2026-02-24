#!/usr/bin/env bash
# Start vLLM server for GLM-4.7-Flash (MoE model)
# Requires: pip install vllm>=0.11.0
# Hardware: 4x GPU with tensor parallelism

set -euo pipefail

MODEL_ID="${MODEL_ID:-zai-org/GLM-4.7-Flash}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"

echo "Starting vLLM server for ${MODEL_ID} on port ${PORT} with TP=${TP_SIZE}"

vllm serve "${MODEL_ID}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len 32768 \
    "$@"
