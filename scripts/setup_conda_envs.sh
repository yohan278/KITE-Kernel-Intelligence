#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

CREATE_ALL=false
WITH_IPW=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      CREATE_ALL=true
      shift
      ;;
    --with-ipw)
      WITH_IPW=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--all] [--with-ipw]"
      exit 1
      ;;
  esac
done

declare -a ENVS=("kite-core")
if [[ "$CREATE_ALL" == "true" ]]; then
  ENVS=("kite-core" "kite-train" "kite-telemetry")
fi

if [[ "$WITH_IPW" == "true" ]]; then
  has_telemetry=false
  for env_name in "${ENVS[@]}"; do
    if [[ "$env_name" == "kite-telemetry" ]]; then
      has_telemetry=true
      break
    fi
  done
  if [[ "$has_telemetry" == "false" ]]; then
    echo "[INFO] --with-ipw requires kite-telemetry (Python 3.13); adding it to setup list"
    ENVS+=("kite-telemetry")
  fi
fi

create_or_update() {
  local env_name="$1"
  local spec_path="$2"

  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "[UPDATE] $env_name from $spec_path"
    conda env update -n "$env_name" -f "$spec_path" --prune
  else
    echo "[CREATE] $env_name from $spec_path"
    conda env create -f "$spec_path"
  fi
}

install_python_deps() {
  local env_name="$1"
  shift
  local extra_packages=("$@")

  echo "[PIP] Installing repo + packages into $env_name"
  conda run -n "$env_name" pip install --upgrade pip setuptools wheel
  conda run -n "$env_name" pip install -e "$ROOT"
  if [[ "${#extra_packages[@]}" -gt 0 ]]; then
    conda run -n "$env_name" pip install "${extra_packages[@]}"
  fi
}

for env_name in "${ENVS[@]}"; do
  create_or_update "$env_name" "$ROOT/envs/${env_name}.yml"
  case "$env_name" in
    kite-core)
      install_python_deps "$env_name" "pytest>=7" "matplotlib>=3.8"
      ;;
    kite-train)
      install_python_deps "$env_name" \
        "pytest>=7" \
        "matplotlib>=3.8" \
        "torch>=2.1" \
        "transformers>=4.40" \
        "peft>=0.10" \
        "trl>=0.9" \
        "accelerate>=0.30" \
        "datasets>=2.18" \
        "tqdm>=4.67" \
        "openai>=2.0" \
        "litellm>=1.81"
      ;;
    kite-telemetry)
      install_python_deps "$env_name" \
        "pytest>=7" \
        "datasets>=2.18" \
        "pynvml>=11.5" \
        "grpcio>=1.62"
      ;;
    *)
      install_python_deps "$env_name"
      ;;
  esac
done

if [[ "$WITH_IPW" == "true" ]]; then
  IPW_PATH=""
  for candidate in \
    "$ROOT/external/ipw_internal/intelligence-per-watt" \
    "$ROOT/external/ipw_internal/ipw_internal/intelligence-per-watt"; do
    if [[ -d "$candidate" ]]; then
      IPW_PATH="$candidate"
      break
    fi
  done

  if [[ -n "$IPW_PATH" ]]; then
    echo "[PIP] Installing local IPW package into kite-telemetry"
    conda run -n "kite-telemetry" pip install -e "$IPW_PATH"
  else
    echo "[WARN] IPW path not found under external/ipw_internal; skipping IPW install"
  fi
fi

echo ""
echo "Conda setup complete."
echo "Available environments:"
conda env list | sed -n '1,200p'
echo ""
echo "Activate with: conda activate kite-core"
