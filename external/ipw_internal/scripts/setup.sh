#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/intelligence-per-watt"

info() { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

# --- uv ---
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
info "uv $(uv --version)"

# --- venv + Python 3.13 (uv fetches it automatically) ---
if [ ! -d "$PKG/.venv" ]; then
    info "Creating virtual environment (Python 3.13)..."
    uv venv "$PKG/.venv" --python 3.13
fi
source "$PKG/.venv/bin/activate"

# --- Install package ---
EXTRAS="${1:-all}"
info "Installing intelligence-per-watt[${EXTRAS}]..."
uv pip install -e "${PKG}[${EXTRAS}]"

# --- Energy monitor (optional, needs rust + protoc) ---
if command -v cargo &>/dev/null && command -v protoc &>/dev/null; then
    info "Building energy monitor..."
    uv run "$ROOT/scripts/build_energy_monitor.py"
else
    warn "Skipping energy monitor build (install rust and protoc first)"
    warn "  Rust:   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    warn "  Protoc: sudo apt install -y protobuf-compiler"
fi

info "Done! Activate with: source $PKG/.venv/bin/activate"
info "Test with: ipw --help"
