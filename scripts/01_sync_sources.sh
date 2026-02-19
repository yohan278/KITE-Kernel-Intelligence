#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[01] Source sync checklist"
echo "- Populate external/KernelBench with pinned commit"
echo "- Populate external/ipw_internal (or symlink)"

test -d external/KernelBench && echo "KernelBench path exists" || echo "KernelBench path missing"
test -d external/ipw_internal && echo "IPW path exists" || echo "IPW path missing"
