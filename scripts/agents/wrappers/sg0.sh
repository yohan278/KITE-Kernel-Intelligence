#!/usr/bin/env bash
set -euo pipefail
"$(cd "$(dirname "$0")/.." && pwd)/90_run_agent.sh" "sg0"
