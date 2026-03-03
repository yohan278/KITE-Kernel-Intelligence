#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WRAP_DIR="$SCRIPT_DIR/wrappers"
mkdir -p "$WRAP_DIR"

AGENTS=(
  sg0 sg1 sg2 sg3 sg4 sg5 sg6 sg7
  sv0
  ex0 ex1 ex2 ex3 ex4 ex5
  mn0 pa0 st0 pl0 tb0
)

for agent in "${AGENTS[@]}"; do
  cat > "$WRAP_DIR/${agent}.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
"\$(cd "\$(dirname "\$0")/.." && pwd)/90_run_agent.sh" "$agent"
EOF
  chmod +x "$WRAP_DIR/${agent}.sh"
done

echo "Generated ${#AGENTS[@]} wrapper scripts under $WRAP_DIR"
