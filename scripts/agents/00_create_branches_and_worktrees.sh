#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKTREE_ROOT="${1:-$ROOT/../agent_worktrees}"
BASE_BRANCH="${2:-main}"

AGENTS=(
  sg0 sg1 sg2 sg3 sg4 sg5 sg6 sg7
  sv0
  ex0 ex1 ex2 ex3 ex4 ex5
  mn0 pa0 st0 pl0 tb0
)

mkdir -p "$WORKTREE_ROOT"
cd "$ROOT"

git fetch --all --prune

for agent in "${AGENTS[@]}"; do
  branch="codex/${agent}"
  wt="$WORKTREE_ROOT/$agent"

  if ! git show-ref --verify --quiet "refs/heads/$branch"; then
    git branch "$branch" "$BASE_BRANCH"
  fi

  if [ ! -d "$wt/.git" ] && [ ! -f "$wt/.git" ]; then
    git worktree add "$wt" "$branch"
  fi

done

echo "Created/verified ${#AGENTS[@]} branches and worktrees at: $WORKTREE_ROOT"
