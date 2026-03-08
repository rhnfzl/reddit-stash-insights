#!/usr/bin/env bash
# Install git hooks for this project.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"

# Pre-push hook
ln -sf "$REPO_ROOT/scripts/pre-push-quality.sh" "$HOOKS_DIR/pre-push"
echo "Installed pre-push hook (ruff + mypy on changed files)"
