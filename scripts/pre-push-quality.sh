#!/usr/bin/env bash
# Pre-push hook: runs ruff + mypy on changed Python files vs upstream.
# Fail-open: warns but doesn't block if tools aren't installed.
set -euo pipefail

# Determine upstream branch to diff against.
UPSTREAM=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo "origin/main")

# Find changed Python files (staged + committed, not yet pushed).
CHANGED_PY=$(git diff --name-only "$UPSTREAM"...HEAD -- '*.py' 2>/dev/null || true)

if [ -z "$CHANGED_PY" ]; then
    echo "[pre-push] No Python files changed, skipping checks."
    exit 0
fi

echo "[pre-push] Checking $(echo "$CHANGED_PY" | wc -l | tr -d ' ') Python file(s)..."

FAILED=0

# --- Ruff ---
if command -v ruff &>/dev/null; then
    echo "[pre-push] Running ruff..."
    if ! echo "$CHANGED_PY" | xargs ruff check; then
        echo "[pre-push] ruff check FAILED"
        FAILED=1
    fi
else
    echo "[pre-push] WARNING: ruff not found, skipping lint"
fi

# --- mypy ---
if command -v mypy &>/dev/null; then
    echo "[pre-push] Running mypy..."
    if ! echo "$CHANGED_PY" | xargs mypy; then
        echo "[pre-push] mypy FAILED"
        FAILED=1
    fi
else
    echo "[pre-push] WARNING: mypy not found, skipping type check"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "[pre-push] Quality checks failed. Push blocked."
    exit 1
fi

echo "[pre-push] All checks passed."
