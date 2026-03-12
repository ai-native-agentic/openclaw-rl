#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="openclaw-rl"
INCLUDE_PATHS=("openclaw")
TEST_PATH=""

GREEN='[0;32m'
RED='[0;31m'
YELLOW='[1;33m'
NC='[0m'
PASS=0
FAIL=0
SKIP=0

run_gate() {
  local label="$1"
  local cmd="$2"
  printf "  %-20s ... " "$label"
  if (cd "$PROJECT_ROOT" && eval "$cmd") >/dev/null 2>&1; then
    printf "${GREEN}PASS${NC}
"
    PASS=$((PASS + 1))
  else
    printf "${RED}FAIL${NC}
"
    FAIL=$((FAIL + 1))
  fi
}

skip_gate() {
  local label="$1"
  local reason="$2"
  printf "  %-20s ... %sSKIP%s (%s)
" "$label" "$YELLOW" "$NC" "$reason"
  SKIP=$((SKIP + 1))
}

echo ""
echo "=== $PROJECT_NAME QA Gates ==="
echo ""

compile_cmd="python3 -m compileall -q"
for path in "${INCLUDE_PATHS[@]}"; do
  compile_cmd+=" '$path'"
done
run_gate "syntax" "$compile_cmd"

if command -v ruff >/dev/null 2>&1 && grep -q '^\[tool\.ruff' "$PROJECT_ROOT/pyproject.toml" 2>/dev/null; then
  ruff_cmd="ruff check"
  for path in "${INCLUDE_PATHS[@]}"; do
    ruff_cmd+=" '$path'"
  done
  run_gate "ruff" "$ruff_cmd"
else
  skip_gate "ruff" "ruff not configured"
fi

if [[ -n "$TEST_PATH" && -e "$PROJECT_ROOT/$TEST_PATH" ]]; then
  if python3 -c 'import pytest' >/dev/null 2>&1; then
    run_gate "pytest" "pytest '$TEST_PATH' -q"
  else
    skip_gate "pytest" "pytest not installed"
  fi
else
  skip_gate "pytest" "no tests configured"
fi

echo ""
echo "=== Results ==="
echo -e "  ${GREEN}PASS: $PASS${NC}  ${YELLOW}SKIP: $SKIP${NC}  ${RED}FAIL: $FAIL${NC}"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
