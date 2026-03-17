#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

cd "${REPO_ROOT}"

export DATASET_DIR="${DATASET_DIR:-${TERMINAL_RL}/dataset}"
export TBENCH_OUTPUT_ROOT="${TBENCH_OUTPUT_ROOT:-${TERMINAL_RL}/build_outputs}"

export TBENCH_DOCKER_IMAGE_SOURCE="${TBENCH_DOCKER_IMAGE_SOURCE:-build}"
export TBENCH_DOCKER_PULL_PREFIX="${TBENCH_DOCKER_PULL_PREFIX:-}"
export COMPOSE_OVERRIDE_PATH="${COMPOSE_OVERRIDE_PATH:-}"

if [ -d "${REPO_ROOT}/.venv" ]; then
  source .venv/bin/activate
fi

# Start the pool server
exec python -m terminal-rl.remote.pool_server \
  --host 0.0.0.0 \
  --port "${ENV_SERVER_PORT:-18081}" \
  --max-tasks "${WORKER_MAX_TASKS:-16}" \
  --max-runs-per-task "${WORKER_MAX_RUNS_PER_TASK:-8}" \
  --output-root "${TBENCH_OUTPUT_ROOT}"
