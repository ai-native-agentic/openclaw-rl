#!/usr/bin/env bash
set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] missing env: ${name}"
    exit 1
  fi
}

NUM_NODES=${NUM_NODES:-2}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL:-8}
PRM_GPUS_PER_ENGINE=${PRM_GPUS_PER_ENGINE:-1}

PRM_ENABLE="${PRM_ENABLE:-1}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-}"
PRM_TEMPERATURE="${PRM_TEMPERATURE:-0.0}"
PRM_MAX_NEW_TOKENS="${PRM_MAX_NEW_TOKENS:-4096}"
PRM_M="${PRM_M:-1}"
PRM_STEP_COEF="${PRM_STEP_COEF:-1.0}"
PRM_SGLANG_URL="${PRM_SGLANG_URL:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export REPO_ROOT

SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${REPO_ROOT}/Megatron-LM}"

export SLIME_PKG_DIR="${REPO_ROOT}/slime"
export MEGATRON_DIR="${REPO_ROOT}/Megatron-LM"

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# Paths: set/export before running (no built-in defaults).
HF_HOME="${HF_HOME:-}"
HF_CKPT="${HF_CKPT:-}"
REF_LOAD="${REF_LOAD:-}"
SAVE_CKPT="${SAVE_CKPT:-}"
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"
ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"


MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
HEAD_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"

unset MASTER_ADDR
export no_proxy="127.0.0.1,${HEAD_ADDR}"
log "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, HEAD_ADDR=${HEAD_ADDR}, NODE_IP=${NODE_IP}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-1}"
export PROVIDER_NAME="${PROVIDER_NAME:-pull}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${HEAD_ADDR}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"

export RAY_TMPDIR=/tmp/ray_${MLP_ROLE_INDEX}

export WORKER_URLS="${WORKER_URLS:-}"

ROUTER_SESSION_NAME="${ROUTER_SESSION_NAME:-terminal_router}"
ROUTER_CONDA_ENV_PATH="${ROUTER_CONDA_ENV_PATH:-}"
ROUTER_PROJECT_DIR="${ROUTER_PROJECT_DIR:-${REPO_ROOT}}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"

CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"
ROUTER_RESTART="${ROUTER_RESTART:-1}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --load "${RESUME_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 3
  --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data "${ROLLOUT_PROMPT_DATA}"
   --input-key task
   --rollout-shuffle
   --reward-key score
   --num-rollout 2000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-max-context-len 16384
   --rollout-temperature 1

   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
  --advantage-estimator step_wise
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type k3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project slime
    --wandb-group qwen3-8B-prm-2nodes-rl_terminal
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate.generate
   --custom-rollout-log-function-path rollout_log.rollout_log
)

PRM_ARGS=(
  --prm-m "${PRM_M}"
  --prm-temperature "${PRM_TEMPERATURE}"
  --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS}"
  --prm-num-gpus "${PRM_GPUS_TOTAL}"
  --prm-num-gpus-per-engine "${PRM_GPUS_PER_ENGINE}"
  --prm-step-coef "${PRM_STEP_COEF}"
)

if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
  PRM_ARGS+=(--prm-enable)
  PRM_ARGS+=(--prm-model-path "${PRM_MODEL_PATH}")
fi
if [[ -n "${PRM_SGLANG_URL}" ]]; then
  PRM_ARGS+=(--prm-sglang-url "${PRM_SGLANG_URL}")
fi

check_gpus() {
  local total_available=$((NUM_NODES * NUM_GPUS_PER_NODE))
  local total_requested=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL + PRM_GPUS_TOTAL))
  if (( total_requested > total_available )); then
    echo "Requested GPUs exceed cluster capacity."
    echo "requested=${total_requested}, available=${total_available}"
    echo "actor=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE)), rollout=${ROLLOUT_GPUS_TOTAL}, prm=${PRM_GPUS_TOTAL}"
    exit 1
  fi
  if [[ "${PRM_ENABLE:-0}" == "1" ]] && (( PRM_GPUS_TOTAL <= 0 )); then
    echo "PRM_ENABLE=1 but PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL} <= 0"
    exit 1
  fi
}

cleanup_prev() {
  log "cleanup previous processes"
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
}

start_router() {
  if [[ ${MLP_ROLE_INDEX} -ne 0 ]]; then
    log "worker node ${MLP_ROLE_INDEX}, skip start_router"
    return 0
  fi

  require_cmd curl
  mkdir -p "${ROUTER_PROJECT_DIR}/logs"
  local logf="${ROUTER_PROJECT_DIR}/logs/router_${ROUTER_PORT}_2nodes.log"

  "${ROUTER_CONDA_ENV_PATH}/bin/python" -m terminal-rl.router_server \
    --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
    > "${logf}" 2>&1 &

  export ROUTER_PID=$!
  log "router started pid=${ROUTER_PID}, log=${logf}"

  trap 'set +e; kill "${ROUTER_PID}" 2>/dev/null || true' EXIT INT TERM

  sleep 1
  tail -n 50 "${logf}" || true
}

check_router() {
  if [[ ${MLP_ROLE_INDEX} -ne 0 ]]; then
    log "worker node ${MLP_ROLE_INDEX}, skip check_router"
    return 0
  fi

  require_cmd curl
  local base_url="http://${CHECK_HOST}:${ROUTER_PORT}"

  log "wait router healthz up to ${CHECK_WAIT_SECS}s: ${base_url}/healthz"
  for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
    if curl -fsS "${base_url}/healthz" >/dev/null 2>&1; then
      log "router is up"
      break
    fi
    sleep 1
  done

  log "curl ${base_url}/status"
  curl -sS "${base_url}/status"
  echo
  log "curl ${base_url}/healthz"
  curl -sS "${base_url}/healthz"
  echo
}

detect_nvlink() {
  local count
  count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
  if [[ "${count:-0}" -gt 0 ]]; then
    export HAS_NVLINK=1
  else
    export HAS_NVLINK=0
  fi
  log "HAS_NVLINK=${HAS_NVLINK} (detected ${count} NVLink references)"
}

maybe_fill_env_server_url() {
  if [[ "${USE_REMOTE_ENV}" == "1" && -z "${ENV_SERVER_URL}" ]]; then
    export ENV_SERVER_URL="http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}"
    if [[ "${START_ENV_POOL_SERVER}" == "0" ]]; then
      export START_ENV_POOL_SERVER=1
    fi
  fi
  log "ENV_SERVER_URL=${ENV_SERVER_URL} START_ENV_POOL_SERVER=${START_ENV_POOL_SERVER}"
}

start_ray_head() {
  require_cmd ray
  mkdir -p "${RAY_TMPDIR}"

  if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
    log "start ray head on node 0"
    ray start --head \
      --node-ip-address "${NODE_IP}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265 \
      --temp-dir "${RAY_TMPDIR}"
  else
    log "worker node ${MLP_ROLE_INDEX}: wait 30s then join ray cluster at ${HEAD_ADDR}:6379"
    sleep 30
    ray start \
      --address="${HEAD_ADDR}:6379" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --node-ip-address "${NODE_IP}" \
      --temp-dir "${RAY_TMPDIR}"
  fi
}

build_runtime_env_json() {
  python3 - <<'PY'
import json, os

parts = [
  os.environ.get("REPO_ROOT",""),
  os.environ.get("SLIME_PKG_DIR",""),
  os.environ.get("MEGATRON_DIR",""),
  os.environ.get("SCRIPT_DIR",""),
]
pythonpath = ":".join([p for p in parts if p])

env_vars = {
  "PYTHONPATH": pythonpath,
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK","0"),
  "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",""),
  "USE_REMOTE_ENV": os.environ.get("USE_REMOTE_ENV","0"),
  "ENV_SERVER_URL": os.environ.get("ENV_SERVER_URL",""),
  "PRM_SGLANG_URL": os.environ.get("PRM_SGLANG_URL",""),
}
print(json.dumps({"env_vars": env_vars}))
PY
}

submit_job() {
  require_env HF_CKPT
  require_env REF_LOAD
  require_env SAVE_CKPT
  require_env ROLLOUT_PROMPT_DATA
  if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
    require_env PRM_MODEL_PATH
  fi

  if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
    log "submit ray job (head node)"
    local runtime_env_json
    runtime_env_json="$(build_runtime_env_json)"
    local submission_id="${RAY_JOB_SUBMISSION_ID:-terminal_qwen3_8b_prm_2nodes_$(date +%Y%m%d_%H%M%S)}"

    ray job submit --address="http://${HEAD_ADDR}:8265" \
      --submission-id "${submission_id}" \
      --no-wait \
      --runtime-env-json="${runtime_env_json}" \
      -- python3 -u ${SLIME_DIR}/train_async.py \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
      --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      "${CUSTOM_ARGS[@]}" \
      "${PRM_ARGS[@]}"

    log "Following live Ray logs for ${submission_id}"
    set +e
    ray job logs --address="http://${HEAD_ADDR}:8265" "${submission_id}" -f --log-style=record
    local ray_log_exit=$?
    local ray_status
    ray_status=$(ray job status --address="http://${HEAD_ADDR}:8265" "${submission_id}" --log-style=record 2>&1)
    echo "${ray_status}"
    set -e

    if [[ "${ray_status}" == *"SUCCEEDED"* ]]; then
      exit 0
    fi

    echo "Ray job failed (submission id: ${submission_id}, logs exit: ${ray_log_exit})"
    exit 1
  else
    log "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
    while ray status > /dev/null 2>&1; do
      sleep 60
    done
    log "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
  fi
}

cleanup_prev

start_router
check_router

check_gpus
detect_nvlink
maybe_fill_env_server_url
export SCRIPT_DIR
start_ray_head
submit_job
