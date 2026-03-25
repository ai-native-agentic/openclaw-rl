#!/bin/bash
#
# Standalone GUI evaluation for Qwen3-VL-4B on a single node. No actor, no training.
# sglang loads the HF checkpoint directly; all GPUs go to rollout.
#
# Usage:
#   bash gui_qwen3vl_4b_eval.sh
#   HF_CKPT=/path/to/checkpoint bash gui_qwen3vl_4b_eval.sh
#
# Key env vars:
#   HF_CKPT                      - Checkpoint to evaluate (default: Qwen3-VL-4B-Thinking)
#   NUM_GPUS                     - Total GPUs (default: 8)
#   ROLLOUT_NUM_GPUS_PER_ENGINE  - GPUs per sglang engine (default: 1)
#   GUI_POOL_MAX_ENVS            - Max concurrent VMs (default: 64)

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SCRIPT_DIR}/../Megatron-LM"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

# ---------------------------
# GUI env pool server configs
# ---------------------------
export GUI_ENV_SERVER_HOST=${GUI_ENV_SERVER_HOST:-"127.0.0.1"}
export GUI_ENV_SERVER_PORT=${GUI_ENV_SERVER_PORT:-18080}
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://${GUI_ENV_SERVER_HOST}:${GUI_ENV_SERVER_PORT}"}
export GUI_ENV_SERVER_MAX_ENVS=${GUI_ENV_SERVER_MAX_ENVS:-64}
export GUI_PREWARM_CONCURRENCY=${GUI_PREWARM_CONCURRENCY:-64}
export GUI_POOL_MAX_ENVS=${GUI_POOL_MAX_ENVS:-${GUI_ENV_SERVER_MAX_ENVS}}
export GUI_PREWARM_ENVS=${GUI_PREWARM_ENVS:-${GUI_POOL_MAX_ENVS}}
export GUI_FORCE_PREWARM_ALL=${GUI_FORCE_PREWARM_ALL:-1}
if [[ "${GUI_FORCE_PREWARM_ALL}" == "1" ]]; then
  export GUI_PREWARM_ENVS=${GUI_POOL_MAX_ENVS}
fi
export GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY:-${GUI_POOL_MAX_ENVS}}
export GUI_POOL_IDLE_TTL_SECONDS=${GUI_POOL_IDLE_TTL_SECONDS:-600}
export GUI_PROVIDER_NAME=${GUI_PROVIDER_NAME:-"volcengine"}
export GUI_REGION=${GUI_REGION:-"cn-beijing"}
export GUI_PATH_TO_VM=${GUI_PATH_TO_VM:-""}
export GUI_ACTION_SPACE=${GUI_ACTION_SPACE:-"pyautogui"}
export GUI_OBSERVATION_TYPE=${GUI_OBSERVATION_TYPE:-"screenshot"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen3vl_agent.Qwen3VLAgentLocal"}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
export GUI_REUSE_VM_ON_RESET=${GUI_REUSE_VM_ON_RESET:-0}
export GUI_RESET_ON_CLOSE=${GUI_RESET_ON_CLOSE:-1}
export GUI_CLIENT_PASSWORD=${GUI_CLIENT_PASSWORD:-"WWbbb8b7b6314"}
export GUI_SCREEN_WIDTH=${GUI_SCREEN_WIDTH:-1920}
export GUI_SCREEN_HEIGHT=${GUI_SCREEN_HEIGHT:-1080}
WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui_4b_eval}
export OSWORLD_PROJECT="${GUI_PROJECT_NAME}"
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_nochrome.json"}

if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
  rm -rf "${GUI_RESULT_DIR}"
fi
mkdir -p "${GUI_RESULT_DIR}"

# ---------------------------
# Volcengine non-secret configs
# ---------------------------
export VOLCENGINE_REGION=${VOLCENGINE_REGION:-"cn-beijing"}
export VOLCENGINE_IMAGE_ID=${VOLCENGINE_IMAGE_ID:-"image-ye7p6rwcel9q7e1x1evg"}
export VOLCENGINE_SUBNET_ID=${VOLCENGINE_SUBNET_ID:-"subnet-mjddx72qnklc5smt1b752ytw"}
export VOLCENGINE_SECURITY_GROUP_ID=${VOLCENGINE_SECURITY_GROUP_ID:-"sg-3hirj05a13v9c3nkipjb1mhm5"}
export VOLCENGINE_ZONE_ID=${VOLCENGINE_ZONE_ID:-"cn-beijing-a"}
export VOLCENGINE_DEFAULT_PASSWORD=${VOLCENGINE_DEFAULT_PASSWORD:-"WWbbb180314"}
export VOLCENGINE_RUNINST_MIN_INTERVAL=${VOLCENGINE_RUNINST_MIN_INTERVAL:-0.1}
export VOLCENGINE_DELINST_MIN_INTERVAL=${VOLCENGINE_DELINST_MIN_INTERVAL:-0.1}
export VOLCENGINE_INSTANCE_TYPE=${VOLCENGINE_INSTANCE_TYPE:-"ecs.e-c1m2.large,ecs.e-c1m4.large,ecs.e-c1m8.large,ecs.e-c1m1.large,ecs.c3al.large,ecs.c3a.large,ecs.c3il.large,ecs.g3il.large,ecs.r3il.large,ecs.c3a.large,ecs.g3a.large,ecs.r3a.large,ecs.c3i.large,ecs.g3i.large,ecs.r3i.large,ecs.g3al.large,ecs.r3al.large,ecs.r1ie.large,ecs.g1ie.large,ecs.c1ie.large,ecs.g3ine.large"}
export download_proxy=${download_proxy:-}

HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen3-VL-4B-Thinking}
echo "Evaluating checkpoint: ${HF_CKPT}"

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-4b-gui-eval
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

# Start GUI env pool server
mkdir -p logs
ENV_SERVER_LOG=${ENV_SERVER_LOG:-"./logs/gui_env_pool_server_4b_eval.log"}
PYTHONPATH="${SLIME_DIR}:${SCRIPT_DIR}:${PYTHONPATH}" \
  python3 -m env_pool_server \
  --host "${GUI_ENV_SERVER_HOST}" \
  --port "${GUI_ENV_SERVER_PORT}" \
  --max-envs "${GUI_POOL_MAX_ENVS}" \
  --prewarm-envs "${GUI_PREWARM_ENVS}" \
  --prewarm-concurrency "${GUI_PREWARM_CONCURRENCY}" \
  --idle-ttl-seconds "${GUI_POOL_IDLE_TTL_SECONDS}" \
  --provider-name "${GUI_PROVIDER_NAME}" \
  --region "${GUI_REGION}" \
  --action-space "${GUI_ACTION_SPACE}" \
  --observation-type "${GUI_OBSERVATION_TYPE}" \
  --reset-on-close "${GUI_RESET_ON_CLOSE}" \
  --client-password "${GUI_CLIENT_PASSWORD}" \
  --screen-width "${GUI_SCREEN_WIDTH}" \
  --screen-height "${GUI_SCREEN_HEIGHT}" \
  > "${ENV_SERVER_LOG}" 2>&1 &
GUI_ENV_SERVER_PID=$!
echo "GUI env pool server PID=${GUI_ENV_SERVER_PID}, log=${ENV_SERVER_LOG}"

cleanup() {
  set +e
  if [[ -n "${GUI_ENV_SERVER_PID}" ]] && kill -0 "${GUI_ENV_SERVER_PID}" 2>/dev/null; then
    kill "${GUI_ENV_SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

for i in {1..60}; do
  if curl -fsS "${GUI_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "GUI env pool server is ready: ${GUI_ENV_SERVER_URL}"
    break
  fi
  sleep 2
done

if (( GUI_PREWARM_ENVS > 0 )); then
  for i in {1..600}; do
    if python3 - "${GUI_ENV_SERVER_URL}" "${GUI_PREWARM_ENVS}" <<'PY'
import json
import sys
import urllib.request

status_url = sys.argv[1].rstrip("/") + "/status"
target = int(sys.argv[2])
with urllib.request.urlopen(status_url, timeout=5) as resp:
    data = json.loads(resp.read().decode("utf-8"))
pool = data.get("pool", {})
total_envs = int(pool.get("total_envs", 0))
ok = bool(data.get("ok", False))
print(f"pool total_envs={total_envs}, target={target}, ok={ok}")
raise SystemExit(0 if ok and total_envs >= target else 1)
PY
    then
      echo "GUI prewarm complete: ${GUI_PREWARM_ENVS}/${GUI_POOL_MAX_ENVS}"
      break
    fi
    sleep 2
    if (( i == 600 )); then
      echo "Timed out waiting prewarm completion: target=${GUI_PREWARM_ENVS}"
      exit 1
    fi
  done
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"GUI_ENV_SERVER_URL\": \"${GUI_ENV_SERVER_URL}\",
    \"GUI_POOL_MAX_ENVS\": \"${GUI_POOL_MAX_ENVS}\",
    \"GUI_TRAJECTORY_CONCURRENCY\": \"${GUI_TRAJECTORY_CONCURRENCY}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_ACTION_SPACE\": \"${GUI_ACTION_SPACE}\",
    \"GUI_OBSERVATION_TYPE\": \"${GUI_OBSERVATION_TYPE}\",
    \"GUI_REUSE_VM_ON_RESET\": \"${GUI_REUSE_VM_ON_RESET}\",
    \"GUI_TEST_CONFIG_BASE_DIR\": \"${GUI_TEST_CONFIG_BASE_DIR}\",
    \"GUI_TRAIN_META_PATH\": \"${GUI_TRAIN_META_PATH}\",
    \"GUI_EVAL_META_PATH\": \"${GUI_EVAL_META_PATH}\",
    \"OSWORLD_PROJECT\": \"${OSWORLD_PROJECT}\",
    \"download_proxy\": \"${download_proxy}\"
  }
}"

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_4b_eval_$(date +%Y%m%d_%H%M%S)"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u eval_only.py \
  --debug-rollout-only \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --rollout-num-gpus ${NUM_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  --hf-checkpoint "${HF_CKPT}" \
  --data-source-path gui_data_source.GuiMetaDataSource \
  --reward-key score \
  --num-rollout 100 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --global-batch-size 1 \
  --rollout-max-response-len 1024 \
  --gui-max-steps 30 \
  --gui-wait-after-reset 60 \
  --gui-max-image-history-length 3 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --eval-temperature 0.0 \
  --gui-eval-max-steps 30 \
  --gui-eval-sleep-after-execution 5.0 \
  --gui-eval-wait-after-reset 60 \
  --n-samples-per-eval-prompt 1 \
  --eval-interval 1 \
  --eval-reward-key acc \
  --eval-function-path generate_with_gui.gui_generate_rollout \
  --custom-generate-function-path generate_with_gui.generate \
  --custom-rm-path generate_with_gui.reward_func \
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE} \
  --sglang-mem-fraction-static 0.85 \
  ${MODEL_ARGS[@]} \
  ${WANDB_ARGS[@]}

echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID}"
set +e
ray job logs --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi

echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
