#!/bin/bash

# for rerun the task (same style as retool scripts)
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
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/data_storage/wyj/slime_export/Megatron-LM}

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# reduce false ray node death under heavy initialization
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# default to 8 GPUs if not set by scheduler
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

# ---------------------------
# GUI env pool server configs
# ---------------------------
export GUI_ENV_SERVER_HOST=${GUI_ENV_SERVER_HOST:-"127.0.0.1"}
export GUI_ENV_SERVER_PORT=${GUI_ENV_SERVER_PORT:-18080}
export GUI_ENV_SERVER_URL=${GUI_ENV_SERVER_URL:-"http://${GUI_ENV_SERVER_HOST}:${GUI_ENV_SERVER_PORT}"}
export GUI_ENV_SERVER_MAX_ENVS=${GUI_ENV_SERVER_MAX_ENVS:-32}
export GUI_PREWARM_CONCURRENCY=${GUI_PREWARM_CONCURRENCY:-32}
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
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"examples.gui.agents.qwen3vl_agent.Qwen3VLAgentLocal"}
MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
export GUI_REUSE_VM_ON_RESET=${GUI_REUSE_VM_ON_RESET:-0}
export GUI_RESET_ON_CLOSE=${GUI_RESET_ON_CLOSE:-1}
export GUI_CLIENT_PASSWORD=${GUI_CLIENT_PASSWORD:-"WWbbb8b7b6314"}
export GUI_SCREEN_WIDTH=${GUI_SCREEN_WIDTH:-1920}
export GUI_SCREEN_HEIGHT=${GUI_SCREEN_HEIGHT:-1080}
# Keep wandb project independent from OSWORLD/result project naming.
WANDB_PROJECT=${WANDB_PROJECT:-slime_gui}
GUI_PROJECT_NAME=${GUI_PROJECT_NAME:-slime_gui}
export OSWORLD_PROJECT="${GUI_PROJECT_NAME}"
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"/data_storage/wyj/slime_export/slime/examples/gui/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"
export GUI_TEST_CONFIG_BASE_DIR=${GUI_TEST_CONFIG_BASE_DIR:-"${SCRIPT_DIR}/evaluation_examples"}
export GUI_TRAIN_META_PATH=${GUI_TRAIN_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/train_nochrome.json"}
export GUI_EVAL_META_PATH=${GUI_EVAL_META_PATH:-"${GUI_TEST_CONFIG_BASE_DIR}/test_multinode.json"}



# Start each run with a clean project result directory.
if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
  rm -rf "${GUI_RESULT_DIR}"
fi
mkdir -p "${GUI_RESULT_DIR}"



# ---------------------------
# Volcengine non-secret configs
# Secrets (AK/SK) should be exported in shell before running this script.
# ---------------------------
export VOLCENGINE_REGION=${VOLCENGINE_REGION:-"cn-beijing"}
export VOLCENGINE_IMAGE_ID=${VOLCENGINE_IMAGE_ID:-"image-ye7p6rwcel9q7e1x1evg"}
export VOLCENGINE_SUBNET_ID=${VOLCENGINE_SUBNET_ID:-"subnet-mjddx72qnklc5smt1b752ytw"}
export VOLCENGINE_SECURITY_GROUP_ID=${VOLCENGINE_SECURITY_GROUP_ID:-"sg-3hirj05a13v9c3nkipjb1mhm5"}
export VOLCENGINE_ZONE_ID=${VOLCENGINE_ZONE_ID:-"cn-beijing-a"}
export VOLCENGINE_DEFAULT_PASSWORD=${VOLCENGINE_DEFAULT_PASSWORD:-"WWbbb180314"}
# Lower interval speeds up create/delete under moderate throttling.
export VOLCENGINE_RUNINST_MIN_INTERVAL=${VOLCENGINE_RUNINST_MIN_INTERVAL:-0.1}
export VOLCENGINE_DELINST_MIN_INTERVAL=${VOLCENGINE_DELINST_MIN_INTERVAL:-0.1}

# Instance type fallback list (comma-separated).
# desktop_env/providers/volcengine/manager.py already supports this format.
export VOLCENGINE_INSTANCE_TYPE=${VOLCENGINE_INSTANCE_TYPE:-"ecs.e-c1m2.large,ecs.e-c1m4.large,ecs.e-c1m8.large,ecs.e-c1m1.large,ecs.c3al.large,ecs.c3a.large,ecs.c3il.large,ecs.g3il.large,ecs.r3il.large,ecs.c3a.large,ecs.g3a.large,ecs.r3a.large,ecs.c3i.large,ecs.g3i.large,ecs.r3i.large,ecs.g3al.large,ecs.r3al.large,ecs.r1ie.large,ecs.g1ie.large,ecs.c1ie.large,ecs.g3ine.large"}

# Optional download proxy used by setup/evaluator code.
# Do not put a default value here; export it in terminal when needed (same as keys).
export download_proxy=${download_proxy:-}

HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18}
# For bridge+multimodal path, prefer HF checkpoint path for ref model loading.
REF_LOAD=${REF_LOAD:-${HF_CKPT}}

CKPT_ARGS=(
  --hf-checkpoint ${HF_CKPT}
  --ref-load ${REF_LOAD}
  --save ${SAVE_CKPT:-/data_storage/wyj/slime_export/ckpt/gui-qwen3vl-8b-thinking-rl}
  --save-interval 20
)

ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
RESUME_LOAD=${RESUME_LOAD:-/data_storage/wyj/slime_export/ckpt/gui-qwen3vl-8b-thinking-rl}
if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
  echo "Resume load enabled: ${RESUME_LOAD}"
else
  echo "Resume load disabled (ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD})"
fi

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}

ROLLOUT_ARGS=(
  --data-source-path examples.gui.gui_data_source.GuiMetaDataSource
  --reward-key score
  --num-rollout 1000
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 512
  --rollout-temperature 1.0
  --gui-max-steps 30
  --gui-wait-after-reset 60
  --gui-sleep-after-execution 0.0
  --gui-max-image-history-length 3
  --gui-max-reward-image-history-length 2
  --num-steps-per-rollout 2
)

ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"
echo "GUI_TRAJECTORY_CONCURRENCY=${GUI_TRAJECTORY_CONCURRENCY} (independent from sglang concurrency)"

EVAL_ARGS=(
  --eval-temperature 0.0
  --gui-eval-max-steps 30
  --gui-eval-sleep-after-execution 0.0
  --gui-eval-wait-after-reset 60
  --n-samples-per-eval-prompt 1
)
if [ -n "${GUI_EVAL_INTERVAL}" ]; then
  EVAL_ARGS+=(--eval-interval "${GUI_EVAL_INTERVAL}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  # VLM train forward in current Megatron bridge expects unsliced vision inputs.
  # CP>1 slices text tokens per rank but does not slice multimodal tensors here,
  # which can trigger "assert video_embeds is None" with image-only data.
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 36
  --megatron-to-hf-mode bridge
  --use-dynamic-batch-size
  --max-tokens-per-gpu 512
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.0
)

#ROUTER_ARGS=(
  # Use SlimeRouter's least-active-worker routing to avoid over-pinning
  # GUI requests with similar prefixes to a single SGLang worker.
  #--use-slime-router
#)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.85
)

CUSTOM_ARGS=(
  --custom-generate-function-path examples.gui.generate_with_gui.generate
  --custom-rm-path examples.gui.generate_with_gui.reward_func
)

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-8b-rl
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

# Start GUI env pool server
mkdir -p logs
ENV_SERVER_LOG=${ENV_SERVER_LOG:-"./logs/gui_env_pool_server.log"}
PYTHONPATH="${SLIME_DIR}:${SCRIPT_DIR}:${PYTHONPATH}" \
  python3 -m examples.gui.env_pool_server \
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

# Wait until env server is healthy
for i in {1..60}; do
  if curl -fsS "${GUI_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "GUI env pool server is ready: ${GUI_ENV_SERVER_URL}"
    break
  fi
  sleep 2
done

# Block until all prewarm envs are visible in the pool.
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048

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

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen3vl_$(date +%Y%m%d_%H%M%S)"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${ROUTER_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

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

