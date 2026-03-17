#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# cluster defaults
# Total Ray cluster nodes (train/policy + prm).
NUM_NODES=${NUM_NODES:-5}
# Nodes used by actor + policy rollout path.
TRAIN_NUM_NODES=${TRAIN_NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((TRAIN_NUM_NODES * ROLLOUT_GPUS_PER_NODE))}
PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL:-8}
PRM_GPUS_PER_ENGINE=${PRM_GPUS_PER_ENGINE:-8}
PRM_ENABLE=${PRM_ENABLE:-1}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:-8}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
    echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
    echo "ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE}, ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
    exit 1
fi

if (( TRAIN_NUM_NODES > NUM_NODES )); then
    echo "TRAIN_NUM_NODES must be <= NUM_NODES"
    echo "TRAIN_NUM_NODES=${TRAIN_NUM_NODES}, NUM_NODES=${NUM_NODES}"
    exit 1
fi

if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
    PRM_NUM_NODES=$((NUM_NODES - TRAIN_NUM_NODES))
    TRAIN_POLICY_GPUS_CAP=$((TRAIN_NUM_NODES * NUM_GPUS_PER_NODE))
    TRAIN_POLICY_GPUS_REQ=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL))
    PRM_GPUS_CAP=$((PRM_NUM_NODES * NUM_GPUS_PER_NODE))

    # Enforce strict non-colocate layout:
    # - train/policy fully occupy TRAIN_NUM_NODES
    # - PRM fully occupies dedicated PRM_NUM_NODES
    if (( PRM_NUM_NODES <= 0 )); then
        echo "PRM_ENABLE=1 requires dedicated PRM nodes: NUM_NODES must be > TRAIN_NUM_NODES."
        echo "NUM_NODES=${NUM_NODES}, TRAIN_NUM_NODES=${TRAIN_NUM_NODES}"
        exit 1
    fi
    if (( TRAIN_POLICY_GPUS_REQ != TRAIN_POLICY_GPUS_CAP )); then
        echo "Strict non-colocate check failed for train/policy layout."
        echo "requested=${TRAIN_POLICY_GPUS_REQ}, capacity=${TRAIN_POLICY_GPUS_CAP}"
        echo "Expected: TRAIN_NUM_NODES*ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL == TRAIN_NUM_NODES*NUM_GPUS_PER_NODE"
        exit 1
    fi
    if (( PRM_GPUS_TOTAL != PRM_GPUS_CAP )); then
        echo "Strict non-colocate check failed for PRM layout."
        echo "PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL}, dedicated_capacity=${PRM_GPUS_CAP}"
        echo "Expected: PRM_GPUS_TOTAL == (NUM_NODES-TRAIN_NUM_NODES)*NUM_GPUS_PER_NODE"
        exit 1
    fi

    TOTAL_AVAILABLE_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
    TOTAL_REQUESTED_GPUS=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL + PRM_GPUS_TOTAL))
    if (( TOTAL_REQUESTED_GPUS > TOTAL_AVAILABLE_GPUS )); then
        echo "Requested GPUs exceed cluster capacity when PRM is enabled."
        echo "requested=${TOTAL_REQUESTED_GPUS}, available=${TOTAL_AVAILABLE_GPUS}"
        echo "actor=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE)), rollout=${ROLLOUT_GPUS_TOTAL}, prm=${PRM_GPUS_TOTAL}"
        exit 1
    fi
fi

# Increase Ray heartbeat/health-check timeouts to prevent worker nodes
# from being mistakenly marked as dead during heavy initialization.
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen2.5-32B.sh"

# Override these paths via env vars if your local layout differs.
HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/ReTool-DeepSeek-R1-Distill-Qwen-32B-SFT}
REF_LOAD=${REF_LOAD:-/data_storage/wyj/systems/huggingface/hub/ReTool-DeepSeek-R1-Distill-Qwen-32B-SFT_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-/data_storage/wyj/slime_export/ckpt/qwen25-32b-retool-prm-rl}
PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/slime_export/data/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/data_storage/wyj/slime_export/data/aime-2024/aime-2024.jsonl}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-/data_storage/wyj/systems/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a}
MEGATRON_PATH=${MEGATRON_PATH:-${SCRIPT_DIR}/../Megatron-LM}

if [[ "${PRM_ENABLE:-0}" == "1" && -z "${PRM_MODEL_PATH}" ]]; then
    echo "PRM_ENABLE=1 requires PRM_MODEL_PATH to be set explicitly."
    echo "Example:"
    echo "  PRM_MODEL_PATH=/path/to/prm-model bash retool_qwen25_32b_prm_5nodes_rl.sh"
    exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 40
   --norm-epsilon 1e-6
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-reward-key acc
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator step_wise
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime_retool
   --wandb-group qwen25-32b-retool-5nodes-prm
   --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${SGLANG_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

PRM_ARGS=(
   # PRM uses framework-hosted engines + dedicated router.
   --prm-m "${PRM_M:-3}"
   --prm-num-gpus "${PRM_GPUS_TOTAL}"
   --prm-num-gpus-per-engine "${PRM_GPUS_PER_ENGINE}"
   --prm-step-coef "${PRM_STEP_COEF:-1.0}"
   --prm-temperature "${PRM_TEMPERATURE:-1.0}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-2048}"
)

if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
  PRM_ARGS+=(--prm-enable)
  PRM_ARGS+=(--prm-model-path "${PRM_MODEL_PATH}")
fi

MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  sleep 30
  ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS_PER_NODE}" --node-ip-address "${NODE_IP}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- python3 train_async.py \
     --actor-num-nodes "${TRAIN_NUM_NODES}" \
     --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
     --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${WANDB_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${SGLANG_ARGS[@]} \
     ${MISC_ARGS[@]} \
     ${CUSTOM_ARGS[@]} \
     ${PRM_ARGS[@]}
else
  # Worker nodes: stay alive until the ray node is stopped.
  echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
  while ray status > /dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
