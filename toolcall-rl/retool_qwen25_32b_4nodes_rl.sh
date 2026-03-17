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
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((NUM_NODES * ROLLOUT_GPUS_PER_NODE))}
SGLANG_GPUS_PER_ENGINE=${SGLANG_GPUS_PER_ENGINE:-8}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
    echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
    echo "ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE}, ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
    exit 1
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
SAVE_CKPT=${SAVE_CKPT:-/data_storage/wyj/slime_export/ckpt/qwen25-32b-retool-rl}
PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/slime_export/data/dapo-math-17k/dapo-math-17k.jsonl}
EVAL_DATA=${EVAL_DATA:-/data_storage/wyj/slime_export/data/aime-2024/aime-2024.jsonl}
MEGATRON_PATH=${MEGATRON_PATH:-${SCRIPT_DIR}/../Megatron-LM}

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
   --advantage-estimator grpo
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
   --wandb-group qwen25-32b-retool-4nodes
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
     --actor-num-nodes "${NUM_NODES}" \
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
     ${CUSTOM_ARGS[@]}
else
  # Worker nodes: stay alive until the ray node is stopped.
  echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
  while ray status > /dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
