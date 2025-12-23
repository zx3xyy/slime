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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

export BASE_DIR=/home/chengze/work/mast

CKPT_ARGS=(
   --hf-checkpoint ${BASE_DIR}/slime_data/GLM-Z1-9B-0414/
   --ref-load ${BASE_DIR}/slime_data/GLM-Z1-9B-0414_torch_dist
   --load ${BASE_DIR}/slime_data/GLM-Z1-9B-0414_slime/
   --save ${BASE_DIR}/slime_data/GLM-Z1-9B-0414_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_DIR}/slime_data/train_data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${BASE_DIR}/slime_data/eval_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
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
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend fused
   # --attention-backend flash
)

# launch the master node of ray in container
# Use IPv6 - let Ray auto-detect the node IP
ray start --head # --node-ip-address=127.0.0.2 --dashboard-host=0.0.0.0

# Get the node IP that Ray detected (IPv6) from ray start output
# Parse from: "Local node IP: 2401:db00:23c:311f:face:0:143:0"
NODE_IP=$(cat /tmp/ray/session_latest/logs/raylet.out 2>/dev/null | grep -oP 'node_ip_address=\K[0-9a-f:]+' | head -1)
# Fallback: try to get from hostname
if [[ -z "$NODE_IP" ]]; then
    NODE_IP=$(python3 -c "import socket; print(socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6)[0][4][0])" 2>/dev/null)
fi
# Fallback: parse from /proc/net/if_inet6
if [[ -z "$NODE_IP" ]]; then
    NODE_IP=$(cat /proc/net/if_inet6 | grep -v "^00000000000000000000000000000001" | head -1 | awk '{print $1}' | sed 's/.\{4\}/&:/g;s/:$//' | sed 's/0000/0/g')
fi
# Final fallback: hardcode based on what we saw
if [[ -z "$NODE_IP" ]]; then
    NODE_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i ~ /:/) print $i}' | head -1)
fi
echo "Detected Node IP: $NODE_IP"

export PYTHONPATH="${BASE_DIR}/sglang/python:${BASE_DIR}/Megatron-LM:${BASE_DIR}/slime:$PYTHONPATH"

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  },
  \"excludes\": [\".git\"]
}"

cd "${SLIME_DIR}"

# export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit \
   --no-wait \
   --address=http://127.0.0.1:8265 \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${BASE_DIR}/slime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 3 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
