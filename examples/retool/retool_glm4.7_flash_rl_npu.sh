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
ulimit -n 65535

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEBUG=1
export RAY_DEDUP_LOGS=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/glm4.7-30B-A3B.sh

CKPT_ARGS=(
   --hf-checkpoint /path/to/models/GLM-4.7-Flash
   --ref-load /path/to/models/GLM-4.7-Flash_torch_dist
   --load /path/to/models/GLM-4.7-Flash/
   --save /path/to/models/GLM-4.7-Flash/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data path/to/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len $((1024 * 7))
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime  /path/to/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $((1024 * 4))
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
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-cuda-graph-max-bs 16
   --sglang-max-running-requests 64
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size -1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --use-flash-attn
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/path/to/Megatron-Bridge/src:/path/to/Megatron-LM/:/path/to/sglang/python:/path/to/slime-ascend:${SCRIPT_DIR}:${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"ASCEND_TOOLKIT_HOME\": \"/path/to/cann/\",
    \"ASCEND_OPP_PATH\": \"/path/to/cann/opp\",
    \"ASCEND_AICPU_PATH\": \"/path/to/cann/\",
    \"ASCEND_HOME_PATH\": \"/path/to/cann/\"
  }
}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} | tee logs/retool-glm-4.7-30B-A3B-16p.log

####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

