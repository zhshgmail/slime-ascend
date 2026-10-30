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
export PYTHONPATH="/path/to/Megatron-LM/:/path/to/sglang/python/:/path/to/MindSpeed/:/path/to/Megatron-Bridge/src/:$PYTHONPATH"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export MULTI_STREAM_MEMORY_REUSE=2
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEBUG=1
export RAY_DEDUP_LOGS=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export LOG_FILE=./logs/train_log_$(date +%Y-%m-%d_%H-%M-%S)_glm47_128k.log

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4.7-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /path/to/models/GLM-4.7-Flash
   --ref-load /path/to/models/GLM-4.7-Flash_torch_dist
   --load /path/to/models/GLM-4.7-Flash_slime/
   --save /path/to/models/GLM-4.7-Flash_slime/
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/datasets/processed_code_debug.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len $((1024 * 32))
   --rollout-temperature 1
   --rollout-max-prompt-len $((1024 * 128))
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /path/to/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --context-parallel-algo megatron_cp_algo
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group glm4.7-flash
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.65
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-cuda-graph-max-bs 16
   --sglang-max-running-requests 64
   
   # ======================= NPU 添加参数 =======================
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 8192
   --sglang-max-prefill-tokens 8192
   --sglang-device npu
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
   --no-gradient-accumulation-fusion
)

NNODES=2
NPUS_PER_NODE=16
# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

# 获取当前IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
   ray start --head --port 6379 --dashboard-host="0.0.0.0" --node-ip-address=$CURRENT_IP --dashboard-port=8265 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
            python3 train.py \
            --actor-num-nodes 1 \
            --actor-num-gpus-per-node 16 \
            --rollout-num-gpus 16 \
            --num-gpus-per-node 16 \
            --save-debug-rollout-data /path/to/exp2_rollout_{rollout_id}.pt \
            ${MODEL_ARGS[@]} \
            ${ROLLOUT_ARGS[@]} \
            ${OPTIMIZER_ARGS[@]} \
            ${GRPO_ARGS[@]} \
            ${WANDB_ARGS[@]} \
            ${PERF_ARGS[@]} \
            ${SGLANG_ARGS[@]} \
            ${MISC_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${EVAL_ARGS[@]} \
            2>&1 | tee "${LOG_FILE}"
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done


else
  # 子节点尝试往主节点注册 ray 直到成功
  while true; do
      # 尝试连接 ray 集群
      ray start --address="$MASTER_ADDR:6379" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi