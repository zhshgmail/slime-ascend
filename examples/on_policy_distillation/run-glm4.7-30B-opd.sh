#!/bin/bash

# usage: bash examples/on_policy_distillation/run-glm4.7-30B-opd.sh

set -ex
ulimit -n 65535

# Start the teacher model server
TEACHER_IP="127.0.0.1" # Use localhost here, you can change it to your IP
TEACHER_PORT=13141
LOCAL_IP=$(hostname -I | awk '{print $1}')

if [ "$LOCAL_IP" == "$TEACHER_IP" ]; then
   LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   ## Launch the teacher model server in the background
   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
       --model-path /root/Qwen3-32B \
       --host 0.0.0.0 \
       --port $TEACHER_PORT \
       --tp 4 \
       --chunked-prefill-size 4096 \
       --mem-fraction-static 0.8 \
       > "$LOG_FILE" 2>&1 &

   echo "Starting teacher model server..."
   tail -f -n 20 "$LOG_FILE"
else
   ## Wait for the teacher model server to be ready
   until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
       echo "Waiting for the teacher model server to start..."
       sleep 15
   done

   curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
   echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
   sleep 10

   export PYTHONBUFFERED=16

   source "/root/slime/scripts/models/glm4.7-30B-A3B.sh"

   CKPT_ARGS=(
      --hf-checkpoint /root/GLM-4.7-FLASH
      --ref-load /root/GLM-4.7-FLASH_torch_dist
      --load /root/GLM-4.7-FLASH_slime/
      --save /root/GLM-4.7-FLASH_slime/
      --save-interval 20
   )

   ROLLOUT_ARGS=(
      --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
      --input-key prompt
      --apply-chat-template
      --rollout-shuffle
      --num-rollout 300
      --rollout-batch-size 16
      --n-samples-per-prompt 4
      --rollout-max-response-len 4096
      --rollout-temperature 1

      --global-batch-size 64
      --balance-data
   )

   RM_ARGS=(
      --custom-rm-path slime.rollout.on_policy_distillation.reward_func
      --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
      --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
   )

   PERF_ARGS=(
      --tensor-model-parallel-size 1
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 8
      --expert-tensor-parallel-size 1

      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1

      --no-gradient-accumulation-fusion
      --use-dynamic-batch-size
      --max-tokens-per-gpu 8192
   )

   GRPO_ARGS=(
      --advantage-estimator grpo
      --use-opd
      --opd-type sglang
      --opd-kl-coef 1.0
      --use-kl-loss
      --kl-loss-coef 0.00
      --kl-loss-type low_var_kl
      --entropy-coef 0.00
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
      # --wandb-group qwen3-8B-test
      # --wandb-key ${WANDB_KEY}
   )

   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.8
      --sglang-enable-dp-attention
      --sglang-dp-size 8
      --sglang-enable-dp-lm-head
      --sglang-moe-dense-tp-size 1

      --sglang-cuda-graph-max-bs 16
      --sglang-max-running-requests 64

      --sglang-disable-radix-cache
      --sglang-chunked-prefill-size -1
      --sglang-device npu
      --sglang-log-level debug
   )


   MISC_ARGS=(
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash

      --moe-token-dispatcher-type alltoall
   )

   # launch the master node of ray in container
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=60000-60050
   export HYDRA_FULL_ERROR=1
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
   ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{
        "env_vars": {
           "PYTHONPATH": "/path/to/Megatron-LM/:/path/to/sglang/python:$PYTHONPATH",
           "CUDA_DEVICE_MAX_CONNECTIONS": "1"
        }
      }' \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 8 \
      --rollout-num-gpus 8 \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${GRPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]} \
      ${RM_ARGS[@]}
fi
