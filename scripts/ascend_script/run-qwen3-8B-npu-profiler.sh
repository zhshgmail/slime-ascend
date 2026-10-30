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
pkill -9 redis

set -ex
ulimit -n 65535

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export PYTHONPATH="/path/to/Megatron-LM/:/path/to/sglang/python:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# make sure "logs" exits
export LOG_FILE=./logs/train_log_$(date +%Y-%m-%d_%H-%M-%S)_qwen3_8b.log

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen3-8B
   --ref-load /path/to/Qwen3-8B.sh_torch_dist
   --load /path/to/Qwen3-8B_slime/
   --save /path/to/Qwen3-8B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /path/to/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --no-gradient-accumulation-fusion
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
   # --wandb-group qwen3-8B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 4 8 16 32 64 128
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 32768
   --sglang-max-prefill-tokens 4000
   --sglang-max-total-tokens 327680
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
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
   --use-flash-attn
)

PROFILER_ARGS=(
   # ---- NPU profiler (torch_npu.profiler) verification ----
   # actor (training) side: collect global rollout step 0 only
   --npu-profile-actor
   --npu-profile-actor-global-step-start 0
   --npu-profile-actor-global-step-end 1
   --npu-profile-actor-level level1
   --npu-profile-actor-export-type db
   --npu-profile-actor-contents npu cpu
   --npu-profile-actor-save-path /data/slime/actor_profile
   # rollout side (sglang server processes), controlled independently
   --npu-profile-rollout
   --npu-profile-rollout-global-step-start 0
   --npu-profile-rollout-global-step-end 1
   --npu-profile-rollout-save-path /data/slime/rollout_profiler
   --npu-profile-rollout-token-start 10
   --npu-profile-rollout-token-end 110
   --npu-profile-rollout-contents npu
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_DEBUG=legacy \
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/path/to/Megatron-LM/:/path/to/Megatron-Bridge/src:$PYTHONPATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"ASCEND_TOOLKIT_HOME\": \"/path/to/ascend-toolkit/latest/\",
    \"ASCEND_OPP_PATH\": \"/path/to/ascend-toolkit/latest/opp/\",
    \"ASCEND_AICPU_PATH\": \"/path/to/ascend-toolkit/latest/\",
    \"ASCEND_HOME_PATH\": \"/path/to/ascend-toolkit/latest/\",
    \"set_env_path\": \"/path/to/nnal/atb/set_env.sh\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"RAY_DEBUG_POST_MORTEM\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
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
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PROFILER_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
