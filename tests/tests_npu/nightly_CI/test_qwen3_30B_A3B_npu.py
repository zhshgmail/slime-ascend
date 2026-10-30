import os

import slime.utils.external_utils.command_utils as U
HF_WEIGHTS_ROOT = "/home/slime-ascend_CI/slime-ascend/weights"
DATASETS_ROOT = "/home/slime-ascend_CI/slime-ascend/datasets"
MEGATRON_WEIGHTS_ROOT = "/home/slime-ascend_CI/slime-ascend/weights-MA"

ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
USE_DEEPEP = bool(int(os.environ.get("SLIME_TEST_USE_DEEPEP", "1")))
USE_FP8_ROLLOUT = bool(int(os.environ.get("SLIME_TEST_USE_FP8_ROLLOUT", "1")))

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
# Eight cards for Megatron training and eight separate cards for rollout.
NUM_GPUS = 16


def execute():
    if USE_FP8_ROLLOUT:
        ckpt_args = (
            f"--hf-checkpoint {HF_WEIGHTS_ROOT}/{MODEL_NAME}-FP8 "
            f"--ref-load {MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist "
        )
    else:
        ckpt_args = (
            f"--hf-checkpoint {HF_WEIGHTS_ROOT}/{MODEL_NAME} "
            f"--ref-load {MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist "
        )

    rollout_args = (
        f"--prompt-data {DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 4 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        f"--eval-prompt-data aime24 {DATASETS_ROOT}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        # Preserve the upstream 8-card model sharding; reducing CP/EP here
        # creates a 54GiB Megatron gradient buffer on each training card.
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 1024 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        "--use-routing-replay "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus 8 "
        "--rollout-num-gpus-per-engine 2 "
        "--sglang-mem-fraction-static 0.50 "
        "--sglang-cuda-graph-max-bs 4 "
        "--sglang-max-running-requests 16 "
        "--sglang-enable-metrics "
    )
    sglang_args += "--sglang-device npu "

    if USE_DEEPEP:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
    )

    if USE_DEEPEP:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        # Short rollouts can still dispatch 510 MoE tokens (280MB minimum).
        extra_env_vars={"HCCL_BUFFSIZE": "400"},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
