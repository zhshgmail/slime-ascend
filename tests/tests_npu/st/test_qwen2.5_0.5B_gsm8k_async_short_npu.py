import os
import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8

def execute():
# ======================= NPU 环境变量 =======================
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60000-60050"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # ======================= NPU 环境变量 =======================
    ckpt_args = (
            "--hf-checkpoint /home/slime-ascend_CI/slime-ascend/weights/Qwen2.5-0.5B-Instruct "
            "--ref-load /home/slime-ascend_CI/slime-ascend/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist "
            "--use-dist-ckpt "
        )
    
    rollout_args = (
        "--prompt-data /home/slime-ascend_CI/slime-ascend/datasets/gsm8k/train.parquet "
        "--input-key question "
        "--label-key answer "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 8 "
        "--eval-prompt-data gsm8k /home/slime-ascend_CI/slime-ascend/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        # "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.55 if TIGHT_DEVICE_MEMORY else 0.65} "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-enable-metrics "
    )
    sglang_args += "--sglang-device npu "

    ci_args = "--ci-test "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--megatron-to-hf-mode bridge "
        "--use-flash-attn "
    )

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
        f"{fault_tolerance_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={
            "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
            "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
            "HCCL_IF_IP": "127.0.0.1",
            "TP_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
        },
    )


if __name__ == "__main__":
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    execute()
