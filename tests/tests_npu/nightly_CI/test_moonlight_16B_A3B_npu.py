import os
import slime.utils.external_utils.command_utils as U

ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))

MODEL_NAME = "Moonlight-16B-A3B-Instruct"
MODEL_TYPE = "moonlight"
NUM_GPUS = 16


def prepare():
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"/workspace/Megatron-LM:/workspace/Megatron-Bridge/src:/workspace/MindSpeed:/workspace/mbridge:{current_pythonpath}"
    U.exec_command("mkdir -p /root/models /root/datasets")

    local_weight_path = f"/home/slime-ascend_CI/slime-ascend/weights/{MODEL_NAME}"
    target_weight_path = f"/root/models/{MODEL_NAME}"
    U.exec_command(f"ln -snf {local_weight_path} {target_weight_path}")

    for dataset_name in ("dapo-math-17k", "aime-2024"):
        local_dataset_path = f"/home/slime-ascend_CI/slime-ascend/datasets/{dataset_name}"
        target_dataset_path = f"/root/datasets/{dataset_name}"
        U.exec_command(f"ln -snf {local_dataset_path} {target_dataset_path}")


def execute():
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60000-60050"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    ckpt_args = (
        f"--hf-checkpoint /home/slime-ascend_CI/slime-ascend/weights/{MODEL_NAME} "
        f"--ref-load /home/slime-ascend_CI/slime-ascend/weights/{MODEL_NAME}_torch_dist "
        "--use-dist-ckpt "
    )

    rollout_args = (
        "--prompt-data /home/slime-ascend_CI/slime-ascend/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data aime /home/slime-ascend_CI/slime-ascend/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 2 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity selective "
        "--micro-batch-size 1 "
        "--use-distributed-optimizer "
        f"--max-tokens-per-gpu 1024 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if TIGHT_HOST_MEMORY else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--rollout-num-gpus 8 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-cuda-graph-max-bs 32 "
        "--sglang-max-running-requests 512 "
    )
    sglang_args += "--sglang-device npu "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--use-flash-attn "
        "--no-scatter-gather-tensors-in-pipeline "
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
        f"{misc_args} "
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
            "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
            "HCCL_IF_IP": "127.0.0.1",
            "TP_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
