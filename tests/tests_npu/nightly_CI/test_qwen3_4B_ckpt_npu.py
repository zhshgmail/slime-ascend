import os
from argparse import ArgumentParser

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))

MODEL_NAME = "qwen3-4b-sft"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    # 1. 软链接本地模型权重
    local_weight_path = f"/home/slime-ascend_CI/slime-ascend/weights/{MODEL_NAME}"
    target_path = f"/root/models/{MODEL_NAME}"
    U.exec_command(f"ln -snf {local_weight_path} {target_path}")
    
    # 2. 软链接本地数据集 (替换掉之前的 U.hf_download_dataset)
    local_dataset_path = "/home/slime-ascend_CI/slime-ascend/datasets/dapo-math-17k"
    target_dataset_path = "/root/datasets/dapo-math-17k"
    U.exec_command(f"ln -snf {local_dataset_path} {target_dataset_path}")

def execute(mode: str = ""):
# ======================= NPU 环境变量 =======================
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60000-60050"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # ======================= NPU 环境变量 =======================
    ckpt_args = f"--hf-checkpoint /home/slime-ascend_CI/slime-ascend/weights/qwen3-4b-sft " f"--ref-load /home/slime-ascend_CI/slime-ascend/weights/qwen3-4b-sft_torch_dist "
    if mode == "save":
        ckpt_args += f"--save /root/models/{MODEL_NAME}_slime "
        ckpt_args += "--save-interval 2 "
    elif mode == "async_save":
        ckpt_args += f"--save /root/models/{MODEL_NAME}_slime "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load /root/models/{MODEL_NAME}_slime "
        ckpt_args += "--ckpt-step 1 "

    rollout_args = (
        "--prompt-data /home/slime-ascend_CI/slime-ascend/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if TIGHT_HOST_MEMORY else 16384} "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        " --use-tis "
        " --grpo-clip 0.2 "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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

    sglang_args = "--rollout-num-gpus-per-engine 4 --rollout-num-gpus 4 --sglang-mem-fraction-static 0.6 --sglang-cuda-graph-max-bs 32 --sglang-device npu "

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
        "--actor-num-gpus-per-node 4 "
        "--megatron-to-hf-mode bridge "
        "--use-flash-attn "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
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
            "GLOO_SOCKET_IFNAME": "lo"
        },
    )


if __name__ == "__main__":
    args = parser.parse_args()
    # TODO also use typer
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute("save" if not args.async_save else "async_save")
    print("🧹 Cleaning up Ray processes before starting load test...")
    U.exec_command("ray stop -f && pkill -9 ray && pkill -9 sglang; sleep 5")
    execute("load")
