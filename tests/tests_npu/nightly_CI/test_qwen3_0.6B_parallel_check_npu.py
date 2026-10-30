
import os

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8


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

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
# ======================= NPU 环境变量 =======================
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60000-60050"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    os.environ["HYDRA_FULL_ERROR"] = "1"
   
    # ======================= NPU 环境变量 =======================
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 24 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
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
    )

    sglang_args = "--rollout-num-gpus 4 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 --sglang-device npu "

    ci_args = "--ci-test --ci-disable-kl-checker "

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
        "--megatron-to-hf-mode bridge "
        "--use-flash-attn "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    for i in range(2):
        U.execute_train(
            train_args=train_args
            + (
                f"--save-debug-rollout-data data-{i}.pt "
                f"--ci-save-grad-norm grad_norms-{i}.pt "
                f"--actor-num-gpus-per-node 4 "
                "--tensor-model-parallel-size 1 "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--use-dynamic-batch-size "
                "--max-tokens-per-gpu 8192 "
            ),
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

        print("🧹 清理外层循环遗留的 SGLang 和 NPU 显存缓存...")
        U.exec_command("pkill -9 sglang || true; sleep 5")

       # 8 GPU CPU 1
        for num_gpus in [4]:
            for tp_size in [1, 2]: 
                for pp_size in [1, 4]:
                    for cp_size in [1, 2,]:
                        
                        total_model_size = tp_size * pp_size * cp_size
                        
                        if total_model_size > num_gpus or num_gpus % total_model_size != 0:
                            continue 
                        num_layers = 28  
                        if num_layers % pp_size != 0:
                            continue
                        
                        # 3. 参数清洗与动态 SP 判断
                        clean_train_args = train_args.replace(sglang_args, "").replace("--no-colocate ", "")
                        sp_arg = "--sequence-parallel " if tp_size > 1 else ""
                        
                        # 4. 组装验证参数
                        args = clean_train_args + (
                            f"--load-debug-rollout-data data-{i}.pt "
                            # f"--ci-load-grad-norm grad_norms-{i}.pt "
                            f"--context-parallel-size {cp_size} "
                            f"--tensor-model-parallel-size {tp_size} "
                            f"--pipeline-model-parallel-size {pp_size} "
                            f"{sp_arg}"
                            f"--actor-num-gpus-per-node {num_gpus} "
                            "--use-dynamic-batch-size "
                            "--max-tokens-per-gpu 8192 "
                        )

                        U.execute_train(
                            train_args=args,
                            num_gpus_per_node=num_gpus,
                            megatron_model_type=MODEL_TYPE,
                            extra_env_vars={
                                "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
                                "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
                                "HCCL_IF_IP": "127.0.0.1",
                                "TP_SOCKET_IFNAME": "lo",
                                "GLOO_SOCKET_IFNAME": "lo",
                            },
                        )
        train_args += "--calculate-per-token-loss "


if __name__ == "__main__":
    # TODO also use typer
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
