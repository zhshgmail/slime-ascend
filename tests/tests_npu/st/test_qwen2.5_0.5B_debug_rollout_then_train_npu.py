"""
Two-phase debug test:
  Phase 1 – debug_rollout_only: launch sglang, generate rollout data for 2 steps,
            and save them to a temp directory.
  Phase 2 – load_debug_rollout_data (train only): skip sglang entirely, load the
            saved rollout data, and run 2 training steps.

Uses Qwen2.5-0.5B-Instruct (smallest supported model) with 2 GPUs.
"""

import os
import tempfile

import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_ROLLOUT = 4

def _common_args(debug_data_dir: str):
    """Arguments shared by both phases."""

    ckpt_args = (
            "--hf-checkpoint /home/slime-ascend_CI/slime-ascend/weights/Qwen2.5-0.5B-Instruct "
            "--ref-load /home/slime-ascend_CI/slime-ascend/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist "
            "--use-dist-ckpt "
        )
    
    rollout_args = (
        "--prompt-data /home/slime-ascend_CI/slime-ascend/datasets/gsm8k/train.parquet "
        "--input-key question "   # 提取问题列
        "--label-key answer "     # 提取答案列
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = "--advantage-estimator grpo " "--eps-clip 0.2 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--megatron-to-hf-mode bridge "
        "--use-flash-attn "
    )

    return f"{ckpt_args} " f"{rollout_args} " f"{optimizer_args} " f"{grpo_args} " f"{perf_args} " f"{misc_args} "


def execute_rollout_only(debug_data_dir: str):
    """Phase 1: rollout-only, save data."""

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.6 if TIGHT_DEVICE_MEMORY else 0.7} "
        "--sglang-cuda-graph-max-bs 32 "
    )
    sglang_args += "--sglang-device npu "

    phase1_args = (
        f"{_common_args(debug_data_dir)} "
        f"{sglang_args} "
        "--debug-rollout-only "
        f"--save-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.pt "
    )

    print("=" * 60)
    print("Phase 1: debug-rollout-only (generate + save rollout data)")
    print("=" * 60)

    U.execute_train(
        train_args=phase1_args,
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


def execute_train_only(debug_data_dir: str):
    """Phase 2: train-only, load saved rollout data."""

    phase2_args = (
        f"{_common_args(debug_data_dir).replace('--colocate ', '')} "
        f"--load-debug-rollout-data {debug_data_dir}/rollout_{{rollout_id}}.pt "
        "--ci-test "
    )

    print("=" * 60)
    print("Phase 2: load-debug-rollout-data (train only)")
    print("=" * 60)

    U.execute_train(
        train_args=phase2_args,
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


def execute():
# ======================= NPU 环境变量 =======================
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "60000-60050"
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "61000-61050"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # ======================= NPU 环境变量 =======================
    debug_data_dir = tempfile.mkdtemp(prefix="slime_debug_rollout_")
    print(f"Using temp dir for rollout data: {debug_data_dir}")

    execute_rollout_only(debug_data_dir)
    execute_train_only(debug_data_dir)

    print("=" * 60)
    print("Both phases completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
