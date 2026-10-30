#!/bin/bash
#  门禁UT 脚本

set -x
WORKSPACE=/workspace/slime-ascend
pr_id=$2
branch=${3:-"v0.2.4"}
apply_patch_smart() {
    local repo_path=$1
    local patch_dir=$2  
    echo "------------------------------------------------"
    echo "🔍 检查仓库: $repo_path"

    # 1. 基础路径检查
    if [ ! -d "$repo_path" ]; then
        echo "❌ 错误: 找不到目录 $repo_path"
        exit 1
    fi
    if [ ! -d "$patch_dir" ] || [ -z "$(ls -A "$patch_dir" 2>/dev/null)" ]; then
        echo "⚠️ 提醒: $patch_dir 下无补丁文件，跳过。"
        return 0
    fi

    cd "$repo_path"  
    # 清理任何可能残留的失败状态
    git am --abort 2>/dev/null || true

    # 2. 【核心修复】查户口预检！检查最近 10 次提交里，有没有 temp 的记录
    if git log -10 --format='%ce' | grep -q "temp@example.com"; then
        echo "⚠️ 预检：代码库已包含 temp 打过的补丁，无需重复应用，安全跳过。"
        return 0
    fi

    # 3. 执行应用
    echo "🚀 状态纯净！正在注入 NPU 补丁..."
    if git am --whitespace=fix "$patch_dir"/*; then
        echo "✅ 补丁应用成功！"
    else
        echo "❌ 严重错误：补丁应用失败，请检查文件系统或基线代码！"
        git am --abort
        exit 1
    fi
}
# --- 执行任务 ---
case "$branch" in
    "v0.2.2")
        PATCH_VERSION="v0.2.2"
        ;;
    "v0.2.4")
        PATCH_VERSION="v0.2.4"
        ;;
    "main" | "master" | "v0.3.0")  # 把 main 和 0.3.0 统一定向到 v0.3.0 补丁
        PATCH_VERSION="v0.3.0"
        ;;
    *)
        echo "⚠️ 警告: 未知分支 [ ${branch} ]，默认回退使用 v0.3.0 补丁..."
        PATCH_VERSION="v0.2.4"
        ;;
esac

# 组合出最终的补丁根目录
PATCH_BASE="${WORKSPACE}/docker/npu_patch/${PATCH_VERSION}"
echo "📦 准备注入补丁，当前选择的补丁目录: ${PATCH_BASE}"

# --- 2. 执行任务 ---
case "$branch" in
    "v0.2.2"|"v0.2.4")
        apply_patch_smart "/workspace/sglang" "${PATCH_BASE}/sglang"
        apply_patch_smart "/workspace/Megatron-LM" "${PATCH_BASE}/megatron"
        apply_patch_smart "/workspace/Megatron-Bridge" "${PATCH_BASE}/megatron-bridge"
        apply_patch_smart "/workspace/MindSpeed" "${PATCH_BASE}/mindspeed"
        apply_patch_smart "/workspace/mbridge" "${PATCH_BASE}/mbridge"
        ;;
    *)
        apply_patch_smart "/workspace/sglang" "${PATCH_BASE}/sglang"
        apply_patch_smart "/workspace/Megatron-LM" "${PATCH_BASE}/megatron"
        apply_patch_smart "/workspace/Megatron-Bridge" "${PATCH_BASE}/megatron-bridge"
        apply_patch_smart "/workspace/mbridge" "${PATCH_BASE}/mbridge"
        apply_patch_smart "/workspace/MegatronAdaptor" "${PATCH_BASE}/megatron_adaptor"
        apply_patch_smart "/workspace/TransformerEngineNPU" "${PATCH_BASE}/transformer_engine_npu"
        ;;
esac


cd ${WORKSPACE}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="/workspace/mbridge:/workspace/Megatron-LM:/workspace/sglang/python:/workspace/MindSpeed:/workspace/Megatron-Bridge/src:$PYTHONPATH"
echo $PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
# 启用共享内存通信
export HCCL_ENABLE_SL=1          # 启用 SL（Shared Memory Link）
export HCCL_SHM_SIZE=2147483648  # 共享内存大小 2GB
# 启用 HCCL 快速初始化
# export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=600

# 启用轮询模式（降低延迟）
export HCCL_POLLING_MODE=1

# 设置缓冲区大小（建议 4MB~8MB）
# export HCCL_BUFFSIZE=8388608  # 8MB

# 启用通信组合优化
export HCCL_COMBINE_MODE=1

# （可选）指定通信算法：0=自动，1=Ring，2=Tree
export HCCL_ALGO_TYPE=0
export HCCL_IF_IP=127.0.0.1

export GLOO_SOCKET_IFNAME=enp23s0f3
#ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9

python /workspace/slime-ascend/tests/tests_npu/ci_check.py  
