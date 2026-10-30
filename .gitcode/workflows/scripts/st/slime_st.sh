#!/bin/bash
set -e
# ==========================================
# CI 基础设施配置 (支持环境变量覆盖)
# ==========================================
WORKSPACE=$1
pr_id=$2
branch=$3
REPO_DIR="${WORKSPACE}"
TEST_REPO_DIR="/workspace/slime-ascend"

# 允许外部 CI 环境变量覆盖默认路径
WEIGHTS_DIR=${SLIME_WEIGHTS_DIR:-"/home/slime-ascend_CI/slime-ascend/weights"}
DATASETS_DIR=${SLIME_DATASETS_DIR:-"/home/slime-ascend_CI/slime-ascend/datasets"}

# ==========================================
# 参数预检
# ==========================================
if [ -z "${WORKSPACE}" ] || [ -z "${branch}" ]; then
    echo "❌ 用法: bash launch_ci.sh <workspace> <pr_id> <branch>"
    exit 1
fi

if [ ! -d "${REPO_DIR}" ]; then
    echo "❌ 错误: 仓库目录 [ ${REPO_DIR} ] 不存在"
    exit 1
fi

if [ -n "${pr_id}" ] && ! echo "${pr_id}" | grep -qE '^[0-9a-zA-Z_-]+$'; then
    echo "❌ 错误: 非法 pr_id [ ${pr_id} ]"
    exit 1
fi

if ! echo "${branch}" | grep -qE '^[0-9a-zA-Z._-]+$'; then
    echo "❌ 错误: 非法 branch [ ${branch} ]"
    exit 1
fi

echo "------------------------------------------------"
echo "🔍 预检：分支 [ ${branch} ]，工作区 [ ${WORKSPACE} ]，PR [ ${pr_id:-N/A} ]"
echo "📁 PR 代码目录: ${REPO_DIR}"
echo "📁 测试目录: ${TEST_REPO_DIR}"

# 测试脚本和镜像中的可编辑安装均使用此路径。先准备 PR 代码，
# 再替换镜像预置目录，避免已删除的文件继续参与测试。
REPO_DIR=$(realpath "${REPO_DIR}")
if [ "${REPO_DIR}" = "${TEST_REPO_DIR}" ] || [[ "${REPO_DIR}" == "${TEST_REPO_DIR}/"* ]]; then
    echo "❌ 错误: PR 代码目录不能位于测试目录内"
    exit 1
fi
if [ -L "${TEST_REPO_DIR}" ] || { [ -e "${TEST_REPO_DIR}" ] && [ ! -d "${TEST_REPO_DIR}" ]; }; then
    echo "❌ 错误: 测试目录不是普通目录"
    exit 1
fi
if [ -d "${TEST_REPO_DIR}" ] && mountpoint -q "${TEST_REPO_DIR}"; then
    echo "❌ 错误: 测试目录是挂载点，不能替换"
    exit 1
fi
STAGING_DIR=$(mktemp -d /workspace/.slime-ascend-pr.XXXXXX)
trap 'rm -rf -- "${STAGING_DIR}"' EXIT
cp -a "${REPO_DIR}/." "${STAGING_DIR}/"
if [ ! -f "${STAGING_DIR}/tests/tests_npu/slime-ascend_ut.sh" ]; then
    echo '❌ 错误: PR 代码中找不到测试入口点 [ tests/tests_npu/slime-ascend_ut.sh ]'
    exit 1
fi
rm -rf -- "${TEST_REPO_DIR}"
mv -- "${STAGING_DIR}" "${TEST_REPO_DIR}"
trap - EXIT

cd "${TEST_REPO_DIR}"
if WORKSPACE="${TEST_REPO_DIR}" bash tests/tests_npu/slime-ascend_ut.sh "${TEST_REPO_DIR}" "${pr_id}" "${branch}"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi
chmod -R 777 "${TEST_REPO_DIR}" || true
exit "${EXIT_CODE}"
