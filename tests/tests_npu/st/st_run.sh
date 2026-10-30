#!/bin/bash

# 使用 -print0 处理路径中的空格
#SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DIR=/workspace/slime-ascend/tests/tests_npu/st
test_scripts=$(find "$SCRIPT_DIR" -type f \( -name "test_*.sh" -o -name "*.py" \))
# 如果没找到脚本，提前退出并报错
if [ -z "$test_scripts" ]; then
    echo "❌ 在 $SCRIPT_DIR 下没有找到任何 .sh 或 .py 测试脚本！"
    exit 1
fi

for script in $test_scripts; do
    echo "==================================================="
    echo "🚀 正在执行测试脚本: $script"
    echo "==================================================="

    # 2. 核心智能路由：看后缀
    if [[ "$script" == *.sh ]]; then
        echo "🔍 检测到 Bash 脚本，使用 bash 执行..."
        COMMAND="bash $script"
    elif [[ "$script" == *.py ]]; then
        echo "🔍 检测到 Python 脚本，使用 python3 执行..."
        COMMAND="python3 $script"
    else
        echo "⚠️ 未知的文件类型，跳过: $script"
        continue
    fi

    # 3. 执行对应的命令，并捕获错误
    if ! $COMMAND; then
        echo "❌ 脚本执行失败，CI 流水线中断: $script"
        exit 1
    fi

    echo "✅ 任务执行完成: $script"
    ray stop --force
    ps -ef | grep torchrun | grep -v grep | awk '{print $2}' | xargs -r kill -9
    echo "计算资源清理完成"

done