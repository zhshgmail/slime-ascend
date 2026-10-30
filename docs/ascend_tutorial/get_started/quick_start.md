# 昇腾（NPU）快速使用

本文档从搭建环境开始，在一小时内带您快速上手 slime-ascend，涵盖 NPU 环境配置、数据准备、训练启动和关键代码解析。

## 硬件支持说明

**slime-ascend** 支持华为昇腾 NPU 硬件平台：

- **Atlas 800T A3**
- **Atlas 800T A2**
- >**备注：A5支持度暂未验证，待Q3正式更新**


**重要说明**：
- 请确保已正确安装 CANN 套件（推荐 CANN 9.0.0 或更高版本）
- 推荐使用 Linux arm64 环境进行部署
- 本版本当前已使用MegatronAdaptor和TransformerEngineNPU代替MindSpeed进行NPU上Megatron-LM的适配支持


## 组件版本

### Python 3.11

| 组件 | A2 版本 | A3 版本 |
| ---- | ------- | ------- |
| 基础镜像 | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |
| CANN | 9.0.0 | 9.0.0 |
| torch | 2.10.0 | 2.10.0 |
| torch_npu | 2.10.0 | 2.10.0 |
| torchvision | 0.25.0 | 0.25.0 |
| Megatron-LM | 1dcf0dafa | 1dcf0dafa |
| Megatron-Bridge | dev_rl | dev_rl |
| triton-ascend | 3.2.1 | 3.2.1 |
| mbridge | 89eb1088 | 89eb1088 |
| SGLang | v0.5.13 | v0.5.13 |
| sgl-kernel-npu | 2026.6.1 | 2026.6.1 |
| nvidia-modelopt | >=0.37.0 | >=0.37.0 |
| Slime | v0.3.0 | v0.3.0 |

### Python 3.12

| 组件 | A2 版本 | A3 版本 |
| ---- | ------- | ------- |
| 基础镜像 | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.12 | 3.12 |
| CANN | 9.1.0 | 9.1.0 |
| torch | 2.10.0 | 2.10.0 |
| torch_npu | 2.10.0 | 2.10.0 |
| torchvision | 0.25.0 | 0.25.0 |
| Megatron-LM | 1dcf0dafa | 1dcf0dafa |
| Megatron-Bridge | dev_rl | dev_rl |
| triton-ascend | 3.2.1 | 3.2.1 |
| mbridge | 89eb1088 | 89eb1088 |
| SGLang | v0.5.13 | v0.5.13 |
| sgl-kernel-npu | 2026.8.17 | 2026.8.17 |
| nvidia-modelopt | >=0.37.0 | >=0.37.0 |
| Slime | v0.3.0 | v0.3.0 |

## 使用指南

| 训练算法 | 支持模型 | 发布状态 |
| -------- | ------- | -------- |
| GRPO | [GLM-4.7-Flash](../examples/glm4.7-30B-A3B.md) | Preview |
| GRPO | [Qwen3-8B](../examples/qwen3-8B.md) <br> [Qwen3-VL-8B](../examples/qwen3-vl-8B.md) <br> [Qwen3.5-9B](../examples/qwen3.5-9B.md)| Preview |

注意："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本。

## 环境配置

### 方式一：Docker（推荐）

我们提供了针对 A3 硬件的 [Dockerfile](../../../docker/npu_docker/v0.3.0/dockerfile_build_guidance.md)，可快速构建完整训练环境。

#### 镜像构建

```bash
# 进入项目根目录
cd {slime-ascend-root-path}

# 构建 A3 镜像
docker build -f Dockerfile.a3.ubuntu22.04.cann90.latest \
  -t slime-ascend:9.0.0-a3-ubuntu22.04-py3.11-latest .

```

#### 启动容器

```bash
# 启动容器（以 A3 为例）
docker run -it --rm \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v $(pwd):/workspace \
  -w /workspace \
  slime-ascend:9.0.0-a3-ubuntu22.04-py3.11-latest
```

> 声明：slime-ascend 中提供的 Ascend 相关 Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通。

### 方式二：本地安装脚本

我们提供了[一键安装脚本](../../../scripts/ascend_script/quick_install.sh)，可快速搭建 NPU 环境：

```bash
conda create -n slime-ascend python=3.11
conda activate slime-ascend
# 进入项目根目录
git clone https://gitcode.com/Ascend/slime-ascend.git
# CANN 安装路径，请根据实际情况修改
export CANN_INSTALL_PATH=/usr/local/Ascend
# NPU 设备类型：A3
export NPU_DEVICE=A3
# 加载 CANN 环境变量
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh
# 运行安装脚本
bash slime-ascend/scripts/ascend_script/quick_install.sh
```

**安装脚本会自动完成以下工作**：

1. 从源码安装 SGLang
2. 安装 torch、torch_npu、triton_ascend 等基础依赖包
3. 安装 sgl-kernel-npu（支持 A2 和 A3 两种硬件）
4. 安装 mbridge 和 Megatron-Bridge
5. 安装 Megatron-LM
6. 安装 MegatronAdaptor和TransformerEngineNPU
7. 安装 slime-ascend
8. 应用所有 NPU 相关补丁

> 注意：请确保使用提供的 quick_install.sh 或从指定的分支源码安装。

## 训练脚本与参数概览

完成上述准备工作后，即可运行 NPU 训练脚本。我们提供了 GLM-4.7 在 NPU 上的训练脚本示例，参照 [glm4.7-30B-A3B.md](../examples/glm4.7-30B-A3B.md)。

## 常见问题排查

### NPU 相关问题

1. **CANN 环境未正确加载**
   - 确保已执行 `source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`
   - 检查 `ASCEND_TOOLKIT_HOME` 等环境变量是否正确设置

2. **NPU 设备不可见**
   - 检查 `ASCEND_RT_VISIBLE_DEVICES` 环境变量
   - 使用 `npu-smi info` 命令验证 NPU 设备状态

3. **显存不足**
   - 调整 `--sglang-mem-fraction-static` 参数（建议 0.6-0.8）
   - 减少 batch size 或并行度

4. **HCCL 通信错误**
   - 检查 `HCCL_HOST_SOCKET_PORT_RANGE` 和 `HCCL_NPU_SOCKET_PORT_RANGE` 配置
   - 确保端口范围内的端口未被占用

更多常见问题请参考 [Q&A](../../zh/get_started/qa.md)
