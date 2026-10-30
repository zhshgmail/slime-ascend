# GLM-4.7-30B-A3B 部署示例
训练脚本位于[run-glm4.7-30B-A3B-npu.sh](../../../scripts/ascend_script/run-glm4.7-30B-A3B-npu.sh)
## 模型与数据集下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash
# 下载模型权重 (GLM-4.7-Flash)
hf download zai-org/GLM-4.7-Flash --local-dir /path/to/GLM-4.7-Flash

# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /path/to/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /path/to/aime-2024
```

## 模型权重转换

### Hugging Face 格式 转换为 Megatron 格式

当使用 Megatron 作为训练后端时，需要先将 Hugging Face 格式的模型权重转换为 Megatron `torch_dist` 格式。

首先，加载目标模型的配置文件。`slime/scripts/models` 目录下包含了支持模型的配置文件。需要 `source` 对应模型的脚本，将配置参数加载到当前环境中。此处我们以 GLM4.7-30B-A3B 模型为例子。

```bash
cd /path/to/slime-ascend
source scripts/models/glm4.7-30B-A3B.sh
```

接下来，运行转换脚本。请注意以下参数：
- `--hf-checkpoint`: 指定已下载的 Hugging Face 模型权重路径。
- `--save`: 指定转换后 `torch_dist` 格式权重的保存路径。

```bash
PYTHONPATH=/path/to/Megatron-LM:/path/to/Megatron-Bridge/src:$PYTHONPATH torchrun --nproc-per-node 8  tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/GLM-4.7-Flash \
    --save /path/to/GLM-4.7-Flash_torch_dist \
    --moe-token-dispatcher-type alltoall
```
## 开始训练
当前环境及数据模型均准备完毕，修改训练脚本中的对应参数,即可开始训练。
主要修改参数包括：
- `--hf-checkpoint`: 指定已下载的 Hugging Face 模型权重路径。
- `--ref-load`: 指定转换后的 `torch_dist` 格式权重路径。
- `--prompt-data`: 指定训练数据集的路径。
- `--eval-prompt-data`: 指定评估数据集的路径。
- `RUNTIME_ENV_JSON`: 在训练脚本中指定运行时环境变量 JSON 字符串，包含 ASCEND 相关环境变量。

运行训练脚本：
```bash
cd /path/to/slime-ascend
bash scripts/ascend_script/run-glm4.7-30B-A3B-npu.sh
```
