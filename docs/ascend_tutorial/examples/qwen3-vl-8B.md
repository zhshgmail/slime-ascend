# Qwen3-VL-8B 部署示例
训练脚本位于[run_grpo_npu.sh](../../../examples/geo3k_vlm_multi_turn/run_grpo_npu.sh)
## 模型与数据集下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash
# 下载模型权重 (Qwen/Qwen3-VL-8B-Instruct)
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/Qwen/Qwen3-VL-8B-Instruct

# 下载数据集 (geo3k_imgurl_processed)
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed \
  --local-dir /path/to/geo3k_imgurl_processed
```

## 环境搭建

```sh
conda create -n slime-ascend python=3.11
conda activate slime-ascend

# 进入项目根目录
git clone https://gitcode.com/Ascend/slime-ascend.git
cd slime-ascend && git checkout 42033a2e283f1c9caa9affc0658128587bec28b7
cd ../

# CANN 安装路径，请根据实际情况修改
export CANN_INSTALL_PATH=/usr/local/Ascend
# NPU 设备类型：A3 
export NPU_DEVICE=A3

# 加载 CANN 环境变量
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

# 运行安装脚本,如果遇到任何错误,请参照报错信息及脚本注解进行排查
bash slime-ascend/scripts/ascend_script/quick_install.sh

pip uninstall triton triton-ascend
pip install triton-ascend==3.2.0

pip uninstall transformers
pip install transformers==4.57.1

pip uninstall torch torch_npu
pip install torch==2.8.0 torch_npu==2.8.0

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

## 开始训练
当前环境及数据模型均准备完毕，修改训练脚本中的对应参数,即可开始训练。
在 slime-ascend 文件夹内对以下文件进行修改（主要修改CANN路径、权重路径及训练集路径）：

**GRPO:**

- `examples/geo3k_vlm_multi_turn/run_grpo_npu.sh`
- `examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_grpo_npu.py`

主要修改参数包括：
- `ckpt_args`, `--load`, `--ref-load`: 指定已下载的 Hugging Face 模型权重路径。
- `DATA_ROOT`: 指定数据集的路径。
- `PYTHONPATH`: 指定`Megatron-Bridge`, `Megatron-LM`, `sglang`的路径。

运行训练脚本：
```bash
cd /path/to/slime-ascend
# GRPO
bash examples/geo3k_vlm_multi_turn/run_grpo_npu.sh
```
