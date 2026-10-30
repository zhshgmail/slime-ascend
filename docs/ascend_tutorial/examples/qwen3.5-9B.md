# Qwen3.5-9B 部署示例

训练脚本位于[npu\_run\_geo3k\_vlm\_multi\_turn\_qwen35\_9b.sh](../../../examples/geo3k_vlm_multi_turn/npu_run_geo3k_vlm_multi_turn_qwen35_9b.sh)

## 版本配套

| 组件                    | 版本                                                                                                    |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| CANN                  | [8.5.1 商发](https://www.hiascend.com/developer/download/community/result?module=cann\&cann=8.5.1) |
| Megatron-Bridge-slime | [qwen35 分支](https://github.com/coding-famer/Megatron-Bridge-slime.git)                                |
| sgl-kernel-npu        | 2026.04.15.rc3                                                                                               |
| triton\_ascend        | 3.2.0                                                                                                 |
| transformers          | 5.3.0                                                                                                 |
| PTA                   | beta 版本未出包，先用内部版本                                                                                     |

> **可选升级：** CANN 9.0.0 b160 配 triton-ascend 3.2.1：
>
> - [TA 3.2.1](https://gitcode.com/Ascend/triton-ascend/releases)
> - [CANN 9.0.0 B160 商发](https://www.hiascend.com/developer/download/community/result?module=cann\&cann=9.0.0)
> - sgl-kernel-npu 2026.05.01.post1

## 模型与数据集下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash
# 下载模型权重 (Qwen/Qwen3.5-9B)
hf download Qwen/Qwen3.5-9B --local-dir /path/to/Qwen/Qwen3.5-9B

# 下载数据集 (geo3k_imgurl_processed)
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed \
  --local-dir /path/to/geo3k_imgurl_processed
```

## 环境搭建


```sh
conda create -n slime-ascend python=3.11
conda activate slime-ascend

# CANN 安装路径，请根据实际情况修改
export CANN_INSTALL_PATH=/usr/local/Ascend

# 加载 CANN 环境变量
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh
```

### 1. 安装 SGLang

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout v0.5.10
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e "python[all_npu]" && cd ..
```

### 2. 安装 sgl-kernel-npu

```bash
git clone https://github.com/sgl-project/sgl-kernel-npu.git && cd sgl-kernel-npu
# source cann
git checkout 2026.04.15.rc3
bash build.sh -a kernels
bash build.sh -a memory-saver
pip install output/sgl_kernel_npu*.whl
pip install output/torch*.whl
cd ..
```

### 3. 安装 mbridge 与 Megatron-Bridge-slime

```bash
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

git clone https://github.com/coding-famer/Megatron-Bridge-slime.git && cd Megatron-Bridge-slime
git checkout qwen35 && cd ..

pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
```

### 4. 安装 Megatron-LM 与 MindSpeed

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
git checkout 3714d81d
pip install -e . && cd ..

git clone https://gitcode.com/Ascend/MindSpeed.git && cd MindSpeed
# 先把 master 分支的 mindspeed/ops/triton 拷贝一份到外面
git checkout fc63de5c
pip install -e . && cd ..
# 把 triton 算子拷贝回来
```

### 5. 安装 slime-ascend

```bash
git clone https://gitcode.com/Ascend/slime-ascend.git && cd slime-ascend
pip install -e . && cd ..
```

### 6. 应用 Patch

```bash
cd sglang
git am ../slime-ascend/docker/npu_patch/v0.2.4/sglang/*
cd ../Megatron-LM/
git am ../slime-ascend/docker/npu_patch/v0.2.4/megatron/*
cd ../Megatron-Bridge-slime/
git am ../slime-ascend/docker/npu_patch/v0.2.4/megatron-bridge-slime/*
cd ../MindSpeed/
git am ../slime-ascend/docker/npu_patch/v0.2.4/mindspeed/*
```

### 7. GDN 算子 Run 包编译与安装


```bash
# source cann
git clone https://github.com/flashserve/flash-linear-attention-npu.git
cd flash-linear-attention-npu && git checkout v26.1.0
bash install_deps.sh

# 通过 --soc 参数精准指定当前设备芯片型号，示例配置 --soc=ascend910_93，杜绝芯片型号不匹配导致的编译失败
bash build.sh --soc=ascend910_93 --pkg --ops=chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd,causal_conv1d

# 安装 run 包
./build_out/cann-ops-transformer-custom_linux-aarch64.run
```

> **注意：** 若 Run 包安装卡顿、执行无响应，原因是 CANN 已有 vendors 目录冲突，需手动清理路径 `cann-8.5.0/opp/vendors/` 下原有文件后，重新执行安装脚本：
>
> ```bash
> rm -rf /usr/local/Ascend/cann-8.5.0/opp/vendors/
> # 再重新运行
> ./build_out/cann-ops-transformer-custom_linux-aarch64.run
> ```

### 8. 适配 WHL 包编译与环境验证

```bash
# torch_custom 是 flash-linear-attention-npu 的子目录
cd torch_custom/fla_npu

bash build.sh
```

### 9. 其他依赖版本

```bash
pip install torchvision==0.23.0
pip uninstall triton triton_ascend -y
pip install triton_ascend==3.2.0
pip install opencv-python-headless
pip install numpy==1.26.0
```

## 开始训练

当前环境及数据模型均准备完毕，修改训练脚本中的对应参数，即可开始训练。
在 slime-ascend 文件夹内对以下文件进行修改（主要修改 CANN 路径、权重路径及训练集路径）：

- `examples/geo3k_vlm_multi_turn/npu_run_geo3k_vlm_multi_turn_qwen35_9b.sh`
- `examples/geo3k_vlm_multi_turn/npu_run_geo3k_vlm_multi_turn_qwen35_9b.py`

主要修改参数包括：

- `PYTHONPATH`: 指定 `Megatron-Bridge-slime`、`Megatron-LM`、`sglang` 的路径。
- `ckpt_args`, `--load`, `--ref-load`: 指定已下载的 Hugging Face 模型权重路径。
- `DATA_ROOT`: 指定数据集的路径。

运行训练脚本：

```bash
cd /path/to/slime-ascend
bash examples/geo3k_vlm_multi_turn/npu_run_geo3k_vlm_multi_turn_qwen35_9b.sh
```

