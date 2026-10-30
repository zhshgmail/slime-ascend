## PD 分离（Prefill-Decode Disaggregation）

PD 分离将 rollout 的 prefill 阶段与 decode 阶段拆分到不同的 worker 上独立运行，可以针对两类计算特性分别配置资源与参数，提升吞吐量并降低延迟。

### 配置方式

slime 支持两种 PD 配置方式。

**1. 简单路径：`--prefill-num-servers`**

如果只有单个 actor model，且只需要简单的 PD layout，可以在 `SGLANG_ARGS` 中设置：

```bash
SGLANG_ARGS=(
   --prefill-num-servers 1
   ...
)
```

适合只想拆开 prefill/decode、但不需要分别调整每个 group 的轻量场景。

**2. 高级路径：`--sglang-config`**

生产级 rollout topology 推荐使用 SGLang Config，并在 yaml 文件中独立配置 prefill 和 decode group，更多详细说明参考 [sglang-config.md](../../zh/advanced/sglang-config.md)。

示例配置文件 `sglang_pd.yaml`：

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
        overrides:
          disaggregation-mode: prefill
          disaggregation-bootstrap-port: 8996
          # Prefill 节点独立配置（可选）
          chunked_prefill_size: 8192
      - worker_type: decode
        num_gpus: 12
        num_gpus_per_engine: 4
        overrides:
          disaggregation-mode: decode
          # Decode 节点独立配置（可选）
          mem_fraction_static: 0.88
```

启动命令：

```bash
python train.py \
  --sglang-config sglang_pd.yaml \
  --rollout-num-gpus 16 \
  ...
```

### NPU 额外配置

在 NPU 上启用 PD 分离，除上述通用配置外，还需要完成以下步骤。

**安装依赖**

```bash
pip install memfabric_hybrid
pip install mf_adapter
```

**配置 MemFabric Bootstrap 地址**

SGLang 的 PD 分离在 NPU 上通过 MemFabric 进行 P 节点与 D 节点之间的通信。需要在 P 节点和 D 节点上统一设置相同的 bootstrap 地址（使用 P 节点的 IP，端口可自由指定）：

```bash
# bootstrap通信地址，设置为 P 节点地址
export ASCEND_MF_STORE_URL="tcp://<P节点IP>:<端口>"

# 示例
export ASCEND_MF_STORE_URL="tcp://7.150.8.255:8000"
```

**配置传输后端**

在 `SGLANG_ARGS` 中添加 `--sglang-disaggregation-transfer-backend` 参数：

```bash
SGLANG_ARGS=(
   --sglang-disaggregation-transfer-backend ascend
   ...
)
```

### 完整示例

```bash
export ASCEND_MF_STORE_URL="tcp://7.150.8.255:8000"

SGLANG_ARGS=(
   --sglang-disaggregation-transfer-backend ascend
   ...
)

ray job submit ... \
   -- python3 train.py \
   --sglang-config sglang_pd.yaml \
   --rollout-num-gpus 16 \
   ${SGLANG_ARGS[@]}
```

> 💡 **提示**：`ASCEND_MF_STORE_URL` 在 P 节点和 D 节点上设置为相同的值，地址指向 P 节点的 IP。端口号可任意选择，只需确保未被占用且网络可达。

---
