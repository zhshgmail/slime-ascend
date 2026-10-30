# 多教师在策略蒸馏（MOPD）

Multi-Teacher On-Policy Distillation（MOPD，多教师在策略蒸馏）让学生模型使用自己生成的 rollout，向多个领域教师学习。每个样本可以按领域路由到一个或多个教师；训练侧通过 importance sampling 修正 rollout 策略与当前策略之间的偏差，并支持 token-level、全词表和 top-k 三种蒸馏方式。

MOPD 适合以下场景：

- 不同领域已有专门教师，例如数学教师、代码教师和视觉几何教师；
- 教师与学生架构不同，需要通过独立 SGLang 服务提供教师概率；
- 全词表教师 logits 的显存、主机内存或网络开销过高，希望用 top-k 近似。

## 与 OPD 的关系

MOPD 基于 THUDM/slime 的多教师在策略蒸馏实现移植。它复用 OPD 的“学生生成、教师打分、学生训练”闭环，但不是简单地把 `--rm-url` 扩展成多个 URL：

| 能力 | OPD | MOPD |
|---|---|---|
| 教师数量 | 单教师 | 一个或多个领域教师 |
| 样本路由 | 所有样本使用同一教师 | 通过 `metadata.mopd_domains` 按样本选择教师 |
| 蒸馏数据 | sampled-token log-prob | sampled-token、full-vocabulary 或 top-k+tail |
| 教师形态 | SGLang/已有 OPD 路径 | SGLang 服务或 Megatron checkpoint |
| 多教师聚合 | 不适用 | 只对该样本有效的教师取平均 |

`--use-mopd` 与 `--use-opd` 互斥。已有单教师 OPD 作业不需要改成 MOPD；只有确实需要领域路由、多教师平均或 full-vocabulary/top-k 蒸馏时才应启用 MOPD。当前文档中的配置均为纯蒸馏。

## 工作流程

一次 MOPD 训练迭代包含以下步骤：

1. 学生的 SGLang rollout engine 根据当前策略生成 response，并保存采样策略的 token log-prob；
2. 根据 `sample.metadata["mopd_domains"]` 选择该样本需要查询的教师；
3. SGLang 教师模式通过 HTTP 获取 sampled-token log-prob，以及 top-k 模式需要的 top-k token 与 log-prob；Megatron 教师模式则在训练进程中执行教师前向；
4. rollout 数据按照 domain 组织并传递给 Megatron actor；
5. actor 重新计算当前学生策略概率，计算 importance sampling 权重和蒸馏目标；
6. 多个有效教师的损失在样本内取平均，然后完成反向传播和权重同步。

关键数据流如下：

```text
student rollout
  ├─ response tokens
  └─ rollout_log_probs (sampling policy μ)
            │
            ▼
domain routing ──► teacher: math
            └────► teacher: geometry
                         │
                         ▼
       sampled log-probs / top-k / full logits
                         │
                         ▼
Megatron actor current policy πθ
  ├─ importance sampling correction
  ├─ per-domain distillation loss
  └─ average valid teachers per sample
```

## 蒸馏目标

通过 `--mopd-distill-type` 选择蒸馏方式。

### Token-level

`token_level` 只比较 rollout 实际采样 token 在教师策略和当前学生策略下的对数概率。两者的差异作为停止梯度的蒸馏优势，并结合 rollout 采样策略与当前学生策略之间的重要性采样校正，通过策略梯度形式更新学生模型。

该模式不需要传输或保存完整词表分布，计算、显存和通信开销最低，适合长序列或远端教师场景。但它只在实际采样到的 token 上提供监督，是对完整分布差异的采样估计，其方差和偏差会受到 rollout 样本质量与覆盖范围影响。

### Full vocabulary

`full_vocab` 在每个 response 位置上使用教师和学生的完整词表概率分布，计算从学生分布到教师分布的反向 KL 散度。它能够利用教师对所有候选 token 的相对偏好，提供信息最完整的分布级监督，不依赖 rollout 是否恰好采样到某个候选 token。

该模式需要保留每个 response 位置的完整词表 logits，显存、主机内存和数据传输开销随 response 长度与词表大小的乘积增长。当前只支持 `--mopd-teacher-mode megatron`，不支持远端 SGLang 教师。

当前端到端验证覆盖同构 Megatron 单教师的短步 full-vocabulary 链路；多教师 full-vocabulary 和长程收敛需要在相应教师 checkpoint 准备完成后单独验收。

### Top-k

`top_k` 是 full-vocabulary 蒸馏的稀疏近似。它保留教师在每个 response 位置上概率最高的若干候选 token，并将词表中其余 token 的总概率质量聚合为一个 tail bucket。训练时同时约束高概率候选 token 的细粒度分布和长尾部分的整体概率质量，从而在主要教师偏好不丢失的前提下降低存储与传输成本。

该模式的开销随 response 长度和 `k` 值的乘积增长。对于远端 SGLang 教师，返回的 top-k 值是基于完整词表 softmax 的 log-prob，可据此计算 top-k 与 tail 的概率质量；对于 Megatron 教师，代码从完整 logits 计算准确的教师 tail mass。

对于 SGLang 教师，推荐优先使用 `top_k`。`k` 越大，近似越准确，但教师请求响应、Ray 传输、CPU 内存和训练侧张量开销也越高。应根据词表大小、response 长度和集群带宽选择 `--mopd-topk-k`。

## 教师运行模式

### SGLang 教师

SGLang 模式适用于教师与学生架构不同，或教师无法与 actor 同时加载到训练资源中的场景。每个教师是一个独立的 `/generate` 服务。

教师配置示例：

```bash
--use-mopd
--mopd-teacher-mode sglang
--mopd-teachers '[
  {"name":"math_teacher","domain":"math","rm_url":"http://127.0.0.1:13141/generate"},
  {"name":"geometry_teacher","domain":"geometry","rm_url":"http://127.0.0.1:13142/generate"}
]'
```

URL 按以下优先级解析：

1. `--mopd-teachers` 每个条目中的 `rm_url`；
2. `MOPD_TEACHER_URLS` 环境变量中的 domain 到 URL 映射；
3. 公共的 `--rm-url`。

例如：

```bash
export MOPD_TEACHER_URLS='{
  "math":"http://127.0.0.1:13141/generate",
  "geometry":"http://127.0.0.1:13142/generate"
}'
```

纯蒸馏时，参数校验会自动选择以下函数：

```bash
--custom-rm-path slime.rollout.mopd.reward_func
--custom-reward-post-process-path slime.rollout.mopd.post_process_rewards
```

SGLang 返回值必须包含：

- 所有模式：`meta_info.input_token_logprobs`；
- top-k 模式：`meta_info.input_top_logprobs`。

请求失败时会按照 `--mopd-teacher-max-retries` 和 `--mopd-teacher-retry-delay` 重试。已经查询但最终失败的 domain 会生成 shape 一致的 `-inf` fallback，使该教师对 loss 的贡献为零，并避免不同 data-parallel rank 产生不同计算图。被 domain routing 主动排除的教师不会生成 fallback。

### Megatron 教师

Megatron 模式将教师 checkpoint 加载到训练进程，适合教师与学生使用兼容模型结构、且主机内存和设备调度能够容纳多模型权重的场景。

```bash
--use-mopd
--mopd-teacher-mode megatron
--mopd-teachers '[
  {"name":"math_teacher","domain":"math"},
  {"name":"code_teacher","domain":"code"}
]'
--mopd-teacher-loads \
  /path/to/math_teacher_torch_dist \
  /path/to/code_teacher_torch_dist
```

`--mopd-teacher-loads` 的顺序必须与 `--mopd-teachers` 一致，路径数量必须与教师数量相同。可选的 `--mopd-teacher-ckpt-steps` 也必须逐项对应。当前主线始终启用多标签 weights backuper，教师加载完成后会恢复 actor 的 checkpoint 临时参数。

## Domain routing

数据样本通过 metadata 指定教师领域：

```json
{
  "prompt": "...",
  "answer": "...",
  "metadata": {
    "mopd_domains": ["math"]
  }
}
```

规则如下：

- `mopd_domains` 可以是字符串或字符串列表；
- 未设置时查询全部教师；
- 只有一个教师时不需要路由，所有样本使用该教师；
- 未知 domain 会被过滤；如果过滤后没有有效 domain，则回退为查询全部教师；
- 一个样本路由到多个教师时，训练侧对有效教师的 loss 取平均。

建议在数据预处理阶段校验 domain 名称。拼写错误会触发“查询全部教师”的兼容回退，可能造成额外请求和错误训练信号。

## 多模态样本

文本样本优先使用预分词的 `input_ids` 查询教师。多模态样本使用：

```text
text = prompt + response
image_data = encoded images
```

教师使用自己的 processor 重新完成文本和图像 tokenization。不要把学生 processor 展开的 image-pad token 直接作为教师 `input_ids`，因为学生和教师的视觉 processor、图像网格或 placeholder 展开长度可能不同，这会导致 token 与 `image_grid_thw` 数量不一致。

因此，多模态 MOPD 需要：

- 打开 `--apply-chat-template`，保证 `sample.prompt` 已经是字符串；
- 数据中的图片路径或内容可以被 rollout worker 读取；
- 教师模型本身支持相应模态和 prompt 中的图像 placeholder。

### Ascend Qwen3-VL 部署注意事项

MOPD loss 本身不依赖特定视觉模型，但 Qwen3-VL non-colocate 作业还依赖通用的模型兼容链路：

1. Megatron→HF converter 必须能映射 vision、embedding、final norm 和 lm head，并正确去除 DDP/`language_model` 前缀；否则 actor→rollout 权重同步会以 `Unknown parameter name` 终止。
2. 当前 Megatron/Megatron-Bridge 版本必须与 Qwen3-VL rotary forward 的 packed-sequence 参数签名一致。
3. 如果 Ascend fast image processor 在 reshape 时报告 Tensor 维数超过设备上限，应同时为外部教师和 slime 管理的 rollout engine 使用原生 image processor。rollout 侧可通过 `--sglang-disable-fast-image-processor` 透传 SGLang 参数。
4. Megatron-Bridge 必须认识实际输出层模块类型；依赖版本过旧时，应升级依赖或使用项目维护的 docker patch，而不是在每个容器手工修改源码。

这些属于 Qwen3-VL/NPU 软件栈兼容，不应写进 MOPD loss 分支。正式部署应确保它们由仓库源码、镜像或依赖补丁提供，并在全新容器中做到无需私人共享 patch 即可运行。

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use-mopd` | false | 启用 MOPD；与 `--use-opd` 互斥 |
| `--mopd-teachers` | 无 | 非空 JSON 列表；每项必须有唯一的 `domain` |
| `--mopd-teacher-mode` | `sglang` | `megatron` 或 `sglang` |
| `--mopd-teacher-loads` | 无 | Megatron 教师 checkpoint 列表 |
| `--mopd-teacher-ckpt-steps` | 无 | 每个 Megatron 教师的 checkpoint step |
| `--mopd-distill-type` | `token_level` | `token_level`、`full_vocab` 或 `top_k` |
| `--mopd-topk-k` | 1024 | top-k 每个 token 保存的教师候选数 |
| `--mopd-eps-low` | 0.2 | importance sampling 有效区间下界 |
| `--mopd-eps-high` | 5.0 | importance sampling 有效区间上界 |
| `--mopd-sampling-logprobs-key` | `rollout_log_probs` | importance sampling 的采样策略概率来源 |
| `--mopd-teacher-max-retries` | 3 | 每个 SGLang 教师请求的最大尝试次数 |
| `--mopd-teacher-retry-delay` | 5.0 | 重试基础等待秒数；实际等待包含递增和随机抖动 |

`--mopd-teachers` 也可以通过 `MOPD_TEACHERS_JSON` 环境变量提供。

模式兼容关系：

| 蒸馏方式 | SGLang 教师 | Megatron 教师 | 主要开销 |
|---|---|---|---|
| `token_level` | 支持 | 支持 | sampled-token log-prob |
| `top_k` | 支持 | 支持 | $O(Rk)$ 教师数据 |
| `full_vocab` | 不支持 | 支持 | $O(RV)$ 教师 logits |

## 配置示例

### 纯 top-k 蒸馏

```bash
python train.py \
  <model-and-cluster-args> \
  --use-mopd \
  --mopd-teacher-mode sglang \
  --mopd-distill-type top_k \
  --mopd-topk-k 128 \
  --mopd-teachers '[
    {"name":"math_teacher","domain":"math","rm_url":"http://127.0.0.1:13141/generate"},
    {"name":"geometry_teacher","domain":"geometry","rm_url":"http://127.0.0.1:13142/generate"}
  ]' \
  --mopd-sampling-logprobs-key rollout_log_probs \
  --mopd-eps-low 0.2 \
  --mopd-eps-high 5.0
```

## 监控指标

不同蒸馏模式的核心指标不同，不能直接比较绝对值或符号。

| 指标 | 模式 | 含义 |
|---|---|---|
| `rollout/mopd_advantages` | token-level | rollout 数据中的 sampled-token MOPD advantage |
| `train/mopd_advantage_mean` | token-level | 训练侧聚合的 MOPD advantage |
| `train/mopd_reverse_kl/<domain>` | token-level | 每个教师在采样 token 上的 log-prob 差 |
| `train/mopd_topk_kl` | top-k | 学生到教师的 top-k+tail 近似 KL |
| `train/mopd_topk_kl/<domain>` | top-k | 每个 domain 的近似 KL |
| `train/mopd_fv_kl` | full vocabulary | 学生到教师的完整词表 KL |
| `train/mopd_fv_kl/<domain>` | full vocabulary | 每个 domain 的完整词表 KL |
| `train/mopd_is_weight_mean` | 全部 | importance sampling 权重均值 |
| `train/mopd_is_nonzero_frac` | 全部 | 未被阈值置零的 token 比例 |
| `train/train_rollout_logprob_abs_diff` | 全部 | actor 与 rollout engine 的 sampled-token log-prob 一致性 |

`rollout/rollout_log_probs` 和 `rollout/log_probs` 都是学生概率：前者来自 rollout engine，后者来自训练后端重算。它们都不是教师概率。

随机混合多 domain 数据时，不同 batch 的 domain 数量和样本数量会变化，因此 raw loss 和 KL 可能明显波动。建议同时查看 per-domain 指标，并使用 10～20 步移动平均观察趋势。若需要降低训练方差，应使用固定 domain 配比的 batch sampler，而不是只在可视化层隐藏波动。

## 验证方法

### CPU 单元测试

MOPD 单元测试应覆盖 payload、URL 解析、domain routing、SGLang 返回值对齐、top-k padding、失败教师 fallback、三类 loss、IS 门控、嵌套 domain 字段切片、Sample 序列化，以及 top-k+tail KL 数值和梯度正确性：

```bash
pytest -q tests/tests_npu/ut/test_mopd_npu.py
```

特别是 `DataIterator` 的测试不能省略。MOPD 字段形如 `dict[domain, list[sample]]`，如果沿用普通 list 的 `vals[i]` 切片方式，会把 domain 字符串字典当成整数索引字典并触发 `KeyError: 0`。

### Ascend NPU 冒烟测试

CPU 测试不能替代真实的 Ray + SGLang + Megatron + NPU 闭环。参考 OPD NPU 用例，MOPD ST 至少需要：

- 启动两个不同端口的 teacher，并分别执行 health check；
- 为 actor、rollout 和外部 teacher 使用互不重叠的 NPU；
- 使用 CI 已缓存的小模型和数据集，限制为 2 个 rollout/训练 step；
- 执行 `top_k` 路径并让两个 domain 都产生数据；
- 在 `finally` 中按 PID 清理本用例启动的 teacher；
- 保存 teacher 和训练日志，失败时输出最后一段日志。

推荐的 8-NPU 小模型布局为 4 卡 actor、2 卡 rollout、2 个单卡 teacher。冒烟测试重点是接口和数据闭环，不承担长程收敛验证。

### 端到端验收

端到端测试至少应确认：

1. 所有教师 `/generate` 请求持续返回 HTTP 200；
2. 日志中没有缺少 `input_token_logprobs` 或 `input_top_logprobs` 的警告；
3. actor forward/backward 和权重同步可以连续完成多个 step；
4. `train/mopd_is_weight_mean` 接近 1；
5. `train/mopd_is_nonzero_frac` 接近 1；
6. `train/train_rollout_logprob_abs_diff` 保持稳定且足够小；
7. `train/mopd_topk_kl` 或 `train/mopd_fv_kl` 为有限非负值；长程作业再检查 10～20 step 移动平均趋势；
8. 文本和多模态 domain 都实际产生非零的 per-domain KL。

只看到 teacher 进程 ready 或 step 计数增加，不能证明 MOPD 生效。必须同时看到教师字段进入 batch、蒸馏指标、有限梯度、optimizer step 和训练后的 actor→rollout 权重同步。

### 已验证配置的判定方法

在 Qwen3-VL 8B 双教师 token-level 复现中，旧版与新主线前三步 `rollout/mopd_advantages` 的绝对差分别约为 `0.003393`、`0.000290` 和 `0.003130`。这证明新主线对齐了旧版计算语义和初始数值尺度，但不意味着随机 rollout 曲线应逐点相同。

对 top-k 配置，验收重点应切换为 `train/mopd_topk_kl` 和 `train/mopd_topk_kl/<domain>`。不要因为日志里不再出现 `rollout/mopd_advantages` 就判断 MOPD 没有执行；该指标只属于 token-level 路径。

当前已完成的端到端参数覆盖还包括：

- SGLang 双教师 domain 路由、URL 环境变量与公共 URL 回退；
- `token_level`、`top_k` 和同构 Megatron 单教师 `full_vocab`；
- Megatron 双 checkpoint 的 `--mopd-teacher-loads` 与 `--mopd-teacher-ckpt-steps`；
- `--mopd-topk-k=128/1024`、两种 sampling log-prob key 和 importance-sampling 上下界；
- 教师请求重试、部分教师持续失败时的零贡献降级。

## 故障排查

### 教师返回值缺少 log-prob

检查 `rm_url` 是否指向教师而不是学生 rollout server，并确认请求包含 `return_logprob=true`。top-k 模式还需要 `top_logprobs_num`，响应中必须存在 `input_top_logprobs`。

### 教师数据长度与 response 不一致

MOPD 会去掉 SGLang 返回的第一个 prompt 占位项，再从尾部对齐到 `response_length`。返回过短时会使用 `-inf` padding。频繁出现该警告通常意味着 tokenizer、prompt 拼接或 SGLang 返回协议不一致，不应长期依赖 padding。

### 多模态 processor 报 image grid 或 placeholder 错误

确认多模态请求使用 `text + image_data`，并让教师 processor 自己重新 tokenize。还应确认 `sample.prompt` 是已经应用 chat template 的字符串。

### 训练在不同 TP/DP rank 间卡住

确认每个 microbatch 都为所有配置 domain 产生相同的 metric key。对于某个 TP shard 没有本地 top-k token 的情况，也必须参加 collective；不能因为本地 top-k 全是 `-inf` 就跳过 KL 函数。

### KL 或 loss 曲线跳变

先分别查看每个 domain 的 KL 和每个 batch 的 domain 组成。少量高难样本、某个 batch 不含稀有 domain、response 长度变化，都可能导致 raw KL 波动。只有同时出现 NaN、importance sampling 大量失效、rollout/actor log-prob 差持续扩大时，才更像数值或权重同步问题。

### 内存或网络压力过大

- SGLang 教师优先使用 `top_k`，避免传输 full vocabulary；
- 减小 `--mopd-topk-k` 或 response 最大长度；
- 不要在长时间训练中持续开启完整 rollout/train debug dump；
- 根据教师吞吐调整并发请求数和 teacher tensor parallel size。

## 实现位置

MOPD 的主要实现分布在：

- `slime/rollout/mopd.py`：教师请求、domain 路由、返回值提取和 reward 包装；
- `slime/utils/arguments.py`：参数定义、模式兼容性校验和 reward 函数自动配置；
- `slime/utils/ppo_utils.py`：full-vocabulary 和 top-k reverse KL；
- `slime/backends/megatron_utils/loss.py`：三种蒸馏 loss、importance sampling 和指标；
- `slime/backends/megatron_utils/actor.py`：Megatron 教师前向及 SGLang top-k 数据转换；
- `slime/backends/megatron_utils/data.py`：对 `dict[domain, list[sample]]` 字段递归执行 micro-batch 切片；
- `slime/backends/megatron_utils/model.py`：把各类 MOPD 字段接入训练 forward/batch keys；
- `slime/ray/rollout.py`：MOPD 字段从 Sample 到训练 batch 的传递；
- `slime/utils/types.py`：Sample 中的 MOPD 数据结构。

移植上游实现时，应以目标分支当前接口为准解决冲突。特别是 loss 归约使用的 mask-sum 字段、rollout batch 结构和权重同步实现可能随主线演进；不能为了让文本 diff 与旧提交一致而覆盖目标分支的新接口，也不能直接删除源分支的 MOPD 分支逻辑。
