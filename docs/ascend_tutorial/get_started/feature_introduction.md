## 特性介绍

我们目前在NPU上支持slime框架中以下特性的使用：

- [On-Policy Distillation](#opdon-policy-distillation)
- [TIS](#tistruncated-importance-sampling)
- [Routing Reply](#moe-routing-reply)
- [Context Parallel](#context-parallel)
- [Retool](#Retool)

高级特性在 [advanced](../advanced/) 进行详细介绍:
- [pd-disaggregation](../advanced/pd-disaggregation.md)


## OPD（On-Policy Distillation）

在线蒸馏让学生模型在训练过程中实时获取教师模型对同一条生成结果的 token 级对数概率（log probabilities），通过拟合教师的输出分布来学习教师的知识，提升模型质量。教师模型通常部署为单独的推理服务。

启用 OPD 需在训练脚本中配置以下奖励模型相关参数 `RM_ARGS` ：

```bash
RM_ARGS=(
   --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
   --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

# ...
ray job submit ... \
   -- python3 train.py \
   ${RM_ARGS[@]}
```

- `--custom-rm-path`：指向自定义奖励函数。该函数将样本的完整 token 序列（prompt + response）发送至教师模型服务，请求返回每个 token 的 log probability，详细内容可参考 [on_policy_distillation.py](../../../examples/on_policy_distillation/on_policy_distillation.py)。
- `--custom-reward-post-process-path`：指向后处理函数。该函数从教师模型的返回结果中提取 response 部分的 token 级 log probabilities，并存入 sample.teacher_log_probs，供训练引擎在计算蒸馏损失（通常为 KL 散度）时使用。
- `--rm-url`：教师模型的推理服务地址（例如 SGLang Server 部署的 /generate 端点），用于实时获取教师模型的输出概率。

更多信息可以参考 [run-glm4.7-30B-opd.sh](../../../examples/on_policy_distillation/run-glm4.7-30B-opd.sh)。
> 💡 **提示**：教师模型与学生模型使用相同 tokenizer 对应的 token id 序列作为输入，因此两者必须是基于相同词表的模型。教师模型不生成新 token，仅对输入序列的每个 token 返回 log probability。

---

## TIS（Truncated Importance Sampling）

TIS 是一种 off-policy 校正技术，针对 rollout 阶段与 train 阶段之间的策略版本不匹配（training-inference mismatch），通过对旧策略样本施加重要性采样权重并截断，在限制方差的同时提高训练稳定性。

其原理可以详见 [Truncated Importance Sampling](https://ionides.github.io/pubs/ionides08-jcgs.pdf)、 [Off-Policy RL 博客](https://fengyao.notion.site/off-policy-rl) 和 [Rollout Correction Methods.md](../../../examples/train_infer_mismatch_helper/README.md)。

要启用 TIS，需要在 `GRPO_ARGS` 中开启 `--use-tis`：

```bash
GRPO_ARGS=(
   --use-tis
   --grpo-clip 0.2
   ...
)
```

---

## MOE Routing Reply

在 MoE 模型的强化学习训练中，路由机制的不稳定性是一个核心挑战：训练的前向与反向传播路由可能不一致，且训练和推理阶段的路由分布也存在显著偏差，严重时会导致训练崩溃。

可以通过在训练时重放 rollout 阶段的路由来稳定 MoE RL 训练，启用 Routing Reply 需要在 `SGLANG_ARGS` 中加入：

```bash
SGLANG_ARGS=(
   --use-rollout-routing-replay
   ...
)
```

可以参考[customization.md](https://gitcode.com/Ascend/slime-ascend/blob/main/docs/zh/get_started/customization.md#18-moe-路由重放)

> 💡 **注意：当前sglang-router0.3.2存在已知[Bug](https://github.com/THUDM/slime/issues/1909)，建议访问[sglang-router仓库](https://github.com/zhuzilin/sgl-router)获取最新release或进行源码安装**

---

## Context Parallel

支持 Megatron 上下文并行（Context Parallel），将超长序列沿 token 维度切分到多张 NPU 上并行计算，突破单卡显存对序列长度的限制。

可配合长上下文数据集（如 128K token）进行训练与推理，用于长文档理解、多轮对话、代码库分析等需要处理超长输入的场景。

启用方式：在训练脚本中配置 Rollout 的长度限制和 Megatron 上下文并行 CP 参数。

```bash
ROLLOUT_ARGS=(
   --rollout-max-response-len $((1024 * 32))   # 最大回复长度
   --rollout-max-prompt-len $((1024 * 128))    # 最大输入长度
   ...
)

PERF_ARGS=(
   --context-parallel-algo megatron_cp_algo
   --context-parallel-size 4
   ...
)
```

> 💡 **提示**：上下文并行通常与张量并行（--tensor-model-parallel-size）结合使用，请根据总 NPU 数量和序列长度合理分配并行维度。

---

## Retool

面向工具调用、代码执行等多步交互的智能体场景，使用 ReTool 框架，让模型在 rollout 过程中自由调用外部工具，并根据工具使用结果计算奖励。

启用 Retool 需在训练脚本中配置自定义生成函数与奖励函数，并关闭默认的聊天模板：

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

ROLLOUT_ARGS=(
   ... # 原有其他 ROLLOUT 参数
   # 注意：此处需要删除 --apply-chat-template
)
```

- `--custom-generate-function-path`：指向基于 ReTool 实现的自定义生成函数，例如 `generate_with_retool.generate`。该函数通过 ReTool 框架控制完整的 rollout 过程：prompt 构建、工具调用循环、停止条件判断等，详细内容可参考 [generate_with_retool.py](../../../examples/retool/generate_with_retool.py)。

- `--custom-rm-path`：指向自定义奖励函数，负责对工具使用结果（如是否成功调用、返回值正确性等）进行打分。

更多信息可以参考 [retool_glm4.7_flash_rl_npu.sh](../../../examples/retool/retool_glm4.7_flash_rl_npu.sh)。

> ⚠️ **注意**：需从 ROLLOUT_ARGS 中删除 --apply-chat-template：因为 ReTool 的生成函数内部会自行处理工具调用格式和消息拼接，无需 slime 再应用默认的聊天模板。