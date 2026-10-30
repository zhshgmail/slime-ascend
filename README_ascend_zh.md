# slime-ascend

slime 的昇腾适配开发仓

[English ](./README_ascend.md) 

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime-ascend** 是为 RL scaling 设计的 LLM post‑training 框架 slime 的华为昇腾 NPU 适配版本。关于 slime 本身的特性与介绍，请参考 [slime 原版文档](./README_zh.md)。

## 最新消息

- [ Mar 19, 2026 ]: 🚀 slime-ascend 已支持 [GLM-4.7-Flash](./docs/ascend_tutorial/examples/glm4.7-30B-A3B.md)、[Qwen3-8B](./docs/ascend_tutorial/examples/qwen3-8B.md) 、[Qwen3-VL-8B](./docs/ascend_tutorial/examples/qwen3-vl-8B.md) 等模型的GRPO算法！
## 目录

- [快速开始](#快速开始)
- [特性介绍](#特性介绍)
- [开发指南](#开发指南)
- [欢迎贡献](#欢迎贡献)
- [安全声明及免责声明](#安全声明)
- [常见 Q&A 与致谢](#常见-qa-与致谢)

slime-ascend整体架构可参考slime原仓架构内容[slime 架构总览](./README_zh.md#架构总览)

### 硬件支持

- **Atlas 800T A3**
- **Atlas 800T A2**
- 推荐配置：CANN 8.5.0 或更高版本; Ascend HDK 25.5.0 或更高版本；
- 推荐环境：Linux arm64


## 快速开始

- **Docker 构建指南**：请参考 [昇腾快速开始指南 - Docker 方式](./docs/ascend_tutorial/get_started/quick_start.md#方式一docker推荐)
- **自定义安装指南**：请参考 [昇腾快速开始指南 - 本地安装脚本](./docs/ascend_tutorial/get_started/quick_start.md#方式二本地安装脚本)

- 详细的模型与算法支持情况，请参考 [昇腾快速开始指南 - 使用指南](./docs/ascend_tutorial/get_started/quick_start.md#使用指南)

- 我们还提供了一些未在快速开始中覆盖的使用示例，请查看 [ascend_script](./scripts/ascend_script/)。

## 特性介绍

目前我们在NPU上支持标准训推分离配置，更多支持特性请见：[feature_introduction](./docs/ascend_tutorial/get_started/feature_introduction.md)。

## 开发指南
  在完成了上述 NPU 环境的搭建后，可以无缝衔接至标准的模型操作链路。关于训练、保存与评估的具体指令和参数设定，请直接参考原主仓文档：
- **训练流程指导**：关于模型训练方式进行配置与启动，请参考slime [快速开始指南](./docs/zh/get_started/quick_start.md#训练脚本与参数概览)。
- **保存流程指导**：关于模型 Checkpoint 的保存、转换与加载流程，请参考slime [快速开始指南](./docs/zh/get_started/quick_start.md#模型权重转换)。
- **评估流程指导**：关于训练后模型回答的评估打分流程，请参考slime [快速开始指南](./docs/zh/get_started/quick_start.md#eval_args-评估参数)。

  其中参数分为三类：
1. **megatron 参数**：slime 会读取 `PYTHONPATH` 中的 megatron 里设置的所有参数，可以通过传入如 `--tensor-model-parallel-size 2` 的方式配置 megatron；
2. **sglang 参数**：支持环境中安装的 sglang 的所有参数，这些参数需要以 `--sglang` 起始，例如 `--mem-fraction-static` 需要通过 `--sglang-mem-fraction-static` 传入。
3. **slime 自身的参数**：请见：[slime/utils/arguments.py](slime/utils/arguments.py)

## 欢迎贡献
 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR 😊

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install

  # 运行 pre-commit 保证代码风格
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- 调试技巧请参考 [debug 指南](docs/zh/developer_guide/debug.md)

## 安全声明

[slime-ascend 安全声明](./docs/ascend_tutorial/SECURITYNOTE.md)

## 免责声明

### 致slime-ascend使用者

1. slime-ascend 提供的模型仅供您用于非商业目的。
2. 对于各模型，slime-ascend 平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的 License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 slime-ascend 模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在 GitCode 提交 issue，我们将及时审视并解决。
4. MindSpeed 功能依赖的 Megatron 等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

### 致数据集所有者

如果您不希望您的数据集在 slime-ascend 中的模型被提及，或希望更新 slime-ascend 中的模型关于您的数据集的描述，请在 GitCode 提交 issue，我们将根据您的 issue 要求删除或更新您的数据集描述。衷心感谢您对 slime-ascend 的理解和贡献。

### License声明

slime-ascend 提供的模型，如模型目录下存在 License 的，以该 License 为准。如模型目录下不存在 License 的，以 Apache 2.0 许可证许可，对应许可证文本可查阅 slime-ascend 根目录。

## 常见 Q&A 与致谢

- 常见问题请见 [Q&A](docs/zh/get_started/qa.md)
- 特别感谢以下项目 & 社区：SGLang、Megatron‑LM、mbridge、OpenRLHF、veRL、Pai-Megatron-Patch、MindSpeed、昇腾社区等。

slime-ascend 由华为公司的下列部门以及昇腾生态合作伙伴联合贡献：

华为公司：
- 计算产品线

感谢来自社区的每一个PR。