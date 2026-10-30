# slime-ascend

Huawei Ascend NPU adaptation repository for slime

[中文](./README_ascend_zh.md)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime-ascend** is the Huawei Ascend NPU adaptation version of slime, an LLM post-training framework designed for RL scaling. For slime's original features and introduction, please refer to [Original slime Documentation](./README_zh.md).

## Latest News

- [ Mar 19, 2026 ]: 🚀 slime-ascend now supports GRPO algorithm for [GLM-4.7-Flash](./docs/ascend_tutorial/examples/glm4.7-30B-A3B.md), [Qwen3-8B](./docs/ascend_tutorial/examples/qwen3-8B.md), [Qwen3-VL-8B](./docs/ascend_tutorial/examples/qwen3-vl-8B.md) models!

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Development Guide](#development-guide)
- [Contributing](#contributing)
- [Security Statement & Disclaimer](#security-statement)
- [FAQ & Acknowledgments](#faq--acknowledgments)

The overall architecture of slime-ascend can be found in [slime Architecture Overview](./README_zh.md#架构总览).

### Hardware Support

- **Atlas 800T A3**
- **Atlas 800T A2**
- Recommended configuration: CANN 8.5.0 or higher
- Recommended environment: Linux arm64

## Quick Start

- **Docker Build Guide**: Please refer to [Ascend Quick Start Guide - Docker Method](./docs/ascend_tutorial/get_started/quick_start.md#方式一docker推荐)
- **Custom Installation Guide**: Please refer to [Ascend Quick Start Guide - Local Installation Script](./docs/ascend_tutorial/get_started/quick_start.md#方式二本地安装脚本)

- For detailed model and algorithm support information, please refer to [Ascend Quick Start Guide - Usage Guide](./docs/ascend_tutorial/get_started/quick_start.md#使用指南)

- We also provide some usage examples not covered in the quick start, please check [ascend_script](./scripts/ascend_script/).

## Features

Currently, we support standard training-inference separation configuration on NPU. For more supported features, please see: [feature_introduction](./docs/ascend_tutorial/get_started/feature_introduction.md).

## Development Guide

After completing the NPU environment setup mentioned above, you can seamlessly transition to the standard model operation pipeline. For specific commands and parameter settings for training, saving, and evaluation, please refer directly to the original main repository documentation:

- **Training Process Guide**: For configuration and launching of model training methods, please refer to slime [Quick Start Guide](./docs/zh/get_started/quick_start.md#训练脚本与参数概览).
- **Saving Process Guide**: For model checkpoint saving, conversion, and loading processes, please refer to slime [Quick Start Guide](./docs/zh/get_started/quick_start.md#模型权重转换).
- **Evaluation Process Guide**: For evaluation and scoring process of model responses after training, please refer to slime [Quick Start Guide](./docs/zh/get_started/quick_start.md#eval_args-评估参数).

Parameters are divided into three categories:
1. **megatron parameters**: slime reads all parameters set in megatron from `PYTHONPATH`, which can be configured by passing arguments like `--tensor-model-parallel-size 2`;
2. **sglang parameters**: Supports all parameters of sglang installed in the environment. These parameters need to start with `--sglang`, for example `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3. **slime's own parameters**: See: [slime/utils/arguments.py](slime/utils/arguments.py)

## Contributing

If you have feature suggestions, performance tuning, or usage experience feedback, please submit an Issue / PR 😊

- Use [pre-commit](https://pre-commit.com/) to ensure submitted code style:

  ```bash
  apt install pre-commit -y
  pre-commit install

  # Run pre-commit to ensure code style
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- For debugging tips, please refer to [debug guide](docs/zh/developer_guide/debug.md)

## Security Statement

[slime-ascend Security Statement](./docs/ascend_tutorial/SECURITYNOTE.md)

## Disclaimer

### To slime-ascend Users

1. The models provided by slime-ascend are for non-commercial purposes only.
2. For each model, the slime-ascend platform only suggestively recommends datasets that can be used for training. Huawei does not provide any datasets. If you use these datasets for training, please pay special attention to comply with the corresponding dataset's License. If you encounter infringement disputes due to using datasets, Huawei assumes no responsibility.
3. If you discover any issues (including but not limited to functional issues, compliance issues) while using slime-ascend models, please submit an issue on gitcode, and we will promptly review and resolve it.
4. Megatron and other third-party open-source software that MindSpeed functionality depends on are provided and maintained by third-party communities. Fixes for issues caused by third-party open-source software depend on the contributions and feedback from relevant communities. You should understand that the MindSpeed repository does not guarantee fixes for issues in third-party open-source software itself, nor does it guarantee testing and correction of all vulnerabilities and errors in third-party open-source software.

### To Dataset Owners

If you do not want your dataset to be mentioned in models within slime-ascend, or wish to update the description of your dataset in slime-ascend models, please submit an issue on gitcode, and we will delete or update your dataset description according to your issue request. We sincerely thank you for your understanding and contribution to slime-ascend.

### License Statement

For models provided by slime-ascend, if a License exists in the model directory, that License shall prevail. If no License exists in the model directory, it is licensed under the Apache 2.0 license. The corresponding license text can be found in the slime-ascend root directory.

## FAQ & Acknowledgments

- For common questions, please see [Q&A](docs/zh/get_started/qa.md)
- Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch, MindSpeed, Ascend community, etc.

slime-ascend is jointly contributed by the following departments of Huawei Company and Ascend ecosystem partners:

Huawei Company:
- Computing Product Line

Thanks for every PR from the community.