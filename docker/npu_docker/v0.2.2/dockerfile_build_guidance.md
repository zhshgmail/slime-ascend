# Ascend Dockerfile Build Guidance

Last updated: 04/26/2026.


## 镜像获取 & 公开镜像地址

昇腾相关镜像请参考 Dockerfile 自行构建。


## Dockerfile 命名规范

Dockerfile 的命名遵循模板：`Dockerfile[.{芯片信息}.{操作系统}.{其他字段}]`，中括号内为可选字段（字段顺序可调整），字段间连接符使用 `.`，字段内连接符使用 `-`。


## 镜像 Tag 命名规范

镜像 tag 的命名推荐遵循模板：`{版本号}[-{芯片信息}-{操作系统}-{Python版本}-{架构类型}-{其他字段}]`，中括号内为可选字段（字段顺序可调整），字段内、字段间连接符均使用 `-`。


## 镜像硬件支持

Atlas 800T A3

Atlas 800T A2


## 镜像内各组件版本信息清单

| 组件 | A2 版本 | A3 版本 |
| ---- | ------- | ------- |
| 基础镜像 | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |
| CANN | 8.5.0 | 8.5.0 |
| torch | 2.8.0 | 2.8.0 |
| torch_npu | 2.8.0.post2 | 2.8.0.post2 |
| torchvision | 0.23.0 | 0.23.0 |
| Megatron-LM | 3714d81d | 3714d81d |
| Megatron-Bridge | dev_rl | dev_rl |
| MindSpeed | fc63de5c | fc63de5c |
| triton-ascend | 3.2.0 | 3.2.0 |
| mbridge | 89eb1088 | 89eb1088 |
| SGLang | v0.5.8 (sglang-slime branch) | v0.5.8 (sglang-slime branch) |
| sgl-kernel-npu | 2026.03.01.post1 | 2026.02.01 |
| transformers | 5.0.0 | 5.0.0 |
| nvidia-modelopt | >=0.37.0 | >=0.37.0 |


## Dockerfile构建镜像脚本清单

| 设备类型 | CANN版本 | 参考文件 |
| -------- | -------- | -------- |
| A3 | 8.5.0 | [Dockerfile.a3.ubuntu22.04.cann850.latest](../v0.2.2/Dockerfile.a3.ubuntu22.04.cann850.latest) |
| A2 | 8.5.0 | [Dockerfile.a2.ubuntu22.04.cann850.latest](../v0.2.2/Dockerfile.a2.ubuntu22.04.cann850.latest) |


## 前置准备

无需额外准备本地文件，所有依赖都会从网络自动下载。


## 镜像构建命令示例

```bash
# Navigate to the directory containing the Dockerfile
cd {slime-ascend-root-path}

# Build the image for A3
docker build -f Dockerfile.a3.ubuntu22.04.cann850.latest -t slime-ascend:8.5.0-a3-ubuntu22.04-py3.11-latest .

# Build the image for A2
docker build -f Dockerfile.a2.ubuntu22.04.cann850.latest -t slime-ascend:8.5.0-a2-ubuntu22.04-py3.11-latest .
```


## 声明

slime-ascend 中提供的 Ascend 相关 Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通，谢谢。
