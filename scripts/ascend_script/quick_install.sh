#!/bin/bash
set -x
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
MEGATRON_COMMIT=1dcf0dafa
MegatronAdaptor_COMMIT=f707a3b6
TransformerEngineNPU_COMMIT=47d60449

source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, got {sys.version}"'

echo "1. install SGLang from source"
git clone -b v0.5.13 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject.toml python/pyproject.toml.backup
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0 setuptools==81.0.0

echo "3. install prebuilt CANN 9.0 wheels"
wget --no-check-certificate https://gitcode.com/gcw_8jFUVqc5/sglang-kernel-npu-builds/releases/download/py312-cann9.0-a3/sgl-kernel-npu-torch2.10.0-py312-cann9.0-a3-aarch64.zip -O sgl-kernel-npu-cp312.zip
rm -rf sgl-kernel-npu-cp312
unzip -q sgl-kernel-npu-cp312.zip
pip install \
    sgl-kernel-npu-cp312/torch_memory_saver-*-cp312-cp312-linux_aarch64.whl \
    sgl-kernel-npu-cp312/sgl_kernel_npu-*-cp312-cp312-linux_aarch64.whl \
    sgl-kernel-npu-cp312/deep_ep-*-cp312-cp312-linux_aarch64.whl
# Link the DeepEP extension from the extracted bundle into site-packages.
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
ln -sf \
    "$(pwd)/sgl-kernel-npu-cp312/lib/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so" \
    "${SITE}/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so"

# (Optional) Confirm whether the import can be successfully
python -c "import deep_ep; print(deep_ep.__path__)"

echo "4. install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout 89eb10887887bc74853f89a4de258c0702932a1c
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout dev_rl && cd ..
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
pip install -e . && cd ..

echo "5. install MA&TE"
git clone https://gitcode.com/Ascend/MegatronAdaptor.git && \
cd MegatronAdaptor/ && git checkout ${MegatronAdaptor_COMMIT} && \
pip install -e . && cd ..
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git&& \
cd TransformerEngineNPU/ && git checkout ${TransformerEngineNPU_COMMIT} && \
pip install -e . && cd ..

echo "6. install triton_ascend"
pip uninstall triton triton_ascend -y
pip install triton_ascend==3.2.1

echo "7. install slime-ascend"
cd slime-ascend/
pip install -e .

echo "8. apply npu patches"
cd ../sglang
git am ../slime-ascend/docker/npu_patch/v0.3.0/sglang/*
cd ../Megatron-LM/
git am ../slime-ascend/docker/npu_patch/v0.3.0/megatron/*
cd ../TransformerEngineNPU/
git am ../slime-ascend/docker/npu_patch/v0.3.0/transformer_engine_npu/*
cd ../Megatron-Bridge/
git am ../slime-ascend/docker/npu_patch/v0.3.0/megatron-bridge/*
cd ../mbridge/
git am ../slime-ascend/docker/npu_patch/v0.3.0/mbridge/*
echo "9. return to slime-ascend directory"
cd ../slime-ascend/
# Create folder to store logs
mkdir -p logs
