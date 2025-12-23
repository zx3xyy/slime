#!/bin/bash

set -ex

# create conda
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

micromamba create -n rl_mast python=3.12 pip -c conda-forge -y
micromamba activate rl_mast
export CUDA_HOME="$CONDA_PREFIX"

export BASE_DIR=${BASE_DIR:-"/home/chengze/work/mast"}
cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n rl_mast cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n rl_mast -c conda-forge cudnn -y

# prevent installing cuda 13.0 for sglang
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout 5e2cda6158e670e64b926a9985d65826c537ac82
# Install the python packages

# Not install with -e to use conda-on-mast
# pip install -e "python[all]"
pip install "./python[all]"


pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
# MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation
pip install --no-build-isolation --no-binary flash-attn flash-attn==2.7.4.post1 --no-deps

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
# pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
# Have to rebuild, otherwise it downloads a cu13
MAX_JOBS=128 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@769ed77

pip install flash-linear-attention==0.4.0
# NVCC_PREPEND_FLAGS="" otherwise there is a bug with nvcc
NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    # pip install -e .
    pip install .

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
cd $BASE_DIR
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout core_v0.14.0 && \
  pip install .

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29

# install slime and apply patches

# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone  https://github.com/THUDM/slime.git
  cd slime/
  export SLIME_DIR=$BASE_DIR/slime
  # pip install -e .
  pip install -r requirements.txt
  pip install .
else
  export SLIME_DIR=$BASE_DIR/
  # pip install -e .
  pip install .
fi

export SLIME_DIR=$BASE_DIR/slime

# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.6/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.6/megatron.patch
