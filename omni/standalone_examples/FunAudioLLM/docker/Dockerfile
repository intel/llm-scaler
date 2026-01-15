# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.45.5.4

ARG https_proxy
ARG http_proxy

# Ensure library path includes Intel optimized Torch libraries
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"

COPY ./patch/cosyvoice.patch /tmp/

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
# Install system dependencies and Python 3.12
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl ffmpeg git libsndfile1 libsm6 libxext6 libgl1 lsb-release numactl wget vim sox libsox-dev linux-libc-dev && \
# Install Intel GPU runtime packages
    apt-get install -y intel-oneapi-dpcpp-ct=2025.2.0-517 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
# Configure Pip to allow system package modification in container
    python3 -m pip config set global.break-system-packages true && \
    pip install --upgrade --ignore-installed pip setuptools wheel && \
# Install Intel-optimized PyTorch stack for XPU
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu

# Prepare CosyVoice (Speech Synthesis)
RUN mkdir -p /audio && \
    cd /audio && \
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git checkout 2d6bb9bd80aad0f46c4324dd0b5ee23a30b090a3 && \
    git submodule update --init --recursive && \
    \
    git apply /tmp/cosyvoice.patch && \
    \
    pip install -r requirements.txt

# Prepare FunASR (Speech Recognition)
RUN cd /audio && \
    git clone https://github.com/modelscope/FunASR.git && \
    cd FunASR && \
    git checkout 36656aa8f7f30cece85ceda7e2c6ec5952a59925 && \
    pip3 install -e . && \
    pip install torchaudio-augmentations humanfriendly positional_encodings

# Reinstall the XPU-specific Triton last to fix any namespace pollution
RUN pip uninstall -y triton pytorch-triton-xpu && \
    pip install pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu

COPY ./cosyvoice3_example.py /audio/CosyVoice/
COPY ./Fun_ASR_Nano_2512_example.py /audio/FunASR/

WORKDIR /audio