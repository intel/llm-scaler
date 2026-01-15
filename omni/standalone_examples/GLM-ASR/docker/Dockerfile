# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.45.5.4

ARG https_proxy
ARG http_proxy
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"

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

# Prepare GLM-ASR
RUN mkdir /audio && \
    cd /audio && \
    git clone https://github.com/zai-org/GLM-ASR.git && \
    cd GLM-ASR && \
    git checkout ecdd2c1d28ae40a6f2cec4647d4b23241fcffa8as && \
    pip install transformers==4.51.3

WORKDIR /audio/GLM-ASR