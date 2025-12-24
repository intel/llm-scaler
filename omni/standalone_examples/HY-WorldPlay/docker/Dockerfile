# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.45.5.4

ARG https_proxy
ARG http_proxy
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"

COPY ./patches/hyworldplay_for_multi_arc.patch /tmp/

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
# Install dependencies and Python 3.12
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev && \
    # Install Intel GPU runtime packages
    apt-get install -y intel-oneapi-dpcpp-ct=2025.2.0-517 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip config set global.break-system-packages true && \
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu && \
    # Install xDit related dependencies
    mkdir /llm && \
    cd /llm && \
    git clone https://github.com/feifeibear/long-context-attention.git && \
    cd long-context-attention && \
    git checkout fc5d55e61b78b3102fd824bea1791cf406cc2a4b && \
    git apply /tmp/yunchang_for_multi_arc.patch && \
    pip install -e . && \
    cd /llm && \
    git clone https://github.com/xdit-project/xDiT.git && \
    cd xDiT && \
    git checkout fb8fb0e437a8745b9629020759de31d1626a4a7b && \
    git apply /tmp/xdit_for_multi_arc.patch && \
    pip install -e .

RUN cd /llm && \
    git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git && \
    cd ./HY-WorldPlay && \
    git checkout c914f2a3a8ec0cb45f702d2078f530d5191ec9e3 && \
    git apply /tmp/hyworldplay_for_multi_arc.patch && \
    pip install -r requirements.txt


ENV CCL_SYCL_ALLTOALL_ARC_LL=1
ENV CCL_SYCL_CCL_BARRIER=1

WORKDIR /llm/HY-WorldPlay
