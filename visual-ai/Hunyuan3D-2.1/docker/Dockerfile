# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/deep-learning-essentials:2025.0.2-0-devel-ubuntu24.04 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics-testing

# Install dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
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
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Install Intel GPU runtime packages
    apt-get update -y && \
    apt-get install -y libze1 libze-dev libze-intel-gpu1 intel-opencl-icd libze-intel-gpu-raytracing && \
    apt-get install -y intel-oneapi-dpcpp-ct=2025.0.1-17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llm

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

# ======= Add oneCCL build =======
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"

RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu && \
    pip install bigdl-core==2.4.0b1 && \
    apt remove python3-blinker -y
    
COPY ./patches/hunyuan3d_for_multi_arc.patch /tmp/

RUN cd /llm && \
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git && \
    cd ./Hunyuan3D-2.1 && \
    git checkout 81b4d9f638cf72bd58efad88353cc649f97dee54 && \
    git apply /tmp/hunyuan3d_for_multi_arc.patch && \
    pip install -r requirements.txt && \
    cd hy3dpaint/custom_rasterizer && \
    pip install --upgrade setuptools>=64 && \
    export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH && \
    export CPATH=/usr/include:/usr/local/lib/python3.10/dist-packages/torch/include/:$CPATH && \
    python3 setup.py install && \
    cd ../.. && \
    cd hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh && \
    cd ../.. && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

WORKDIR /llm/Hunyuan3D-2.1
