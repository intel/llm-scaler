# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM amr-registry.caas.intel.com/intelanalytics/llm-scaler-platform:25.38.4.1

ARG https_proxy
ARG http_proxy
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics-testing && \
# Install dependencies and Python 3.10
    apt-get update -y && \
    apt-get install -y software-properties-common libgl1 && \
    apt-get install -y libxrender1 libxfixes3 libx11-dev libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 libxkbcommon0 libsm6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu && \
    pip install oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    pip install bigdl-core-xe-all==2.6.0 --extra-index-url https://download.pytorch.org/whl/xpu && \
    apt remove python3-blinker -y && \
    wget https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    pip install bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    # Install xDit related dependencies
    mkdir /llm && \
    cd /llm && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    git clone https://github.com/analytics-zoo/long-context-attention.git -b xpu-main && \
    cd long-context-attention && \
    pip install -e . && \
    cd /llm && \
    git clone https://github.com/analytics-zoo/xDiT.git -b xpu-main && \
    cd xDiT && \
    pip install -e . && \
# Install ComfyUI
    cd /llm && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install -r requirements.txt && \
    cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite && \
    cd comfyui-videohelpersuite && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/xiangyuT/raylight.git -b xpu_main && \
    cd raylight && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    cd comfyui-easy-use && \
    pip install -r requirements.txt

COPY ./workflows/* /llm/ComfyUI/user/default/workflows/

WORKDIR /llm/ComfyUI
