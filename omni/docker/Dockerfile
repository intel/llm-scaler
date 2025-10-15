# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.38.4.2

ARG https_proxy
ARG http_proxy
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"

COPY ./patches/yunchang_for_multi_arc.patch /tmp/
COPY ./patches/xdit_for_multi_arc.patch /tmp/
COPY ./patches/raylight_for_multi_arc.patch /tmp/
COPY ./patches/xinference_device_utils.patch /tmp/
COPY ./patches/comfyui_for_multi_arc.patch /tmp/
COPY ./patches/comfyui_voxcpm_for_xpu.patch /tmp/


# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
# Install dependencies and Python 3.10
    apt-get update -y && \
    apt-get install -y software-properties-common libgl1 && \
    apt-get install -y libxrender1 libxfixes3 libx11-dev libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 libxkbcommon0 libsm6 ffmpeg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu && \
    pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ && \
    pip install bigdl-core-xe-all==2.7.0b20250625 && \
    apt remove python3-blinker -y && \
    wget https://download.blender.org/pypi/bpy/bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    pip install bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    rm bpy-4.0.0-cp310-cp310-manylinux_2_28_x86_64.whl && \
    # Install xDit related dependencies
    mkdir /llm && \
    cd /llm && \
    ln -s /usr/bin/python3 /usr/bin/python && \
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
    pip install -e . && \
# Install ComfyUI
    cd /llm && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git checkout 51696e3fdcdfad657cb15854345fbcbbe70eef8d && \
    git apply /tmp/comfyui_for_multi_arc.patch && \
    pip install -r requirements.txt && \
    cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite && \
    cd comfyui-videohelpersuite && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/komikndr/raylight.git && \
    cd raylight && \
    git checkout ff8e90ba1f2c2d23e3ac23746910ddfb523fc8f1 && \
    git apply /tmp/raylight_for_multi_arc.patch && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    cd comfyui-easy-use && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && \
    apt install libcairo2-dev pkg-config python3-dev -y && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm && \
    cd comfyui-voxcpm && \
    git checkout 044dd93c0effc9090fb279117de5db4cd90242a0 && \
    git apply /tmp/comfyui_voxcpm_for_xpu.patch && \
    pip install -r requirements.txt && \
# Install Xinference
    pip install "xinference[transformers]" && \
    patch /usr/local/lib/python3.10/dist-packages/xinference/device_utils.py < /tmp/xinference_device_utils.patch && \
    pip install kokoro Jinja2==3.1.6 jieba ordered-set pypinyin cn2an pypinyin-dict && \
    # Clean
    rm -rf /tmp/*

COPY ./workflows/* /llm/ComfyUI/user/default/workflows/
COPY ./example_inputs/* /llm/ComfyUI/input/

WORKDIR /llm/ComfyUI
