# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.45.5.4

ARG https_proxy
ARG http_proxy
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"

COPY ./patches/yunchang_for_multi_arc.patch /tmp/
COPY ./patches/xdit_for_multi_arc.patch /tmp/
COPY ./patches/raylight_for_multi_arc.patch /tmp/
COPY ./patches/xinference_device_utils.patch /tmp/
COPY ./patches/comfyui_for_multi_arc.patch /tmp/
COPY ./patches/comfyui_voxcpm_for_xpu.patch /tmp/
COPY ./patches/comfyui_hunyuan3d_for_xpu.patch /tmp/
COPY ./patches/sglang_diffusion_for_multi_arc.patch /tmp/
COPY ./patches/comfyui_hy_motion1.patch /tmp/
COPY ./patches/comfyui_gguf_xpu.patch /tmp/



# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
# Install dependencies and Python 3.10
    apt-get update -y && \
    # apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
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
    pip install -e . && \
# Install ComfyUI
    cd /llm && \
    git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git checkout acd0e536533cdf038bbaa32730cd12a438cc3a60 && \
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
    git checkout c91ef47afb02e522bf08ff199ea2182642de0368 && \
    git apply /tmp/raylight_for_multi_arc.patch && \
    pip install ray==2.49.2 && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    cd comfyui-easy-use && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm && \
    cd comfyui-voxcpm && \
    git checkout 7875a8a36a531b7efef8cac96dc74001d3cb5895 && \
    git apply /tmp/comfyui_voxcpm_for_xpu.patch && \
    pip install -r requirements.txt && \
    cd .. && \
    git clone https://github.com/billwuhao/ComfyUI_IndexTTS.git && \
    cd ComfyUI_IndexTTS && \
    pip install -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt && \
# Install ComfyUI Hunyuan3D Node
    cd /llm/ComfyUI/custom_nodes && \
    apt remove python3-blinker -y && \
    apt update -y && \
    apt install -y libopengl0 libglx0 libgl1 libxrender1 libxfixes3 libx11-dev libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 libxkbcommon0 libsm6 ffmpeg && \
    git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git && \
    cd ComfyUI-Hunyuan3d-2-1 && \
    git checkout 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
    git apply /tmp/comfyui_hunyuan3d_for_xpu.patch && \
    pip install bigdl-core==2.4.0b1 rembg realesrgan && \
    pip install -r requirements.txt && \
    cd hy3dpaint/custom_rasterizer && \
    python setup.py install && \
    cd ../DifferentiableRenderer && \
    python setup.py install && \
# Install HY-Motion1 Node
    cd /llm/ComfyUI/custom_nodes && \
    git clone https://github.com/jtydhr88/ComfyUI-HY-Motion1.git && \
    cd ComfyUI-HY-Motion1 && \
    git checkout 47b1d0d96eeac6d4b99509ede91490eaba93440a && \
    git apply /tmp/comfyui_hy_motion1.patch && \
    pip install -r requirements.txt && \
# Install ComfyUI-GGUF Node
    cd /llm/ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && \
    git checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a && \
    git apply /tmp/comfyui_gguf_xpu.patch && \
    pip install -r requirements.txt && \
# Install ComfyUI-CacheDiT Node
    cd /llm/ComfyUI/custom_nodes && \
    git clone https://github.com/xlite-dev/comfyui-cache-dit.git && \
    cd comfyui-cache-dit && \
    pip install -r requirements.txt && \
# Install Xinference
    pip install "xinference[transformers]==1.15.0" && \
    patch /usr/local/lib/python3.12/dist-packages/xinference/device_utils.py < /tmp/xinference_device_utils.patch && \
    pip install kokoro Jinja2==3.1.6 jieba ordered-set pypinyin cn2an pypinyin-dict && \
# Install SGlang Diffusion
    cd /llm && \
    git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout a83484275daf46efa6d61f89dc333059fd0a04a5 && \
    git apply /tmp/sglang_diffusion_for_multi_arc.patch && \
    pip install -e "python[diffusion]" && \
    pip install triton==3.5.0 && \
    pip install pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu --force-reinstall && \
    cd /llm && \
    git clone https://github.com/sgl-project/sgl-kernel-xpu.git && \
    cd sgl-kernel-xpu && \
    pip install -v . && \
# Install SGlang Diffusion for ComfyUI
    cd /llm/ComfyUI/custom_nodes && \
    cp -r /llm/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion . && \
    # Clean
    rm -rf /tmp/*

COPY ./workflows/* /llm/ComfyUI/user/default/workflows/
COPY ./example_inputs/* /llm/ComfyUI/input/
COPY ./tools/* /llm/tools/
COPY ./entrypoints/* /llm/entrypoints/

ENV CCL_SYCL_ALLTOALL_ARC_LL=1
ENV CCL_SYCL_CCL_BARRIER=1

WORKDIR /llm/entrypoints
