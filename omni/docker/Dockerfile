# syntax=docker/dockerfile:1
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=intel/llm-scaler-platform:26.13.7.1
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG ONEAPI_VERSION=2025.2.0-517

# ======== Builder Stage ========
FROM ${BASE_IMAGE} AS builder

ARG https_proxy
ARG http_proxy
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG ONEAPI_VERSION

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# System packages
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor \
        | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -y && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl ffmpeg git libsndfile1 libsm6 libxext6 libgl1 \
        lsb-release numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0 && \
    apt remove python3-blinker -y && \
    apt-get install -y intel-oneapi-dpcpp-ct=${ONEAPI_VERSION} && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm -f /usr/lib/python3*/EXTERNALLY-MANAGED

# PyTorch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install pytest

# omni_xpu_kernel
COPY ./omni_xpu_kernel /tmp/omni_xpu_kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp/omni_xpu_kernel && \
    pip install . --no-build-isolation && \
    rm -rf /tmp/omni_xpu_kernel

# xDiT / long-context-attention
COPY ./patches/yunchang_for_multi_arc.patch /tmp/
COPY ./patches/xdit_for_multi_arc.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /llm && \
    cd /llm && \
    git clone --depth 1 https://github.com/feifeibear/long-context-attention.git && \
    cd long-context-attention && \
    git fetch --depth 1 origin fc5d55e61b78b3102fd824bea1791cf406cc2a4b && \
    git checkout fc5d55e61b78b3102fd824bea1791cf406cc2a4b && \
    git apply /tmp/yunchang_for_multi_arc.patch && \
    pip install -e . && \
    cd /llm && \
    git clone --depth 1 https://github.com/xdit-project/xDiT.git && \
    cd xDiT && \
    git fetch --depth 1 origin fb8fb0e437a8745b9629020759de31d1626a4a7b && \
    git checkout fb8fb0e437a8745b9629020759de31d1626a4a7b && \
    git apply /tmp/xdit_for_multi_arc.patch && \
    pip install -e . && \
    rm -f /tmp/yunchang_for_multi_arc.patch /tmp/xdit_for_multi_arc.patch

# ComfyUI core (upstream, no patches)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git fetch --depth 1 origin 64b8457f55cd7fb54ca7a956d9c73b505e903e0c && \
    git checkout 64b8457f55cd7fb54ca7a956d9c73b505e903e0c && \
    pip install -r requirements.txt

# ComfyUI-OmniXPU: Intel XPU acceleration via custom node (replaces core patch)
COPY ./ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
ARG IMAGE_TAG="0.1.0-dev"
RUN cd /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU && \
    git init && git add -A && \
    git -c user.name="llm-scaler" -c user.email="noreply@intel.com" \
        commit --allow-empty -m "ComfyUI-OmniXPU ${IMAGE_TAG}"

# ComfyUI custom nodes (active)
COPY ./patches/raylight_for_multi_arc.patch /tmp/
COPY ./patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite && \
    pip install -r comfyui-videohelpersuite/requirements.txt && \
    git clone --depth 1 https://github.com/komikndr/raylight.git && \
    cd raylight && \
    git fetch --depth 1 origin c91ef47afb02e522bf08ff199ea2182642de0368 && \
    git checkout c91ef47afb02e522bf08ff199ea2182642de0368 && \
    git apply /tmp/raylight_for_multi_arc.patch && \
    pip install ray==2.49.2 -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    pip install -r comfyui-easy-use/requirements.txt && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && \
    git fetch --depth 1 origin e8b689a513c3e6b63edc44066560ca5919c0576e && \
    git checkout e8b689a513c3e6b63edc44066560ca5919c0576e && \
    git apply /tmp/comfyui_controlnet_aux_depth_anything_v2_xpu.patch && \
    cd .. && \
    pip install -r comfyui_controlnet_aux/requirements.txt && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    git fetch --depth 1 origin a41e0d85b822ddc4bbd1256417aa2d58d39e94ce && \
    git checkout a41e0d85b822ddc4bbd1256417aa2d58d39e94ce && \
    pip install -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Jasonzzt/ComfyUI-CacheDiT.git && \
    pip install -r ComfyUI-CacheDiT/requirements.txt && \
    git clone --depth 1 https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git && \
    cd ComfyUI-GGUF-XPU && \
    pip install -r requirements.txt && \
    rm -f /tmp/raylight_for_multi_arc.patch /tmp/comfyui_controlnet_aux_depth_anything_v2_xpu.patch

# nunchaku-torch runtime + ComfyUI-nunchaku-XPU custom node (SVDQuant W4A4/W4A16 diffusion inference)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/xiangyuT/nunchaku-torch.git && \
    cd nunchaku-torch && \
    git fetch --depth 1 origin e09fae78f138fc0f5e86ad630be14433bd9713af && \
    git checkout e09fae78f138fc0f5e86ad630be14433bd9713af && \
    pip install -e . --no-deps && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git && \
    cd ComfyUI-nunchaku-XPU && \
    git fetch --depth 1 origin 1b78483a6af31297d5291f61965502079aed3407 && \
    git checkout 1b78483a6af31297d5291f61965502079aed3407 && \
    pip install -r requirements.txt && \
    pip install --no-deps kernels==0.14.0

# ComfyUI disabled custom nodes (optional: --build-arg INSTALL_DISABLED_NODES=true)
ARG INSTALL_DISABLED_NODES=true
COPY ./patches/comfyui_voxcpm_for_xpu.patch /tmp/
COPY ./patches/comfyui_indextts.patch /tmp/
COPY ./patches/comfyui_hunyuan3d_for_xpu.patch /tmp/
COPY ./patches/comfyui_hy_motion1.patch /tmp/
COPY ./patches/comfyui_seedvr2_xpu.patch /tmp/
COPY ./patches/comfyui_flashvsr_ultra_fast_xpu.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_DISABLED_NODES}" = "true" ]; then \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm.disabled && \
    cd comfyui-voxcpm.disabled && \
    git fetch --depth 1 origin 7875a8a36a531b7efef8cac96dc74001d3cb5895 && \
    git checkout 7875a8a36a531b7efef8cac96dc74001d3cb5895 && \
    git apply /tmp/comfyui_voxcpm_for_xpu.patch && \
    pip install -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/billwuhao/ComfyUI_IndexTTS.git ComfyUI_IndexTTS.disabled && \
    cd ComfyUI_IndexTTS.disabled && \
    git apply /tmp/comfyui_indextts.patch && \
    pip install -r requirements.txt torchcodec==0.9.0 && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git ComfyUI-Hunyuan3d-2-1.disabled && \
    cd ComfyUI-Hunyuan3d-2-1.disabled && \
    git fetch --depth 1 origin 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
    git checkout 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
    git apply /tmp/comfyui_hunyuan3d_for_xpu.patch && \
    pip install rembg realesrgan -r requirements.txt && \
    cd hy3dpaint/custom_rasterizer && python setup.py install && \
    cd ../DifferentiableRenderer && python setup.py install && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/jtydhr88/ComfyUI-HY-Motion1.git ComfyUI-HY-Motion1.disabled && \
    cd ComfyUI-HY-Motion1.disabled && \
    git fetch --depth 1 origin 47b1d0d96eeac6d4b99509ede91490eaba93440a && \
    git checkout 47b1d0d96eeac6d4b99509ede91490eaba93440a && \
    git apply /tmp/comfyui_hy_motion1.patch && \
    pip install -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git ComfyUI-SeedVR2_VideoUpscaler.disabled && \
    cd ComfyUI-SeedVR2_VideoUpscaler.disabled && \
    git fetch --depth 1 origin 4490bd1f482e026674543386bb2a4d176da245b9 && \
    git checkout 4490bd1f482e026674543386bb2a4d176da245b9 && \
    git apply /tmp/comfyui_seedvr2_xpu.patch && \
    pip install -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git ComfyUI-FlashVSR_Ultra_Fast.disabled && \
    cd ComfyUI-FlashVSR_Ultra_Fast.disabled && \
    git fetch --depth 1 origin 4820b3f02347bddcbbb9a5a85ab7638fe976366e && \
    git checkout 4820b3f02347bddcbbb9a5a85ab7638fe976366e && \
    git apply /tmp/comfyui_flashvsr_ultra_fast_xpu.patch ; \
    fi && \
    rm -f /tmp/*.patch

# Xinference
COPY ./patches/xinference_device_utils.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "xinference[transformers]==1.15.0" \
        kokoro Jinja2==3.1.6 jieba ordered-set pypinyin cn2an pypinyin-dict && \
    patch /usr/local/lib/python${PYTHON_VERSION}/dist-packages/xinference/device_utils.py \
        < /tmp/xinference_device_utils.patch && \
    rm -f /tmp/xinference_device_utils.patch

# SGLang Diffusion + sgl-kernel-xpu
COPY ./patches/sglang_diffusion_for_multi_arc.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git fetch --depth 1 origin 1ae3218d036c554e6c109f18233d1f75bf775c1d && \
    git checkout 1ae3218d036c554e6c109f18233d1f75bf775c1d && \
    git apply /tmp/sglang_diffusion_for_multi_arc.patch && \
    cp python/pyproject_xpu.toml python/pyproject.toml && \
    sed -i '/sgl-kernel/d' python/pyproject.toml && \
    sed -i 's/"transformers==[0-9.]*"/"transformers"/' python/pyproject.toml && \
    sed -E -i 's/(torch[a-z]*)([=<>!~]+[0-9.+a-z]+)/\1/g' python/pyproject.toml && \
    pip install -e "python[diffusion]" && \
    pip install \
        'transformers==5.0.0' \
        'huggingface_hub==1.3.5' \
        'diffusers==0.38.0' \
        'peft==0.19.1' && \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install triton==3.5.0 && \
    pip install pytorch-triton-xpu==3.5.0 \
        --index-url https://download.pytorch.org/whl/xpu --force-reinstall && \
    cd /llm && \
    git clone --depth 1 https://github.com/sgl-project/sgl-kernel-xpu.git && \
    cd sgl-kernel-xpu && \
    pip install -v . && \
    cp -r /llm/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion \
        /llm/ComfyUI/custom_nodes/ && \
    rm -f /tmp/sglang_diffusion_for_multi_arc.patch

# Application files
COPY ./workflows/* /llm/ComfyUI/user/default/workflows/
COPY ./example_inputs/* /llm/ComfyUI/input/
COPY ./tools/* /llm/tools/
COPY ./entrypoints/* /llm/entrypoints/

# ======== Runtime Stage ========
FROM ${BASE_IMAGE} AS runtime

ARG https_proxy
ARG http_proxy
ARG PYTHON_VERSION=3.12
ARG ONEAPI_VERSION=2025.2.0-517

# System packages
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor \
        | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -y && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl ffmpeg git libsndfile1 libsm6 libxext6 libgl1 \
        lsb-release numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0 && \
    apt remove python3-blinker -y && \
    apt-get install -y intel-oneapi-dpcpp-ct=${ONEAPI_VERSION} && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm -f /usr/lib/python3*/EXTERNALLY-MANAGED

# Copy built artifacts from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/ /usr/local/lib/python${PYTHON_VERSION}/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /usr/local/lib/*.so* /usr/local/lib/
COPY --from=builder /llm/ /llm/

# Fix for multi-XPU support: copy SYCL native binary files to a common library path
RUN cp /opt/intel/oneapi/compiler/2025.3/lib/libsycl-native-*.spv /usr/local/lib/

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:${LD_LIBRARY_PATH}"
ENV CCL_SYCL_ALLTOALL_ARC_LL=1
ENV CCL_SYCL_CCL_BARRIER=1
ENV OMNIXPU_INTERPOLATE_FIX=0
ENV OMNIXPU_ATTENTION=0

WORKDIR /llm/entrypoints
