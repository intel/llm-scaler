# syntax=docker/dockerfile:1
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG BUILDER_IMAGE=intel/omix:0.1.0-devel-ubuntu24.04
ARG RUNTIME_IMAGE=intel/pytorch:xpu-2.11.0-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.11.0+xpu
ARG IMAGE_TAG

FROM ${BUILDER_IMAGE} AS builder

ARG https_proxy
ARG http_proxy
ARG no_proxy
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG IMAGE_TAG

RUN test -n "${IMAGE_TAG}"

ENV VIRTUAL_ENV=/opt/buildenv \
    PATH="/opt/buildenv/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/opt/buildenv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${LD_LIBRARY_PATH}"

SHELL ["bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl ffmpeg git patch ca-certificates \
        python3-dev python3-venv python3-pip \
        build-essential ninja-build cmake pkg-config \
        libsndfile1 libsm6 libxext6 libgl1 numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0

RUN python3 -m venv ${VIRTUAL_ENV} && \
    pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install pytest

COPY ./omni_xpu_kernel /tmp/omni_xpu_kernel

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

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git fetch --depth 1 origin bb131be9e83d2f773c90f1d6f1e4b248a498c8c5 && \
    git checkout bb131be9e83d2f773c90f1d6f1e4b248a498c8c5 && \
    pip install -r requirements.txt

COPY ./ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
RUN cd /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU && \
    git init && git add -A && \
    git -c user.name="llm-scaler" -c user.email="noreply@intel.com" \
        commit --allow-empty -m "ComfyUI-OmniXPU ${IMAGE_TAG}"

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
    git fetch --depth 1 origin 55b6497e5c5e4337d2de12943fd97a499bd9c2e5 && \
    git checkout 55b6497e5c5e4337d2de12943fd97a499bd9c2e5 && \
    pip install -r requirements.txt && \
    pip install --no-deps kernels==0.14.0

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

COPY ./patches/xinference_device_utils.patch /tmp/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "xinference[transformers]==1.15.0" \
        kokoro Jinja2==3.1.6 jieba ordered-set pypinyin cn2an pypinyin-dict && \
    patch ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/xinference/device_utils.py \
        < /tmp/xinference_device_utils.patch && \
    rm -f /tmp/xinference_device_utils.patch

COPY ./patches/sglang_diffusion_for_multi_arc.patch /tmp/

ARG DPCPP_SYCL_TARGET=bmg
ARG SGL_KERNEL_XPU_COMMIT=9aef6481524906d11956bf4b4ab8e243c5128b4e

RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/intel/oneapi/setvars.sh --force && \
    export TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET} && \
    export OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET} && \
    mkdir -p /wheels && \
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
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install triton==3.5.0 && \
    pip install pytorch-triton-xpu==3.5.0 \
        --index-url https://download.pytorch.org/whl/xpu --force-reinstall && \
    cd /llm && \
    git clone --depth 1 https://github.com/sgl-project/sgl-kernel-xpu.git && \
    cd sgl-kernel-xpu && \
    git fetch --depth 1 origin ${SGL_KERNEL_XPU_COMMIT} && \
    git checkout ${SGL_KERNEL_XPU_COMMIT} && \
    CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8 pip wheel -v . \
        --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET} \
        --wheel-dir /wheels --no-deps && \
    pip install --no-deps /wheels/sgl_kernel-*.whl && \
    cp -r /llm/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion \
        /llm/ComfyUI/custom_nodes/ && \
    rm -f /tmp/sglang_diffusion_for_multi_arc.patch

RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/intel/oneapi/setvars.sh --force && \
    export TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET} && \
    export CUTLASS_SYCL_ROOT=/llm/sgl-kernel-xpu/build/_deps/repo-cutlass-sycl-src && \
    export OMNI_XPU_REQUIRE_CUTE=1 && \
    export OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET} && \
    cd /tmp/omni_xpu_kernel && \
    grep -Fqx "__version__ = \"${IMAGE_TAG}\"" omni_xpu_kernel/_version.py && \
    CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8 \
        pip wheel . --wheel-dir /wheels --no-deps --no-build-isolation && \
    pip install --no-deps /wheels/omni_xpu_kernel-*.whl && \
    rm -rf /tmp/omni_xpu_kernel

RUN ls -lh /wheels/

COPY ./workflows/* /llm/ComfyUI/user/default/workflows/
COPY ./example_inputs/* /llm/ComfyUI/input/
COPY ./tools/* /llm/tools/
COPY ./entrypoints/* /llm/entrypoints/

RUN pip freeze --exclude-editable --all | \
    sed -E \
        -e '/^(omni_xpu_kernel|sgl-kernel) @ file:/d' \
        -e '/^(torch|torchaudio|torchvision|triton|triton-xpu|pytorch-triton-xpu)==/d' \
        -e '/^(dpcpp-cpp-rt|impi-rt|intel-cmplr-lib-rt|intel-cmplr-lib-ur|intel-cmplr-lic-rt|intel-opencl-rt|intel-openmp|intel-pti|intel-sycl-rt|mkl|oneccl|oneccl-devel|onemkl-license|onemkl-sycl-blas|onemkl-sycl-dft|onemkl-sycl-lapack|onemkl-sycl-rng|onemkl-sycl-sparse|tbb|tcmlib|umf)==/d' \
        -e '/^(pip|setuptools|wheel)==/d' \
        > /runtime-requirements.txt && \
    rm -rf /llm/sgl-kernel-xpu

FROM ${RUNTIME_IMAGE} AS runtime

ARG https_proxy
ARG http_proxy
ARG no_proxy
ARG PYTHON_VERSION
ARG IMAGE_TAG
ARG DPCPP_SYCL_TARGET=bmg

LABEL org.opencontainers.image.version="${IMAGE_TAG}"

COPY --from=builder /llm/ /llm/
COPY --from=builder /runtime-requirements.txt /tmp/runtime-requirements.txt

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET} \
    OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET} \
    LD_LIBRARY_PATH="/usr/local/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:${LD_LIBRARY_PATH}" \
    CCL_SYCL_ALLTOALL_ARC_LL=1 \
    CCL_SYCL_CCL_BARRIER=1 \
    OMNIXPU_INTERPOLATE_FIX=0 \
    OMNI_IMAGE_VERSION="${IMAGE_TAG}"

SHELL ["bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,from=builder,source=/wheels,target=/tmp/wheels \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl ffmpeg git libsndfile1 libsm6 libxext6 libgl1 numactl wget \
        libopengl0 libglx0 libxrender1 libxfixes3 libxi6 libxxf86vm1 \
        libxcursor1 libxrandr2 libxinerama1 libxkbcommon0 && \
    pip install --no-deps -r /tmp/runtime-requirements.txt && \
    pip install --no-deps \
        /tmp/wheels/sgl_kernel-*.whl \
        /tmp/wheels/omni_xpu_kernel-*.whl && \
    pip install --no-deps -e /llm/long-context-attention && \
    pip install --no-deps -e /llm/xDiT && \
    pip install --no-deps -e /llm/nunchaku-torch && \
    pip install --no-deps -e /llm/sglang/python && \
    rm -f /tmp/runtime-requirements.txt && \
    rm -rf /var/lib/apt/lists/* /root/.cmake /var/tmp/* && \
    find /opt/venv -name '*.pyc' -delete && \
    { find /opt/venv -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; }

# omni_xpu_kernel links against the oneDNN 3.9 ABI used during build.
COPY --from=builder /opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3.9 /usr/local/lib/libdnnl.so.3.9

RUN ln -sf libdnnl.so.3.9 /usr/local/lib/libdnnl.so.3 && \
    ln -sf libdnnl.so.3.9 /usr/local/lib/libdnnl.so && \
    ldconfig

ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib"

# Copy only the oneDNN runtime library required by omni_xpu_kernel.
COPY --from=builder \
    /opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3.9 \
    /usr/local/lib/libdnnl.so.3.9

RUN ln -sf libdnnl.so.3.9 /usr/local/lib/libdnnl.so.3 && \
    ln -sf libdnnl.so.3.9 /usr/local/lib/libdnnl.so && \
    ldconfig

ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

WORKDIR /llm/entrypoints
