# syntax=docker/dockerfile:1
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ComfyUI-focused Omni development image.
#
# The default image intentionally excludes Xinference, SGLang/SGLDiffusion,
# and the disabled media/3D node bundle. Raylight's xDiT-based multi-XPU
# ComfyUI integration is not part of either flavor; this image targets one XPU.

ARG BASE_IMAGE=intel/omix:0.1.0-devel-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.11.0+xpu
ARG TORCHVISION_VERSION=0.26.0+xpu
ARG TORCHAUDIO_VERSION=2.11.0+xpu
ARG ONEDNN_VERSION=2025.3.0
ARG XPU_TARGET=bmg
ARG MAX_JOBS=8
ARG IMAGE_TAG

ARG SYCL_TLA_REPOSITORY=https://github.com/intel/sycl-tla.git
ARG SYCL_TLA_COMMIT=2fc09973bfdf15755090fcb0e3b6ad236408a992
ARG COMFY_KITCHEN_REPOSITORY=https://github.com/xiangyuT/comfy-kitchen-xpu.git
ARG COMFY_KITCHEN_COMMIT=9b4776cc7507c47f16a53a5b8a991c1ba2bc354a
ARG COMFY_KITCHEN_VERSION=0.2.18+xpu.1
ARG LLM_SCALER_SOURCE_REVISION=unknown
ARG LLM_SCALER_SOURCE_DIRTY=unknown

# ======== Stable OS/toolchain base ========
# Keep device, image-version, Kitchen, and local-source arguments out of this
# stage. Changes to those inputs must not invalidate apt or Python setup.
FROM ${BASE_IMAGE} AS os-base

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ca-certificates curl ffmpeg git patch \
        python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
        build-essential ninja-build cmake pkg-config \
        libsndfile1 libsm6 libxext6 libgl1 \
        lsb-release numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0

# ======== Stable Python/Torch base ========
# This stage is shared by the ComfyUI dependency, Kitchen, and kernel branches.
# It changes only when the base Python/Torch/oneDNN environment changes.
FROM os-base AS python-base

ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG ONEDNN_VERSION

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/intel/oneapi/compiler/latest/bin:${PATH}" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/dnnl/latest/lib:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/mpi/latest/lib"

RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV} && \
    pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install \
        pytest \
        onednn==${ONEDNN_VERSION} \
        onednn-devel==${ONEDNN_VERSION}

RUN mkdir -p /llm /wheels /llm/patches

# ======== Pinned native headers ========
# Header acquisition is independent of both local native projects. A kernel
# source change reuses this stage; a Kitchen change never reaches it.
FROM python-base AS sycl-tla

ARG SYCL_TLA_REPOSITORY
ARG SYCL_TLA_COMMIT

# Fetch native headers directly. This replaces the old hidden dependency on the
# copy of sycl-tla populated as a side effect of building sgl-kernel-xpu.
RUN cd /llm && \
    git clone --filter=blob:none --no-checkout \
        "${SYCL_TLA_REPOSITORY}" sycl-tla && \
    git -C sycl-tla fetch --depth 1 origin "${SYCL_TLA_COMMIT}" && \
    git -C sycl-tla checkout --detach FETCH_HEAD

# ======== Stable ComfyUI dependency tree ========
FROM python-base AS comfyui-deps

# ComfyUI core and its Python dependencies are target-independent and stay
# ahead of local source COPY instructions for Docker cache reuse.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/comfyanonymous/ComfyUI.git && \
    git -C ComfyUI fetch --depth 1 origin \
        bb131be9e83d2f773c90f1d6f1e4b248a498c8c5 && \
    git -C ComfyUI checkout --detach FETCH_HEAD && \
    pip install -r /llm/ComfyUI/requirements.txt

# Pin every default custom node to the exact revision accepted in the previous
# full image. These stable remote-source layers stay ahead of all frequently
# edited local native sources.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager && \
    git -C comfyui-manager fetch --depth 1 origin \
        7955e638db7d4a4b8bf7a614e724a2013b83dfd7 && \
    git -C comfyui-manager checkout --detach FETCH_HEAD && \
    pip install -r comfyui-manager/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        comfyui-videohelpersuite && \
    git -C comfyui-videohelpersuite fetch --depth 1 origin \
        4ee72c065db22c9d96c2427954dc69e7b908444b && \
    git -C comfyui-videohelpersuite checkout --detach FETCH_HEAD && \
    pip install -r comfyui-videohelpersuite/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    git -C comfyui-easy-use fetch --depth 1 origin \
        54d080bf6a4f52da287e984f305243c10db097f5 && \
    git -C comfyui-easy-use checkout --detach FETCH_HEAD && \
    pip install -r comfyui-easy-use/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/kijai/ComfyUI-KJNodes.git && \
    git -C ComfyUI-KJNodes fetch --depth 1 origin \
        a41e0d85b822ddc4bbd1256417aa2d58d39e94ce && \
    git -C ComfyUI-KJNodes checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-KJNodes/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/Jasonzzt/ComfyUI-CacheDiT.git && \
    git -C ComfyUI-CacheDiT fetch --depth 1 origin \
        ab1a1448a2476a33493c9a9b04e1685971b9e73a && \
    git -C ComfyUI-CacheDiT checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-CacheDiT/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git && \
    git -C ComfyUI-GGUF-XPU fetch --depth 1 origin \
        4b8a633e8434036820a2bb9d18cb77ae691e788c && \
    git -C ComfyUI-GGUF-XPU checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-GGUF-XPU/requirements.txt

# Nunchaku is retained because it is a ComfyUI quantized-model backend. Its
# Python runtime is independent of the removed SGLang service.
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/xiangyuT/nunchaku-torch.git && \
    git -C nunchaku-torch fetch --depth 1 origin \
        e09fae78f138fc0f5e86ad630be14433bd9713af && \
    git -C nunchaku-torch checkout --detach FETCH_HEAD && \
    pip install -e nunchaku-torch --no-deps && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git && \
    git -C ComfyUI-nunchaku-XPU fetch --depth 1 origin \
        55b6497e5c5e4337d2de12943fd97a499bd9c2e5 && \
    git -C ComfyUI-nunchaku-XPU checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-nunchaku-XPU/requirements.txt

# Keep the only patched third-party node last. Editing this local patch no
# longer invalidates any of the unpatched third-party node layers above.
COPY ./patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch /llm/patches/
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git -C comfyui_controlnet_aux fetch --depth 1 origin \
        e8b689a513c3e6b63edc44066560ca5919c0576e && \
    git -C comfyui_controlnet_aux checkout --detach FETCH_HEAD && \
    git -C comfyui_controlnet_aux apply \
        /llm/patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch && \
    pip install -r comfyui_controlnet_aux/requirements.txt

# ======== Kitchen wheel branch ========
# Kitchen and Omni kernel deliberately branch from separate cached parents.
# Updating either source/commit builds only that wheel, then reruns the cheap
# final installation/verification layer.
FROM python-base AS kitchen-wheel

ARG COMFY_KITCHEN_REPOSITORY
ARG COMFY_KITCHEN_COMMIT

RUN cd /llm && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_KITCHEN_REPOSITORY}" comfy-kitchen-xpu && \
    git -C comfy-kitchen-xpu fetch --depth 1 origin \
        "${COMFY_KITCHEN_COMMIT}" && \
    git -C comfy-kitchen-xpu checkout --detach FETCH_HEAD

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/comfy-kitchen-xpu && \
    pip wheel . --wheel-dir /wheels --no-deps --no-build-isolation

# ======== Omni kernel wheel branch ========
FROM sycl-tla AS kernel-wheel

ARG IMAGE_TAG
ARG XPU_TARGET
ARG MAX_JOBS

ENV TORCH_XPU_ARCH_LIST=${XPU_TARGET} \
    OMNI_XPU_DEVICE=${XPU_TARGET} \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
    CUTLASS_SYCL_ROOT=/llm/sycl-tla \
    OMNI_XPU_REQUIRE_CUTE=1

# COPY --link keeps the source snapshot independent of changes to earlier base
# layers; only this native branch consumes the frequently edited kernel tree.
COPY --link ./omni_xpu_kernel /llm/omni_xpu_kernel
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /llm/omni_xpu_kernel && \
    pip wheel -v . --wheel-dir /wheels --no-deps --no-build-isolation && \
    pip install --no-deps --force-reinstall \
        /wheels/omni_xpu_kernel-*.whl

# ======== Final ComfyUI development assembly ========
FROM comfyui-deps AS builder-comfyui

ARG IMAGE_TAG
ARG XPU_TARGET
ARG MAX_JOBS
ARG COMFY_KITCHEN_VERSION

ENV TORCH_XPU_ARCH_LIST=${XPU_TARGET} \
    OMNI_XPU_DEVICE=${XPU_TARGET} \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
    CUTLASS_SYCL_ROOT=/llm/sycl-tla \
    OMNI_XPU_REQUIRE_CUTE=1

COPY --from=sycl-tla /llm/sycl-tla /llm/sycl-tla
COPY --from=kitchen-wheel /llm/comfy-kitchen-xpu /llm/comfy-kitchen-xpu
COPY --from=kernel-wheel /llm/omni_xpu_kernel /llm/omni_xpu_kernel
COPY --from=kitchen-wheel /wheels/ /wheels/
COPY --from=kernel-wheel /wheels/ /wheels/

# All third-party requirements are already installed. Force the two locally
# iterated native packages into the final environment without dependency
# resolution so neither can silently replace the accepted Torch/Kitchen stack.
# Artifact and device acceptance belongs in tools/validate_comfyui_image.py,
# where the check can run with the real /dev/dri device rather than Docker's
# device-less build sandbox.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --force-reinstall \
        /wheels/omni_xpu_kernel-*.whl \
        /wheels/comfy_kitchen-${COMFY_KITCHEN_VERSION}-*.whl

# The local custom node is intentionally a thin ComfyUI integration layer. It
# is copied after Kitchen/kernel installation so adapter-only edits stay cheap.
COPY ./ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
RUN cd /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU && \
    git init && git add -A && \
    git -c user.name="llm-scaler" -c user.email="noreply@intel.com" \
        commit --allow-empty -m "ComfyUI-OmniXPU ${IMAGE_TAG}"

# Retain project sources needed for in-container development. The focused image
# intentionally relies on ComfyUI's Template Browser instead of bundled
# workflow/input snapshots.
COPY ./patches /llm/patches
COPY ./tools /llm/tools
COPY ./entrypoints/start_comfyui.sh /llm/entrypoints/start_comfyui.sh

# ======== ComfyUI Runtime ========
# The runtime is intentionally the completed development builder rather than a
# fresh os-base plus copies of /opt/venv and /llm. Both stages use the same
# OMIX devel base, so copying the complete environment only creates very large
# replacement layers after each kernel/Kitchen edit and slows image export.
FROM builder-comfyui AS runtime-comfyui

ARG XPU_TARGET
ARG IMAGE_TAG
ARG SYCL_TLA_COMMIT
ARG COMFY_KITCHEN_COMMIT
ARG COMFY_KITCHEN_VERSION
ARG LLM_SCALER_SOURCE_REVISION
ARG LLM_SCALER_SOURCE_DIRTY

RUN cp /opt/intel/oneapi/compiler/latest/lib/libsycl-native-*.spv \
    /usr/local/lib/

LABEL org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.revision="${LLM_SCALER_SOURCE_REVISION}" \
      com.intel.omni.image-flavor="comfyui" \
      com.intel.omni.xpu-target="${XPU_TARGET}" \
      com.intel.omni.llm-scaler.source-dirty="${LLM_SCALER_SOURCE_DIRTY}" \
      com.intel.omni.sycl-tla.revision="${SYCL_TLA_COMMIT}" \
      com.intel.omni.comfy-kitchen.version="${COMFY_KITCHEN_VERSION}" \
      com.intel.omni.comfy-kitchen.revision="${COMFY_KITCHEN_COMMIT}"

ENV OMNIXPU_INTERPOLATE_FIX=0 \
    OMNI_IMAGE_VERSION="${IMAGE_TAG}" \
    OMNI_IMAGE_FLAVOR="comfyui" \
    OMNI_IMAGE_XPU_TARGET="${XPU_TARGET}" \
    OMNI_LLM_SCALER_SOURCE_REVISION="${LLM_SCALER_SOURCE_REVISION}" \
    OMNI_LLM_SCALER_SOURCE_DIRTY="${LLM_SCALER_SOURCE_DIRTY}" \
    OMNI_COMFY_KITCHEN_VERSION="${COMFY_KITCHEN_VERSION}"

WORKDIR /llm/entrypoints
CMD ["/bin/bash"]
