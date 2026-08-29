# syntax=docker/dockerfile:1
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ComfyUI-focused Omni development image.
#
# The default image intentionally excludes Xinference, SGLang/SGLDiffusion,
# and the disabled media/3D node bundle. Raylight's xDiT-based multi-XPU
# ComfyUI integration is not part of this image; this image targets one XPU.

ARG BASE_IMAGE=intel/omix:0.3.0-devel-ubuntu24.04@sha256:53e2c4503beeea4aff906dea180933be672449bcf04eb38df3d89622a1cd0967
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.13.0+xpu
ARG TORCHVISION_VERSION=0.28.0+xpu
ARG TORCHAUDIO_VERSION=2.11.0+xpu
ARG ONEDNN_VERSION=2026.0.0
ARG ONEDNN_SOURCE_REPOSITORY=https://github.com/uxlfoundation/oneDNN.git
ARG ONEDNN_SOURCE_COMMIT=03c022d3ffdcee958cfacbe720048e725fdf644c
ARG ONEDNN_PATCH_SHA256=0a7afff4134f115b4bc53f46301ca3d62b1c11dc02e32c64635c469769fcdaeb
ARG XPU_TARGET=bmg
ARG MAX_JOBS=8
ARG IMAGE_TAG

ARG SYCL_TLA_REPOSITORY=https://github.com/intel/sycl-tla.git
ARG SYCL_TLA_COMMIT=2fc09973bfdf15755090fcb0e3b6ad236408a992
ARG COMFYUI_REPOSITORY=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFYUI_COMMIT=7a131a3afadc8200120f67f9236311a2c48b7445
ARG COMFYUI_VERSION=0.33.4
ARG COMFYUI_FRONTEND_VERSION=1.49.6
ARG COMFYUI_WORKFLOW_TEMPLATES_VERSION=0.11.46
ARG COMFYUI_MANAGER_VERSION=4.2.2
ARG COMFY_KITCHEN_REPOSITORY=https://github.com/xiangyuT/comfy-kitchen-xpu.git
ARG COMFY_KITCHEN_COMMIT=9eccb7fa42edf14bc4a4c41aafd645ff1f1dcb75
ARG COMFY_KITCHEN_VERSION=0.2.31
ARG COMFY_AIMDO_REPOSITORY=https://github.com/xiangyuT/comfy-aimdo-xpu.git
ARG COMFY_AIMDO_COMMIT=063d66e5345fea58d1a4e8aa6f160ccc0c593f16
ARG COMFY_AIMDO_VERSION=0.4.13
ARG COMFY_GGUF_REPOSITORY=https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git
ARG COMFY_GGUF_COMMIT=39671fe73117ba97de7011e7e06e32599dcda06d
ARG COMFY_NUNCHAKU_REPOSITORY=https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git
ARG COMFY_NUNCHAKU_COMMIT=cc0f6236b6c329178ad4ef58452a874e774c7b8e
ARG COMFY_NUNCHAKU_VERSION=1.2.1+xpu.3
ARG COMFY_SOL_ATTN_REPOSITORY=https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git
ARG COMFY_SOL_ATTN_COMMIT=5f1c4aac3ca32a00b0b4c15ddbb7cb53fa43344d
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
ARG ONEDNN_SOURCE_REPOSITORY
ARG ONEDNN_SOURCE_COMMIT
ARG ONEDNN_PATCH_SHA256
ARG MAX_JOBS

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
        onednn-devel==${ONEDNN_VERSION} && \
    pip check

COPY ./patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch \
    /llm/patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch

# oneDNN 3.11 disabled its optimized BF16 INT4 dequantization path in
# 0d99c32b03614d6943b993974f91d419e3c3a0f6. That changes the observable
# BF16 materialization contract used by Kitchen SVDQuant. Build the exact
# 3.11.2 release with the reviewed two-site restoration instead of downgrading
# the runtime or weakening exact correctness. Keep the Python distribution
# version pinned above and record source, patch, and installed-DSO identities.
RUN set -euo pipefail && \
    echo "${ONEDNN_PATCH_SHA256}  /llm/patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch" \
        | sha256sum --check - && \
    git clone --filter=blob:none --no-checkout \
        "${ONEDNN_SOURCE_REPOSITORY}" /tmp/onednn-source && \
    git -C /tmp/onednn-source fetch --depth 1 origin \
        "${ONEDNN_SOURCE_COMMIT}" && \
    git -C /tmp/onednn-source checkout --detach FETCH_HEAD && \
    test "$(git -C /tmp/onednn-source rev-parse HEAD)" \
        = "${ONEDNN_SOURCE_COMMIT}" && \
    git -C /tmp/onednn-source apply --check \
        /llm/patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch && \
    git -C /tmp/onednn-source apply \
        /llm/patches/onednn-v3.11.2-enable-bf16-int4-dequantization.patch && \
    git -C /tmp/onednn-source diff --check && \
    set +u && \
    source /opt/intel/oneapi/setvars.sh --force && \
    set -u && \
    cmake -S /tmp/onednn-source -B /tmp/onednn-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" \
        -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx \
        -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
        -DDNNL_BUILD_DOC=OFF \
        -DDNNL_BUILD_EXAMPLES=OFF \
        -DDNNL_BUILD_TESTS=OFF \
        -DDNNL_CPU_RUNTIME=NONE \
        -DDNNL_GPU_RUNTIME=SYCL \
        -DDNNL_GPU_VENDOR=INTEL \
        -DONEDNN_BUILD_GRAPH=OFF && \
    cmake --build /tmp/onednn-build --parallel "${MAX_JOBS}" && \
    cmake --install /tmp/onednn-build && \
    test "$(readlink -f "${VIRTUAL_ENV}/lib/libdnnl.so.3")" \
        = "${VIRTUAL_ENV}/lib/libdnnl.so.3.11" && \
    mkdir -p /llm/manifests && \
    library_sha256="$(sha256sum "${VIRTUAL_ENV}/lib/libdnnl.so.3.11" \
        | awk '{print $1}')" && \
    printf '%s\n' \
        "schema_version=1" \
        "package_version=${ONEDNN_VERSION}" \
        "source_repository=${ONEDNN_SOURCE_REPOSITORY}" \
        "source_revision=${ONEDNN_SOURCE_COMMIT}" \
        "patch_sha256=${ONEDNN_PATCH_SHA256}" \
        "library_path=${VIRTUAL_ENV}/lib/libdnnl.so.3.11" \
        "library_sha256=${library_sha256}" \
        > /llm/manifests/onednn-runtime.env && \
    rm -rf /tmp/onednn-source /tmp/onednn-build && \
    pip check

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
ARG COMFYUI_REPOSITORY
ARG COMFYUI_COMMIT
ARG COMFYUI_FRONTEND_VERSION
ARG COMFYUI_WORKFLOW_TEMPLATES_VERSION
ARG COMFYUI_MANAGER_VERSION
ARG COMFY_KITCHEN_VERSION
ARG COMFY_AIMDO_VERSION

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --filter=blob:none --no-checkout \
        "${COMFYUI_REPOSITORY}" ComfyUI && \
    git -C ComfyUI fetch --depth 1 origin \
        "${COMFYUI_COMMIT}" && \
    git -C ComfyUI checkout --detach FETCH_HEAD && \
    grep -Ev \
        '^(torch|torchvision|torchaudio)([[:space:]<>=!~].*)?$|^(comfyui-frontend-package|comfyui-workflow-templates|comfy-kitchen|comfy-aimdo)==' \
        /llm/ComfyUI/requirements.txt \
        > /tmp/comfyui-requirements.txt && \
    grep -qx \
        "comfyui_manager==${COMFYUI_MANAGER_VERSION}" \
        /llm/ComfyUI/manager_requirements.txt && \
    pip install \
        -r /tmp/comfyui-requirements.txt \
        "comfyui-frontend-package==${COMFYUI_FRONTEND_VERSION}" \
        "comfyui-workflow-templates==${COMFYUI_WORKFLOW_TEMPLATES_VERSION}" \
        "comfyui-manager==${COMFYUI_MANAGER_VERSION}" \
        "comfy-kitchen==${COMFY_KITCHEN_VERSION}" \
        "comfy-aimdo==${COMFY_AIMDO_VERSION}" && \
    pip check

# ComfyUI integrates Node Manager through the versioned Python package
# above. Do not also clone the legacy ComfyUI-Manager custom node: loading both
# creates two manager boundaries and makes missing-node behavior dependent on
# startup flags rather than the pinned package contract. Kitchen and AIMDO are
# removed from upstream requirements only so their accepted official versions
# can be installed explicitly above. The XPU implementations are installed as
# separately named providers and never overwrite these official packages.

# Pin every remaining default custom node to an exact accepted revision. These
# stable remote-source layers stay ahead of frequently edited native sources.

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        comfyui-videohelpersuite && \
    git -C comfyui-videohelpersuite fetch --depth 1 origin \
        4ee72c065db22c9d96c2427954dc69e7b908444b && \
    git -C comfyui-videohelpersuite checkout --detach FETCH_HEAD && \
    pip install -r comfyui-videohelpersuite/requirements.txt

# Easy-Use v1.3.6 (release tag resolved to an immutable commit).
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    git -C comfyui-easy-use fetch --depth 1 origin \
        b5e31ef12ad9d0b187b545c2707735cc7d581c52 && \
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
        1d92bbd86ec59aa6223fe2368849b7413a1acb93 && \
    git -C ComfyUI-CacheDiT checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-CacheDiT/requirements.txt

# Sol-Attn owns only ComfyUI dispatch; its Intel XPU implementation is built
# into the pinned omni_xpu_kernel wheel below. Keep the custom node at an exact
# revision and do not install Triton or build a second native sidecar.
ARG COMFY_SOL_ATTN_REPOSITORY
ARG COMFY_SOL_ATTN_COMMIT

RUN cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_SOL_ATTN_REPOSITORY}" ComfyUI-SolAttn && \
    git -C ComfyUI-SolAttn fetch --depth 1 origin \
        "${COMFY_SOL_ATTN_COMMIT}" && \
    git -C ComfyUI-SolAttn checkout --detach FETCH_HEAD

ARG COMFY_GGUF_REPOSITORY
ARG COMFY_GGUF_COMMIT

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_GGUF_REPOSITORY}" ComfyUI-GGUF-XPU && \
    git -C ComfyUI-GGUF-XPU fetch --depth 1 origin \
        "${COMFY_GGUF_COMMIT}" && \
    git -C ComfyUI-GGUF-XPU checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-GGUF-XPU/requirements.txt

# The XPU custom node includes its nunchaku_torch runtime. Keep the source in
# custom_nodes for ComfyUI discovery and install the same checkout as one
# distribution so the bundled runtime is importable without a second repo.
ARG COMFY_NUNCHAKU_REPOSITORY
ARG COMFY_NUNCHAKU_COMMIT

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_NUNCHAKU_REPOSITORY}" ComfyUI-nunchaku-XPU && \
    git -C ComfyUI-nunchaku-XPU fetch --depth 1 origin \
        "${COMFY_NUNCHAKU_COMMIT}" && \
    git -C ComfyUI-nunchaku-XPU checkout --detach FETCH_HEAD && \
    pip install -r ComfyUI-nunchaku-XPU/requirements.txt && \
    pip install --no-deps --no-build-isolation ./ComfyUI-nunchaku-XPU

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
ARG COMFY_KITCHEN_VERSION
ARG TORCH_VERSION
ARG XPU_TARGET

RUN cd /llm && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_KITCHEN_REPOSITORY}" comfy-kitchen-xpu && \
    git -C comfy-kitchen-xpu fetch --depth 1 origin \
        "${COMFY_KITCHEN_COMMIT}" && \
    git -C comfy-kitchen-xpu checkout --detach FETCH_HEAD

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/comfy-kitchen-xpu && \
    mkdir -p /wheels/kitchen-source /wheels/kitchen-provider && \
    pip wheel . --wheel-dir /wheels/kitchen-source \
        --no-deps --no-build-isolation && \
    source_wheels=(/wheels/kitchen-source/comfy_kitchen-${COMFY_KITCHEN_VERSION}-*.whl) && \
    test "${#source_wheels[@]}" -eq 1 && \
    python packaging/xpu_runtime_provider/build_wheel.py \
        --source-wheel "${source_wheels[0]}" \
        --output-dir /wheels/kitchen-provider \
        --source-revision "${COMFY_KITCHEN_COMMIT}" \
        --torch-version "${TORCH_VERSION}" \
        --xpu-target "${XPU_TARGET}"

# ======== AIMDO XPU wheel branch ========
FROM python-base AS aimdo-wheel

ARG COMFY_AIMDO_REPOSITORY
ARG COMFY_AIMDO_COMMIT
ARG COMFY_AIMDO_VERSION
ARG TORCH_VERSION
ARG XPU_TARGET

RUN cd /llm && \
    git clone --filter=blob:none --no-checkout \
        "${COMFY_AIMDO_REPOSITORY}" comfy-aimdo-xpu && \
    git -C comfy-aimdo-xpu fetch --depth 1 origin \
        "${COMFY_AIMDO_COMMIT}" && \
    git -C comfy-aimdo-xpu checkout --detach FETCH_HEAD && \
    pip install setuptools-scm==10.2.1 && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /llm/comfy-aimdo-xpu && \
    test -f /opt/venv/include/unified-runtime/ur_api.h && \
    UR_INCLUDE_DIR=/opt/venv/include/unified-runtime \
        ./scripts/build-linux-xpu.sh && \
    mkdir -p /wheels/aimdo-source /wheels/aimdo-provider && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${COMFY_AIMDO_VERSION}" \
        pip wheel . --wheel-dir /wheels/aimdo-source \
            --no-deps --no-build-isolation && \
    source_wheels=(/wheels/aimdo-source/comfy_aimdo-${COMFY_AIMDO_VERSION}-*.whl) && \
    test "${#source_wheels[@]}" -eq 1 && \
    python packaging/xpu_runtime_provider/build_wheel.py \
        --source-wheel "${source_wheels[0]}" \
        --output-dir /wheels/aimdo-provider \
        --source-revision "${COMFY_AIMDO_COMMIT}" \
        --torch-version "${TORCH_VERSION}" \
        --xpu-target "${XPU_TARGET}"

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
ARG COMFYUI_VERSION
ARG COMFY_KITCHEN_VERSION
ARG COMFY_AIMDO_VERSION

ENV TORCH_XPU_ARCH_LIST=${XPU_TARGET} \
    OMNI_XPU_DEVICE=${XPU_TARGET} \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
    CUTLASS_SYCL_ROOT=/llm/sycl-tla \
    OMNI_XPU_REQUIRE_CUTE=1

COPY --from=sycl-tla /llm/sycl-tla /llm/sycl-tla
COPY --from=kitchen-wheel /llm/comfy-kitchen-xpu /llm/comfy-kitchen-xpu
COPY --from=aimdo-wheel /llm/comfy-aimdo-xpu /llm/comfy-aimdo-xpu
COPY --from=kernel-wheel /llm/omni_xpu_kernel /llm/omni_xpu_kernel
COPY --from=kitchen-wheel /wheels/kitchen-provider/ /wheels/providers/
COPY --from=aimdo-wheel /wheels/aimdo-provider/ /wheels/providers/
COPY --from=kernel-wheel /wheels/ /wheels/

# All third-party requirements, including official Kitchen and AIMDO, are
# already installed. Install the kernel and the two privately named XPU runtime
# providers without dependency resolution. The providers own no canonical
# comfy_kitchen or comfy_aimdo files, so later official upgrades are disjoint.
# Artifact and device acceptance belongs in tools/validate_comfyui_image.py,
# where the check can run with the real /dev/dri device rather than Docker's
# device-less build sandbox.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --force-reinstall \
        /wheels/omni_xpu_kernel-*.whl \
        /wheels/providers/comfy_kitchen_xpu_runtime-${COMFY_KITCHEN_VERSION}-*.whl \
        /wheels/providers/comfy_aimdo_xpu_runtime-${COMFY_AIMDO_VERSION}-*.whl && \
    pip check

# The local custom node is intentionally a thin ComfyUI integration layer. It
# is copied after Kitchen/kernel installation so adapter-only edits stay cheap.
COPY ./ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
RUN cd /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU && \
    git init && git add -A && \
    git -c user.name="llm-scaler" -c user.email="noreply@intel.com" \
        commit --allow-empty -m "ComfyUI-OmniXPU ${IMAGE_TAG}"

# Preserve the complete installed dependency snapshot used by image
# validation. This records transitive ComfyUI and custom-node packages in
# addition to the explicitly versioned frontend/templates/manager boundary.
RUN mkdir -p /llm/manifests && \
    sha256sum /wheels/providers/*.whl \
        > /llm/manifests/xpu-runtime-providers.sha256 && \
    pip freeze --all | LC_ALL=C sort \
        > /llm/manifests/comfyui-python-freeze.txt && \
    pip list --format=freeze | grep -Ei \
        '^(torch|torchvision|torchaudio|omni[-_]xpu[-_]kernel|comfy-kitchen-xpu-runtime|comfy-aimdo-xpu-runtime)==' \
        | LC_ALL=C sort \
        > /llm/manifests/omni-runtime-constraints.txt && \
    test "$(wc -l < /llm/manifests/omni-runtime-constraints.txt)" -eq 6

# Retain project sources needed for in-container development. The focused image
# intentionally relies on ComfyUI's Template Browser instead of bundled
# workflow/input snapshots.
COPY ./patches /llm/patches
COPY ./tools /llm/tools
RUN mkdir -p /models/host /data/input /data/output /data/user
COPY ./configs/comfyui_host_models.yaml /llm/configs/comfyui_host_models.yaml
COPY ./entrypoints/start_comfyui.sh /llm/entrypoints/start_comfyui.sh
COPY --chmod=755 ./entrypoints/omix_torch_runtime.sh /llm/entrypoints/omix_torch_runtime.sh

# ======== ComfyUI Runtime ========
# The runtime is intentionally the completed development builder rather than a
# fresh os-base plus copies of /opt/venv and /llm. Both stages use the same
# OMIX devel base, so copying the complete environment only creates very large
# replacement layers after each kernel/Kitchen edit and slows image export.
FROM builder-comfyui AS runtime-comfyui

ARG BASE_IMAGE
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG ONEDNN_VERSION
ARG ONEDNN_SOURCE_REPOSITORY
ARG ONEDNN_SOURCE_COMMIT
ARG ONEDNN_PATCH_SHA256
ARG XPU_TARGET
ARG IMAGE_TAG
ARG SYCL_TLA_COMMIT
ARG COMFYUI_COMMIT
ARG COMFYUI_VERSION
ARG COMFYUI_FRONTEND_VERSION
ARG COMFYUI_WORKFLOW_TEMPLATES_VERSION
ARG COMFYUI_MANAGER_VERSION
ARG COMFY_KITCHEN_REPOSITORY
ARG COMFY_KITCHEN_COMMIT
ARG COMFY_KITCHEN_VERSION
ARG COMFY_AIMDO_REPOSITORY
ARG COMFY_AIMDO_COMMIT
ARG COMFY_AIMDO_VERSION
ARG COMFY_GGUF_COMMIT
ARG COMFY_NUNCHAKU_COMMIT
ARG COMFY_NUNCHAKU_VERSION
ARG COMFY_SOL_ATTN_REPOSITORY
ARG COMFY_SOL_ATTN_COMMIT
ARG LLM_SCALER_SOURCE_REVISION
ARG LLM_SCALER_SOURCE_DIRTY

# The focused image is single-XPU. OMIX 0.3 / oneAPI 2026.1 no longer ships
# the legacy multi-XPU libsycl-native SPIR-V blobs copied by older images.
RUN mkdir -p /llm/ComfyUI/user

LABEL org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.revision="${LLM_SCALER_SOURCE_REVISION}" \
      com.intel.omni.image-flavor="comfyui" \
      com.intel.omni.base-image="${BASE_IMAGE}" \
      com.intel.omni.torch.version="${TORCH_VERSION}" \
      com.intel.omni.torchvision.version="${TORCHVISION_VERSION}" \
      com.intel.omni.torchaudio.version="${TORCHAUDIO_VERSION}" \
      com.intel.omni.onednn.package.version="${ONEDNN_VERSION}" \
      com.intel.omni.onednn.source.repository="${ONEDNN_SOURCE_REPOSITORY}" \
      com.intel.omni.onednn.source.revision="${ONEDNN_SOURCE_COMMIT}" \
      com.intel.omni.onednn.patch.sha256="${ONEDNN_PATCH_SHA256}" \
      com.intel.omni.xpu-target="${XPU_TARGET}" \
      com.intel.omni.llm-scaler.source-dirty="${LLM_SCALER_SOURCE_DIRTY}" \
      com.intel.omni.sycl-tla.revision="${SYCL_TLA_COMMIT}" \
      com.intel.omni.comfyui.version="${COMFYUI_VERSION}" \
      com.intel.omni.comfyui.revision="${COMFYUI_COMMIT}" \
      com.intel.omni.comfyui.frontend.version="${COMFYUI_FRONTEND_VERSION}" \
      com.intel.omni.comfyui.workflow-templates.version="${COMFYUI_WORKFLOW_TEMPLATES_VERSION}" \
      com.intel.omni.comfyui.manager.version="${COMFYUI_MANAGER_VERSION}" \
      com.intel.omni.comfy-kitchen.version="${COMFY_KITCHEN_VERSION}" \
      com.intel.omni.comfy-aimdo.version="${COMFY_AIMDO_VERSION}" \
      com.intel.omni.comfy-kitchen-xpu-runtime.repository="${COMFY_KITCHEN_REPOSITORY}" \
      com.intel.omni.comfy-kitchen-xpu-runtime.version="${COMFY_KITCHEN_VERSION}" \
      com.intel.omni.comfy-kitchen-xpu-runtime.revision="${COMFY_KITCHEN_COMMIT}" \
      com.intel.omni.comfy-aimdo-xpu-runtime.repository="${COMFY_AIMDO_REPOSITORY}" \
      com.intel.omni.comfy-aimdo-xpu-runtime.version="${COMFY_AIMDO_VERSION}" \
      com.intel.omni.comfy-aimdo-xpu-runtime.revision="${COMFY_AIMDO_COMMIT}" \
      com.intel.omni.comfy-gguf.revision="${COMFY_GGUF_COMMIT}" \
      com.intel.omni.comfy-nunchaku.version="${COMFY_NUNCHAKU_VERSION}" \
      com.intel.omni.comfy-nunchaku.revision="${COMFY_NUNCHAKU_COMMIT}" \
      com.intel.omni.comfy-sol-attn.repository="${COMFY_SOL_ATTN_REPOSITORY}" \
      com.intel.omni.comfy-sol-attn.revision="${COMFY_SOL_ATTN_COMMIT}"

ENV OMNIXPU_INTERPOLATE_FIX=0 \
    SOL_ATTN_XPU_EXPERIMENTAL=1 \
    UR_INCLUDE_DIR="/opt/venv/include/unified-runtime" \
    PIP_CONSTRAINT="/llm/manifests/omni-runtime-constraints.txt" \
    OMNI_IMAGE_VERSION="${IMAGE_TAG}" \
    OMNI_IMAGE_FLAVOR="comfyui" \
    OMNI_BASE_IMAGE="${BASE_IMAGE}" \
    OMNI_TORCH_VERSION="${TORCH_VERSION}" \
    OMNI_TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    OMNI_TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" \
    OMNI_ONEDNN_VERSION="${ONEDNN_VERSION}" \
    OMNI_ONEDNN_SOURCE_REPOSITORY="${ONEDNN_SOURCE_REPOSITORY}" \
    OMNI_ONEDNN_SOURCE_REVISION="${ONEDNN_SOURCE_COMMIT}" \
    OMNI_ONEDNN_PATCH_SHA256="${ONEDNN_PATCH_SHA256}" \
    OMNI_IMAGE_XPU_TARGET="${XPU_TARGET}" \
    OMNI_LLM_SCALER_SOURCE_REVISION="${LLM_SCALER_SOURCE_REVISION}" \
    OMNI_LLM_SCALER_SOURCE_DIRTY="${LLM_SCALER_SOURCE_DIRTY}" \
    OMNI_COMFYUI_VERSION="${COMFYUI_VERSION}" \
    OMNI_COMFYUI_REVISION="${COMFYUI_COMMIT}" \
    OMNI_COMFYUI_FRONTEND_VERSION="${COMFYUI_FRONTEND_VERSION}" \
    OMNI_COMFYUI_WORKFLOW_TEMPLATES_VERSION="${COMFYUI_WORKFLOW_TEMPLATES_VERSION}" \
    OMNI_COMFYUI_MANAGER_VERSION="${COMFYUI_MANAGER_VERSION}" \
    OMNI_COMFY_KITCHEN_VERSION="${COMFY_KITCHEN_VERSION}" \
    OMNI_COMFY_KITCHEN_PROVIDER_REPOSITORY="${COMFY_KITCHEN_REPOSITORY}" \
    OMNI_COMFY_KITCHEN_PROVIDER_REVISION="${COMFY_KITCHEN_COMMIT}" \
    OMNI_COMFY_AIMDO_VERSION="${COMFY_AIMDO_VERSION}" \
    OMNI_COMFY_AIMDO_PROVIDER_REPOSITORY="${COMFY_AIMDO_REPOSITORY}" \
    OMNI_COMFY_AIMDO_PROVIDER_REVISION="${COMFY_AIMDO_COMMIT}" \
    OMNI_COMFY_GGUF_REVISION="${COMFY_GGUF_COMMIT}" \
    OMNI_COMFY_NUNCHAKU_VERSION="${COMFY_NUNCHAKU_VERSION}" \
    OMNI_COMFY_NUNCHAKU_REVISION="${COMFY_NUNCHAKU_COMMIT}" \
    OMNI_COMFY_SOL_ATTN_REVISION="${COMFY_SOL_ATTN_COMMIT}"

# Retain OMIX's oneAPI environment initialization, then restore the venv's
# Torch-matched 2026.0 SYCL/UR runtime ahead of the OMIX 2026.1 build
# toolchain. Without this final reorder the inherited OMIX entrypoint mixes
# libsycl from the Torch wheel with libur_loader from the compiler image.
ENTRYPOINT ["/llm/entrypoints/omix_torch_runtime.sh"]

WORKDIR /llm/entrypoints
CMD ["/bin/bash"]
