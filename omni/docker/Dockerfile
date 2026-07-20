# syntax=docker/dockerfile:1
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Full Omni two-stage development image built with the repository's original two-stage
# builder/runtime structure.
#
# The runtime stage deliberately uses the OMIX devel base as well.  Therefore
# the single delivered image keeps the complete oneAPI compiler and GPU runtime
# while the builder/runtime boundary still isolates the build procedure.

ARG BASE_IMAGE=intel/omix:0.1.0-devel-ubuntu24.04
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.11.0+xpu
ARG TORCHVISION_VERSION=0.26.0+xpu
ARG TORCHAUDIO_VERSION=2.11.0+xpu
ARG SGL_KERNEL_COMMIT=9aef6481524906d11956bf4b4ab8e243c5128b4e
ARG DPCPP_SYCL_TARGET=bmg
ARG MAX_JOBS=8
ARG IMAGE_TAG

# ======== Builder Stage ========
# OMIX devel already provides oneAPI 2025.3 (compiler + runtime libraries), so
# unlike llm-scaler-platform we must not reinstall a separately pinned DPC++
# compiler from Intel's apt repository.
FROM ${BASE_IMAGE} AS builder

ARG https_proxy
ARG http_proxy
ARG no_proxy=localhost,127.0.0.1,::1,intel.com,.intel.com
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG SGL_KERNEL_COMMIT
ARG DPCPP_SYCL_TARGET
ARG MAX_JOBS
ARG IMAGE_TAG

RUN test -n "${IMAGE_TAG}"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/intel/oneapi/compiler/latest/bin:${PATH}" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET} \
    OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET} \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Make both Python entrypoints and native extensions work without requiring the
# user to source setvars.sh first.  Source builds below still source setvars.sh
# so they receive the complete oneAPI compiler environment.
ENV LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/dnnl/latest/lib:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/mpi/latest/lib:${LD_LIBRARY_PATH}"

SHELL ["bash", "-c"]

# System packages
# The compiler is deliberately not installed here: it is supplied and
# maintained as part of the OMIX devel base image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ca-certificates curl ffmpeg git patch \
        python3-dev python3-venv python3-pip \
        build-essential ninja-build cmake pkg-config \
        libsndfile1 libsm6 libxext6 libgl1 \
        lsb-release numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0

# Isolated Python environment.  This avoids modifying OMIX's system Python and
# keeps every Python package used by the delivered image under /opt/venv.
RUN python3 -m venv ${VIRTUAL_ENV} && \
    pip install --upgrade pip setuptools wheel

# PyTorch
# Torch 2.11 must match the native extensions built in this image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/xpu && \
    pip install pytest

RUN mkdir -p /llm /wheels

# omni_xpu_kernel is installed AFTER sgl-kernel-xpu (below): its cute FMHA
# extension needs the cutlass-sycl headers that sgl-kernel-xpu fetches into
# build/_deps/repo-cutlass-sycl-src.
# Keep the complete local source under /llm so it can be transferred to and
# rebuilt inside the runtime image.
COPY ./omni_xpu_kernel /llm/omni_xpu_kernel

# Keep project patches for the final image as part of the source-development
# environment and use them directly from this stable path.
COPY ./patches /llm/patches

# xDiT / long-context-attention
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/feifeibear/long-context-attention.git && \
    cd long-context-attention && \
    git fetch --depth 1 origin fc5d55e61b78b3102fd824bea1791cf406cc2a4b && \
    git checkout fc5d55e61b78b3102fd824bea1791cf406cc2a4b && \
    git apply /llm/patches/yunchang_for_multi_arc.patch && \
    pip install -e . && \
    cd /llm && \
    git clone --depth 1 https://github.com/xdit-project/xDiT.git && \
    cd xDiT && \
    git fetch --depth 1 origin fb8fb0e437a8745b9629020759de31d1626a4a7b && \
    git checkout fb8fb0e437a8745b9629020759de31d1626a4a7b && \
    git apply /llm/patches/xdit_for_multi_arc.patch && \
    pip install -e .

# ComfyUI core (upstream, no patches)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git fetch --depth 1 origin bb131be9e83d2f773c90f1d6f1e4b248a498c8c5 && \
    git checkout bb131be9e83d2f773c90f1d6f1e4b248a498c8c5 && \
    pip install -r requirements.txt

# ComfyUI-OmniXPU: Intel XPU acceleration via custom node (replaces core patch)
COPY ./ComfyUI-OmniXPU /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU
RUN cd /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU && \
    git init && git add -A && \
    git -c user.name="llm-scaler" -c user.email="noreply@intel.com" \
        commit --allow-empty -m "ComfyUI-OmniXPU ${IMAGE_TAG}"

# ComfyUI custom nodes (active)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite && \
    pip install -r comfyui-videohelpersuite/requirements.txt && \
    git clone --depth 1 https://github.com/komikndr/raylight.git && \
    cd raylight && \
    git fetch --depth 1 origin c91ef47afb02e522bf08ff199ea2182642de0368 && \
    git checkout c91ef47afb02e522bf08ff199ea2182642de0368 && \
    git apply /llm/patches/raylight_for_multi_arc.patch && \
    pip install ray==2.49.2 -r requirements.txt && \
    cd /llm/ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use && \
    pip install -r comfyui-easy-use/requirements.txt && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    cd comfyui_controlnet_aux && \
    git fetch --depth 1 origin e8b689a513c3e6b63edc44066560ca5919c0576e && \
    git checkout e8b689a513c3e6b63edc44066560ca5919c0576e && \
    git apply /llm/patches/comfyui_controlnet_aux_depth_anything_v2_xpu.patch && \
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
    pip install -r requirements.txt

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
    git fetch --depth 1 origin 55b6497e5c5e4337d2de12943fd97a499bd9c2e5 && \
    git checkout 55b6497e5c5e4337d2de12943fd97a499bd9c2e5 && \
    pip install -r requirements.txt && \
    pip install --no-deps kernels==0.14.0

# ComfyUI disabled custom nodes (optional: --build-arg INSTALL_DISABLED_NODES=true)
# SeedVR2 remains optional as requested; it is not installed in the default
# image when INSTALL_DISABLED_NODES=false.
ARG INSTALL_DISABLED_NODES=false
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_DISABLED_NODES}" = "true" ]; then \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm.disabled && \
        cd comfyui-voxcpm.disabled && \
        git fetch --depth 1 origin 7875a8a36a531b7efef8cac96dc74001d3cb5895 && \
        git checkout 7875a8a36a531b7efef8cac96dc74001d3cb5895 && \
        git apply /llm/patches/comfyui_voxcpm_for_xpu.patch && \
        pip install -r requirements.txt && \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/billwuhao/ComfyUI_IndexTTS.git ComfyUI_IndexTTS.disabled && \
        cd ComfyUI_IndexTTS.disabled && \
        git apply /llm/patches/comfyui_indextts.patch && \
        pip install -r requirements.txt torchcodec==0.9.0 && \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git ComfyUI-Hunyuan3d-2-1.disabled && \
        cd ComfyUI-Hunyuan3d-2-1.disabled && \
        git fetch --depth 1 origin 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
        git checkout 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
        git apply /llm/patches/comfyui_hunyuan3d_for_xpu.patch && \
        pip install rembg realesrgan -r requirements.txt && \
        cd hy3dpaint/custom_rasterizer && python setup.py install && \
        cd ../DifferentiableRenderer && python setup.py install && \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/jtydhr88/ComfyUI-HY-Motion1.git ComfyUI-HY-Motion1.disabled && \
        cd ComfyUI-HY-Motion1.disabled && \
        git fetch --depth 1 origin 47b1d0d96eeac6d4b99509ede91490eaba93440a && \
        git checkout 47b1d0d96eeac6d4b99509ede91490eaba93440a && \
        git apply /llm/patches/comfyui_hy_motion1.patch && \
        pip install -r requirements.txt && \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git ComfyUI-SeedVR2_VideoUpscaler.disabled && \
        cd ComfyUI-SeedVR2_VideoUpscaler.disabled && \
        git fetch --depth 1 origin 4490bd1f482e026674543386bb2a4d176da245b9 && \
        git checkout 4490bd1f482e026674543386bb2a4d176da245b9 && \
        git apply /llm/patches/comfyui_seedvr2_xpu.patch && \
        pip install -r requirements.txt && \
        cd /llm/ComfyUI/custom_nodes && \
        git clone --depth 1 https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast.git ComfyUI-FlashVSR_Ultra_Fast.disabled && \
        cd ComfyUI-FlashVSR_Ultra_Fast.disabled && \
        git fetch --depth 1 origin 4820b3f02347bddcbbb9a5a85ab7638fe976366e && \
        git checkout 4820b3f02347bddcbbb9a5a85ab7638fe976366e && \
        git apply /llm/patches/comfyui_flashvsr_ultra_fast_xpu.patch; \
    fi

# Xinference
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "xinference[transformers]==1.15.0" \
        kokoro Jinja2==3.1.6 jieba ordered-set pypinyin cn2an pypinyin-dict && \
    XINFERENCE_DEVICE_UTILS="$(python -c 'import pathlib, xinference; print(pathlib.Path(xinference.__file__).parent / "device_utils.py")')" && \
    patch "${XINFERENCE_DEVICE_UTILS}" < /llm/patches/xinference_device_utils.patch

# SGLang Diffusion + sgl-kernel-xpu
# Keep the checked-out sources and native build directory in /llm so the
# kernel can be rebuilt and debugged inside a running container.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    mkdir -p /wheels && \
    cd /llm && \
    git clone --depth 1 https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git fetch --depth 1 origin 1ae3218d036c554e6c109f18233d1f75bf775c1d && \
    git checkout 1ae3218d036c554e6c109f18233d1f75bf775c1d && \
    git apply /llm/patches/sglang_diffusion_for_multi_arc.patch && \
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
    git fetch --depth 1 origin ${SGL_KERNEL_COMMIT} && \
    git checkout ${SGL_KERNEL_COMMIT} && \
    pip wheel -v . \
        --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET} \
        --wheel-dir /wheels --no-deps && \
    pip install --no-deps /wheels/sgl_kernel-*.whl && \
    cp -r /llm/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion \
        /llm/ComfyUI/custom_nodes/

# omni_xpu_kernel (installed here so its cute FMHA ext can use sgl-kernel-xpu's
# cutlass-sycl headers). CUTLASS_SYCL_ROOT points at the fetched tree; the cute
# ext is AOT-compiled for OMNI_XPU_DEVICE (bmg = Arc B-series).
# The setup.py source also contains the Torch-2.11/oneDNN include-order fix; the
# resulting extension must reference dnnl_primitive_attr_set_zero_points
# (without the _v2 suffix).
ENV CUTLASS_SYCL_ROOT=/llm/sgl-kernel-xpu/build/_deps/repo-cutlass-sycl-src \
    OMNI_XPU_REQUIRE_CUTE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /llm/omni_xpu_kernel && \
    grep -Fqx "__version__ = \"${IMAGE_TAG}\"" omni_xpu_kernel/_version.py && \
    pip wheel -v . --wheel-dir /wheels --no-deps --no-build-isolation && \
    pip install --no-deps --force-reinstall /wheels/omni_xpu_kernel-*.whl && \
    EXT="$(find ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/omni_xpu_kernel -type f -name '_C*.so' | head -1)" && \
    test -n "${EXT}" && \
    nm -D "${EXT}" | grep -q ' U dnnl_primitive_attr_set_zero_points$' && \
    ! nm -D "${EXT}" | grep -q 'dnnl_primitive_attr_set_zero_points_v2'

# Build-time smoke test.  GPU execution is tested after image creation with
# /dev/dri mounted, but compiler availability and native-extension loading can
# already be verified here without access to a physical XPU.
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && \
    command -v icpx && \
    command -v cmake && \
    command -v ninja && \
    test -f /llm/omni_xpu_kernel/setup.py && \
    test -d /llm/sgl-kernel-xpu/build && \
    python -c "import torch; from omni_xpu_kernel import _C; import sgl_kernel; print(torch.__version__); print('native kernel imports OK')"

# ComfyUI-Manager dependencies
# Install these during image build so a fresh/ephemeral container does not need
# PyPI access or spend about one minute installing GitPython at startup.  This
# deliberately comes after native kernel builds to preserve their expensive
# Docker cache when Manager requirements change.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /llm/ComfyUI/custom_nodes/comfyui-manager/requirements.txt && \
    python -c "import git; print('ComfyUI-Manager dependencies OK')"

# Application files
COPY ./workflows/* /llm/ComfyUI/user/default/workflows/
COPY ./example_inputs/* /llm/ComfyUI/input/
COPY ./tools/* /llm/tools/
COPY ./entrypoints/* /llm/entrypoints/

# Wheel inventory is useful in CI logs and remains available in the runtime.
RUN ls -lh /wheels

# ======== Runtime Stage ========
# This is intentionally OMIX devel rather than a compiler-free PyTorch runtime.
# It supplies a maintained, coherent oneAPI compiler + GPU runtime environment
# without manually selecting and copying individual native libraries.
FROM ${BASE_IMAGE} AS runtime

ARG https_proxy
ARG http_proxy
ARG no_proxy=localhost,127.0.0.1,::1,intel.com,.intel.com
ARG PYTHON_VERSION
ARG DPCPP_SYCL_TARGET
ARG MAX_JOBS
ARG IMAGE_TAG

RUN test -n "${IMAGE_TAG}"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/intel/oneapi/compiler/latest/bin:${PATH}" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET} \
    OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET} \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
    CUTLASS_SYCL_ROOT=/llm/sgl-kernel-xpu/build/_deps/repo-cutlass-sycl-src \
    OMNI_XPU_REQUIRE_CUTE=1

ENV LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/dnnl/latest/lib:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/tbb/latest/lib:/opt/intel/oneapi/mpi/latest/lib:${LD_LIBRARY_PATH}"

SHELL ["bash", "-c"]

# System packages
# Runtime applications and in-container source builds need the same OS-level
# headers and build tools. oneAPI itself is already supplied by OMIX devel and
# is intentionally not reinstalled from a separate apt repository.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ca-certificates curl ffmpeg git patch \
        python3-dev python3-venv python3-pip \
        build-essential ninja-build cmake pkg-config \
        libsndfile1 libsm6 libxext6 libgl1 \
        lsb-release numactl wget vim linux-libc-dev \
        libopengl0 libglx0 libxrender1 libxfixes3 libx11-dev \
        libxi6 libxxf86vm1 libxcursor1 libxrandr2 libxinerama1 \
        libxkbcommon0

# Copy built artifacts from builder
# Copy complete environments/trees, not a hand-maintained list of shared
# libraries.  Editable installs continue to resolve because their /llm paths
# are identical in builder and runtime.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /llm /llm
COPY --from=builder /wheels /wheels

# Fix for multi-XPU support: copy SYCL native binary files to a common library path
# The files already ship with this OMIX base; no files are removed.
RUN cp /opt/intel/oneapi/compiler/latest/lib/libsycl-native-*.spv /usr/local/lib/

LABEL org.opencontainers.image.version="${IMAGE_TAG}"

ENV CCL_SYCL_ALLTOALL_ARC_LL=1 \
    CCL_SYCL_CCL_BARRIER=1 \
    OMNIXPU_INTERPOLATE_FIX=0 \
    OMNI_IMAGE_VERSION="${IMAGE_TAG}"

# Final-image smoke test: prove that this stage—not only the builder—contains
# the compiler, source/build trees, application installs, and loadable native
# extensions.  Physical XPU and multi-card execution are validated after build.
RUN source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && \
    command -v icpx && \
    command -v cmake && \
    command -v ninja && \
    test -f /llm/omni_xpu_kernel/setup.py && \
    test -d /llm/sgl-kernel-xpu/build && \
    test -f /llm/ComfyUI/main.py && \
    test -d /llm/sglang/python/sglang && \
    EXT="$(find ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/omni_xpu_kernel -type f -name '_C*.so' | head -1)" && \
    test -n "${EXT}" && \
    nm -D "${EXT}" | grep -q ' U dnnl_primitive_attr_set_zero_points$' && \
    ! nm -D "${EXT}" | grep -q 'dnnl_primitive_attr_set_zero_points_v2' && \
    ! ldd "${EXT}" | grep -q 'not found' && \
    python -c "import git; import torch; from omni_xpu_kernel import _C; import sgl_kernel; import sglang; print(torch.__version__); print('runtime development environment OK')"

# No strip, source removal, build-directory removal, or compiler removal here.
# The final image is intentionally a complete development image.
WORKDIR /llm/entrypoints
CMD ["/bin/bash"]
