# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Three-stage OMIX build of llm-scaler-vllm (Intel XPU).
#
#   rust-build : ubuntu:22.04                         -- builds vLLM Rust extensions.
#   builder    : intel/omix:0.1.0-devel-ubuntu24.04  -- oneAPI 2025.3 DPC++ compiler
#                                                         + oneCCL/oneDNN/oneMKL.  Compiles
#                                                         all native wheels.  NOT shipped.
#   runtime    : intel/omix:0.1.0-devel-ubuntu24.04  -- torch 2.12.0+xpu + GPU UMD driver
#                                                         in /opt/venv.  Shipped.
#
# Why: vLLM v0.26.0 requires separately built Rust extensions, while an Intel
# PyTorch 2.12 runtime image is not available.  OMIX provides the required GPU
# runtime, and the final stage installs torch 2.12.0+xpu into an isolated venv.
# Build outputs still cross stage boundaries only as source trees and wheels.
#
# Scope ("lean core"): vLLM + custom ESIMD kernels + vllm-xpu-kernels + core serving
# deps + bigdl-core (ships vllm_int4_for_multi_arc.so used by the sym_int4 path).
# Dropped vs legacy: MinerU and UCX/NIXL (PD disaggregation).
#
# Build context root MUST be the `vllm/` dir (so ./patches, ./custom-esimd-kernels-vllm
# and ./examples resolve).

# =========================================================================
# Stage 1: rust-build -- compile vLLM Rust extensions
# =========================================================================
FROM ubuntu:22.04 AS rust-build

ARG VLLM_VERSION=v0.26.0

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_BUILD_JOBS=4

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential unzip python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /workspace/vllm

WORKDIR /workspace/vllm

RUN ./tools/install_protoc.sh && \
    python3 -m pip install --no-cache-dir -r requirements/build/rust.txt

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    bash build_rust.sh

# =========================================================================
# Stage 2: builder -- compile every native wheel with the oneAPI toolchain
# =========================================================================
FROM intel/omix:0.1.0-devel-ubuntu24.04 AS builder

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG VLLM_VERSION=v0.26.0
# vllm-xpu-kernels pinned base commit (identical to legacy build).
ARG VLLM_XPU_KERNELS_COMMIT=a6929869587fa6c40dbe393eedc96eeab076bfbb

ENV VIRTUAL_ENV=/opt/buildenv \
    PATH="/root/.local/bin:/opt/buildenv/bin:${PATH}" \
    UV_HTTP_TIMEOUT=500 \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    UV_INDEX_STRATEGY="unsafe-best-match" \
    UV_LINK_MODE="copy" \
    VLLM_TARGET_DEVICE=xpu \
    WHEELS=/wheels

# Required: build RUN steps use `source .../setvars.sh` and `set -eo pipefail`,
# both bash-only (unavailable under the default /bin/sh).
SHELL ["bash", "-c"]

COPY ./patches/vllm_for_multi_arc.patch ./patches/vllm_xpu_kernels.patch /tmp/
COPY ./custom-esimd-kernels-vllm /src/custom-esimd-kernels-vllm

# --- build tooling (the oneAPI DPC++ compiler is already in the OMIX devel base) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        git curl ca-certificates \
        python3-dev python3-venv python3-pip \
        ninja-build cmake pkg-config numactl

# --- isolated build venv (avoid OMIX system-python PEP 668 block; clean resolution) ---
RUN python3 -m venv ${VIRTUAL_ENV}

# --- uv for fast, deterministic installs (matches legacy build) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# --- torch MUST match the runtime venv (2.12.0+xpu) so SyclExtension .so files
#     link against the same libtorch ABI (rpath $ORIGIN/../../torch/lib) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --upgrade pip wheel setuptools && \
    uv pip install torch==2.12.0+xpu torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/xpu

RUN mkdir -p ${WHEELS}

# ---------------------------------------------------------------------------
# 2a. vLLM v0.26.0 + multi-arc patch  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --depth 1 -b "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git /src/vllm && \
    cd /src/vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind

COPY --from=rust-build /workspace/vllm/vllm/vllm-rs /src/vllm/vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/vllm/_rust_*.so /src/vllm/vllm/

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build

# ---------------------------------------------------------------------------
# 2b. custom ESIMD kernels  ->  wheel  (TORCH_XPU_ARCH_LIST=bmg-g21)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/custom-esimd-kernels-vllm && \
    TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel && \
    cp dist/*.whl ${WHEELS}/ && \
    rm -rf build dist

# ---------------------------------------------------------------------------
# 2c. vllm-xpu-kernels (patch + pinned commit)  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git /src/vllm-xpu-kernels && \
    cd /src/vllm-xpu-kernels && \
    git checkout ${VLLM_XPU_KERNELS_COMMIT} && \
    git apply /tmp/vllm_xpu_kernels.patch && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i '/^torch==/s/^/# /' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt && \
    uv pip install -r requirements.txt && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS} && \
    rm -rf build .deps

# Wheel inventory (visible in the build log for sanity).
RUN ls -la ${WHEELS}/

# =========================================================================
# Stage 3: runtime -- production wheel install on OMIX with torch 2.12
# =========================================================================
FROM intel/omix:0.1.0-devel-ubuntu24.04 AS runtime

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG ONECCL_INSTALLER="intel-oneccl-2021.15.9.14_offline.sh"
ARG ONECCL_INSTALLER_SHA256="f7ab81b6ed1b10dd35fadec366a78046d8af214888dfd625047ce8953d5aa4ef"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_QUANTIZE_Q40_LIB="/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so" \
    VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /llm-scaler/vllm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsndfile1 libsm6 libxext6 libgl1 numactl curl \
        python3-dev python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv ${VIRTUAL_ENV} && \
    pip install --upgrade pip wheel setuptools && \
    pip install torch==2.12.0+xpu torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/xpu

# --- patched source trees (post-patch vLLM, vllm-xpu-kernels, and the ESIMD
#     kernels) kept in the shipped image for reference / editable-install dev
#     workflows.  These are plain COPY (not bind-mounted): unlike the wheels
#     below, they're meant to persist in the final image. ---
COPY --from=builder /src/vllm ./vllm
COPY --from=builder /src/vllm-xpu-kernels ./vllm-xpu-kernels
COPY --from=builder /src/custom-esimd-kernels-vllm ./custom-esimd-kernels-vllm

# --- thin runtime OS deps, vLLM + kernel wheels, lean-core serving deps, and the
#     triton-xpu fix/shrink all in ONE layer: cleanup must happen in the same RUN
#     that creates the bloat, otherwise a later layer can't reclaim the space.
#     Wheels are bind-mounted (not COPY'd) from the builder stage: a COPY would
#     bake them into their own image layer that a later `rm` could never reclaim
#     (union FS layers are immutable); the bind mount makes them visible only for
#     this RUN's lifetime, so nothing from /wheels ever lands in the final image. ---
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    set -e && \
    curl -fL --retry 3 -o "${ONECCL_INSTALLER}" "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.9/${ONECCL_INSTALLER}" && \
    printf "%s  %s\n" "${ONECCL_INSTALLER_SHA256}" "${ONECCL_INSTALLER}" > /tmp/oneccl.sha256 && \
    sha256sum -c /tmp/oneccl.sha256 && \
    rm -f /tmp/oneccl.sha256 && \
    bash "${ONECCL_INSTALLER}" -s -x -f /tmp && \
    ONECCL_CUP="/tmp/${ONECCL_INSTALLER%.sh}/packages/intel.oneapi.lin.ccl.runtime,v=2021.15.9+13/cupPayload.cup" && \
    ONECCL_DST="/tmp/oneccl-runtime" && \
    python3 -m zipfile -e "${ONECCL_CUP}" "${ONECCL_DST}" && \
    ONECCL_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib" && \
    rm -f "${ONECCL_LIB}/libccl.so" "${ONECCL_LIB}/libccl.so.1" "${ONECCL_LIB}/libccl_openmp.so" "${ONECCL_LIB}/libccl_openmp.so.0" && \
    ln -sf libccl.so.1 "${ONECCL_LIB}/libccl.so" && \
    ln -sf libccl.so.1.0 "${ONECCL_LIB}/libccl.so.1" && \
    ln -sf libccl_openmp.so.0 "${ONECCL_LIB}/libccl_openmp.so" && \
    ln -sf libccl_openmp.so.0.1 "${ONECCL_LIB}/libccl_openmp.so.0" && \
    ONECCL_CPU_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib/ccl/cpu/lib" && \
    rm -f "${ONECCL_CPU_LIB}/libccl.so" "${ONECCL_CPU_LIB}/libccl.so.1" "${ONECCL_CPU_LIB}/libccl_openmp.so" "${ONECCL_CPU_LIB}/libccl_openmp.so.0" && \
    ln -sf libccl.so.1 "${ONECCL_CPU_LIB}/libccl.so" && \
    ln -sf libccl.so.1.0 "${ONECCL_CPU_LIB}/libccl.so.1" && \
    ln -sf libccl_openmp.so.0 "${ONECCL_CPU_LIB}/libccl_openmp.so" && \
    ln -sf libccl_openmp.so.0.1 "${ONECCL_CPU_LIB}/libccl_openmp.so.0" && \
    mkdir -p /opt/intel/oneapi/ccl && \
    rm -rf /opt/intel/oneapi/ccl/2021.15 && \
    cp -a /tmp/oneccl-runtime/_installdir/ccl/2021.15 /opt/intel/oneapi/ccl/ && \
    rm "${ONECCL_INSTALLER}" && \
    echo "source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force" >> /root/.bashrc && \
    rm -f /opt/intel/oneapi/ccl/latest && \
    ln -s /opt/intel/oneapi/ccl/2021.15 /opt/intel/oneapi/ccl/latest && \
    find /opt/intel/oneapi/ccl -mindepth 1 -maxdepth 1 -type d -not -name 2021.15 -exec rm -rf {} + && \
    rm -rf "/tmp/${ONECCL_INSTALLER%.sh}" /tmp/oneccl-runtime && \
    pip install /wheels/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-deps --force-reinstall \
        /wheels/custom_esimd_kernels_vllm-*.whl \
        /wheels/vllm_xpu_kernels-*.whl && \
    pip install transformers==5.8.0 accelerate hf_transfer 'modelscope!=1.15.0' && \
    pip install librosa soundfile decord ijson && \
    pip install bigdl-core==2.4.0b2 && \
    (pip uninstall -y triton triton-xpu || true) && \
    pip install --no-deps triton-xpu==3.7.1 \
        --extra-index-url https://download.pytorch.org/whl/xpu && \
    (pip uninstall -y oneccl oneccl-devel || true) && \
    rm -f /opt/venv/lib/libccl.so /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 \
        /opt/venv/lib/libccl_openmp.so /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so.0.1 && \
    ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0 /opt/venv/lib/libccl.so.1 && \
    ln -sf /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so && \
    ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl_openmp.so.0.1 /opt/venv/lib/libccl_openmp.so.0 && \
    ln -sf /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so && \
    TRITON=/opt/venv/lib/python3.12/site-packages/triton && \
    strip --strip-debug ${TRITON}/_C/libtriton.so && \
    rm -f ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git && \
    ln -s libMLIRDialectPlugin.so ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git

ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force && exec vllm serve \"$@\"", "--"]
