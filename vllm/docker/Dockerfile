# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage OMIX build of llm-scaler-vllm (Intel XPU).
#
#   builder : intel/omix:latest-devel-ubuntu24.04   -- oneAPI 2025.3 DPC++ compiler
#                                                       + oneCCL/oneDNN/oneMKL.  Compiles
#                                                       all native wheels.  NOT shipped.
#   runtime : intel/pytorch:xpu-2.11.0-ubuntu24.04  -- torch 2.11.0+xpu + GPU UMD driver
#                                                       in /opt/venv.  No compiler.  Shipped.
#
# Why: the legacy single-base build (llm-scaler-platform) carried the whole oneAPI
# toolchain into the final image (~25-34GB).  Here the compiler lives only in the
# builder stage; the runtime stage receives pre-built wheels, so the shipped image
# is much smaller.
#
# Scope ("lean core"): vLLM + custom ESIMD kernels + vllm-xpu-kernels + core serving
# deps + bigdl-core (ships vllm_int4_for_multi_arc.so used by the sym_int4 path).
# Dropped vs legacy: MinerU, UCX/NIXL (PD disaggregation), triton reinstall.
#
# Build context root MUST be the `vllm/` dir (so ./patches, ./custom-esimd-kernels-vllm
# and ./examples resolve).

# =========================================================================
# Stage 1: builder -- compile every native wheel with the oneAPI toolchain
# =========================================================================
FROM intel/omix:latest-devel-ubuntu24.04 AS builder

ARG http_proxy
ARG https_proxy
ARG no_proxy
# vllm-xpu-kernels pinned base commit (identical to legacy build).
ARG VLLM_XPU_KERNELS_COMMIT=3cab97a

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    VIRTUAL_ENV=/opt/buildenv \
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

# --- torch MUST match the runtime base (2.11.0+xpu) so SyclExtension .so files link
#     against the same libtorch ABI the runtime ships (rpath $ORIGIN/../../torch/lib) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --upgrade pip wheel setuptools && \
    uv pip install torch==2.11.0+xpu torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/xpu

RUN mkdir -p ${WHEELS}

# ---------------------------------------------------------------------------
# 1a. vLLM v0.21.0 + multi-arc patch  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --depth 1 -b v0.21.0 https://github.com/vllm-project/vllm.git /src/vllm && \
    cd /src/vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS}

# ---------------------------------------------------------------------------
# 1b. custom ESIMD kernels  ->  wheel  (TORCH_XPU_ARCH_LIST=bmg-g21)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/custom-esimd-kernels-vllm && \
    TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel && \
    cp dist/*.whl ${WHEELS}/

# ---------------------------------------------------------------------------
# 1c. vllm-xpu-kernels (patch + pinned commit)  ->  wheel
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
    pip wheel --no-build-isolation --no-deps . -w ${WHEELS}

# Wheel inventory (visible in the build log for sanity).
RUN ls -la ${WHEELS}/

# =========================================================================
# Stage 2: runtime -- lightweight serving image (no compiler, no /opt/intel/oneapi)
# =========================================================================
FROM intel/pytorch:xpu-2.11.0-ubuntu24.04

ARG http_proxy
ARG https_proxy
ARG no_proxy

# The base already exports PATH=/opt/venv/bin:... and DEVICE=xpu.
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_QUANTIZE_Q40_LIB="/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so" \
    VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /llm-scaler/vllm

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
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsndfile1 libsm6 libxext6 libgl1 numactl curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install /wheels/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip install --no-deps --force-reinstall \
        /wheels/custom_esimd_kernels_vllm-*.whl \
        /wheels/vllm_xpu_kernels-*.whl && \
    pip install transformers==5.8.0 accelerate hf_transfer 'modelscope!=1.15.0' && \
    pip install librosa soundfile decord ijson && \
    pip install bigdl-core==2.4.0b2 && \
    pip uninstall -y triton triton-xpu || true && \
    pip install --no-deps triton-xpu==3.7.0 \
        --extra-index-url https://download.pytorch.org/whl/xpu && \
    TRITON=/opt/venv/lib/python3.12/site-packages/triton && \
    strip --strip-debug ${TRITON}/_C/libtriton.so && \
    rm -f ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git && \
    ln -s libMLIRDialectPlugin.so ${TRITON}/plugins/libMLIRDialectPlugin.so.22.0git

# --- entrypoint: no oneCCL vars.sh (absent in this base); multi-card relies on
#     torch 2.11 built-in XCCL. ---
ENTRYPOINT ["bash", "-c", "exec vllm serve \"$@\"", "--"]
