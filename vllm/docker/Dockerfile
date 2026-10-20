# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Three-stage OMIX build of llm-scaler-vllm (Intel XPU).
#
#   rust-build : ubuntu:22.04                         -- builds vLLM Rust extensions.
#   builder    : intel/omix:0.4.0-devel-ubuntu26.04  -- oneAPI 2026.1.1 DPC++ compiler
#                                                         + oneCCL/oneDNN/oneMKL.  Compiles
#                                                         all native wheels.  NOT shipped.
#   runtime    : intel/omix:0.4.0-devel-ubuntu26.04  -- torch 2.14.0+xpu + GPU UMD driver
#                                                         in /opt/venv.  Shipped.
#
# vLLM v0.26.0 requires separately built Rust extensions. The Intel PyTorch
# runtime is installed into a Python 3.12.14 isolated venv. OMIX supplies
# the compiler and GPU UMD; its default Python 3.14 is not the wheel interpreter.
# Build outputs cross stage boundaries only as Rust artifacts, source trees, and wheels.
#
# Scope ("lean core"): vLLM + custom ESIMD kernels + vllm-xpu-kernels + LMCache
# with the KVWeave CPU codec + core serving deps + bigdl-core (ships
# vllm_int4_for_multi_arc.so used by the sym_int4 path).
# Dropped vs legacy: MinerU and UCX/NIXL (PD disaggregation).
#
# Build context root MUST be the `vllm/` dir (so ./patches, ./custom-esimd-kernels-vllm
# and ./examples resolve).

# =========================================================================
# Stage 1: rust-build -- compile vLLM Rust extensions
# =========================================================================
FROM ubuntu:22.04 AS rust-build

ARG http_proxy
ARG https_proxy
ARG no_proxy
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

RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    ./tools/install_protoc.sh && \
    python3 -m pip install --no-cache-dir -r requirements/build/rust.txt

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    bash build_rust.sh

# =========================================================================
# Stage 2: builder -- compile every native wheel with the oneAPI toolchain
# =========================================================================
FROM intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f AS python-base

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PYTHON_VERSION=3.12.14
ENV PATH="/root/.local/bin:${PATH}" \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_HTTP_TIMEOUT=500 \
    UV_LINK_MODE=copy \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_NATIVE_TLS=true \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu https://wheels.vllm.ai/xpu" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu https://wheels.vllm.ai/xpu" \
    UV_CONSTRAINT=/opt/xpu.constraints.txt \
    PIP_CONSTRAINT=/opt/xpu.constraints.txt \
    UV_OVERRIDE=/opt/intel-runtime.overrides.txt
SHELL ["bash", "-c"]
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh && \
    uv python install "${PYTHON_VERSION}"
COPY ./docker/runtime-xpu.constraints.txt ./docker/intel-runtime.overrides.txt \
     ./docker/lmcache_xpu_runtime_requirements.txt /opt/
RUN sed -n -E 's/^([A-Za-z0-9_.-]+[=<>!~]+[^ ]+).*/\1/p' \
        /opt/runtime-xpu.constraints.txt /opt/lmcache_xpu_runtime_requirements.txt \
        > /opt/xpu.constraints.txt

FROM python-base AS builder

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG VLLM_VERSION=v0.26.0
# vllm-xpu-kernels pinned base commit (aligned with Dockerfile.dev).
ARG VLLM_XPU_KERNELS_COMMIT=a6929869587fa6c40dbe393eedc96eeab076bfbb
# LMCache v0.4.7 and KVWeave sources validated with vLLM v0.26.0.
ARG LMCACHE_COMMIT=307439a1d7a4eb49fcaac1b565ad5c3b3b3f918f
ARG KVWEAVE_COMMIT=4eacc5a8613060a066655acb4f20edbd63a40703

ENV VIRTUAL_ENV=/opt/buildenv \
    PATH="/root/.local/bin:/opt/buildenv/bin:${PATH}" \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY="unsafe-best-match" \
    UV_LINK_MODE="copy" \
    VLLM_TARGET_DEVICE=xpu \
    WHEELS=/wheels

# Required: build RUN steps use `source .../setvars.sh` and `set -eo pipefail`,
# both bash-only (unavailable under the default /bin/sh).
SHELL ["bash", "-c"]

COPY ./patches/vllm_for_multi_arc.patch ./patches/vllm_xpu_kernels.patch \
     ./patches/lmcache_kvweave.patch ./patches/kvweave_torch_2_14.patch /tmp/
COPY ./custom-esimd-kernels-vllm /src/custom-esimd-kernels-vllm

# --- build tooling (the oneAPI DPC++ compiler is already in the OMIX devel base) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        git curl ca-certificates \
        ninja-build cmake pkg-config numactl

# --- isolated build venv (avoid OMIX system-python PEP 668 block; clean resolution) ---
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    uv venv --python "${PYTHON_VERSION}" \
        --seed ${VIRTUAL_ENV}

# --- torch MUST match the runtime venv (2.14.0+xpu) so SyclExtension .so files
#     link against the same libtorch ABI (rpath $ORIGIN/../../torch/lib) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    uv pip install --upgrade pip wheel setuptools && \
    uv pip install torch==2.14.0+xpu torchaudio==2.11.0+xpu \
        torchvision==0.29.0+xpu triton==3.8.0+xpu && \
    uv pip install -r /opt/intel-runtime.overrides.txt

RUN mkdir -p ${WHEELS}

# ---------------------------------------------------------------------------
# 2a. vLLM v0.26.0 + multi-arc patch  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone --depth 1 -b "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git /src/vllm && \
    cd /src/vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    sed -i 's/^torch==2.12.0$/torch==2.14.0/' requirements/xpu.txt && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind

COPY --from=rust-build /workspace/vllm/vllm/vllm-rs /src/vllm/vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/vllm/_rust_*.so /src/vllm/vllm/

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    mkdir -p ${WHEELS}/build-logs && \
    (pip wheel --verbose --no-build-isolation --no-deps . -w ${WHEELS} \
        > ${WHEELS}/build-logs/vllm.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/vllm.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/vllm.log && \
    rm -rf build

# ---------------------------------------------------------------------------
# 2b. custom ESIMD kernels  ->  wheel  (TORCH_XPU_ARCH_LIST=bmg-g21)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/custom-esimd-kernels-vllm && \
    sed -i 's/c++17/c++20/g' setup.py esimd_build_extention.py && \
    mkdir -p ${WHEELS}/build-logs && \
    (TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel \
        > ${WHEELS}/build-logs/custom-esimd.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/custom-esimd.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/custom-esimd.log && \
    cp dist/*.whl ${WHEELS}/ && \
    rm -rf build dist

# ---------------------------------------------------------------------------
# 2c. vllm-xpu-kernels (patch + pinned commit)  ->  wheel
# ---------------------------------------------------------------------------
# oneDNN 3.12.3 fixes zero-pool pause/resume under native SYCL graph recording.
ARG ONEDNN_COMMIT=6db5f1ba860ca8f65c3f65c4498ad867e9642447
# analytics-zoo/vllm-xpu-kernels#21, commit 488ad801e0f5b32b56235243d21e3ec57a16a54f.
# Preserve the pinned kernel base; compile the packed MTP cache fix into the wheel.
COPY ./patches/vllm_xpu_gdn_mtp_packed_cache.patch /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git /src/vllm-xpu-kernels && \
    cd /src/vllm-xpu-kernels && \
    git checkout ${VLLM_XPU_KERNELS_COMMIT} && \
    git apply /tmp/vllm_xpu_kernels.patch && \
    git apply --check /tmp/vllm_xpu_gdn_mtp_packed_cache.patch && \
    git apply /tmp/vllm_xpu_gdn_mtp_packed_cache.patch && \
    sed -i "s/80afa71049cd69a3df32adcccb623b12cd7baa22/${ONEDNN_COMMIT}/" CMakeLists.txt && \
    sed -i 's/torch == 2.12.0+xpu/torch == 2.14.0+xpu/' pyproject.toml && \
    sed -i 's/set(CMAKE_CXX_STANDARD 17)/set(CMAKE_CXX_STANDARD 20)/g' cmake/utils.cmake && \
    sed -i 's/ -Werror")/ -Werror -Wno-error=deprecated-this-capture")/' CMakeLists.txt && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i '/^torch==/s/^/# /' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt && \
    uv pip install -r requirements.txt && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    (pip wheel --verbose --no-build-isolation --no-deps . -w ${WHEELS} \
        > ${WHEELS}/build-logs/vllm-xpu-kernels.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/vllm-xpu-kernels.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/vllm-xpu-kernels.log && \
    rm -rf build .deps

# ---------------------------------------------------------------------------
# 2d. KVWeave CPU codec (torch 2.14 XPU metadata patch)  ->  wheel
# ---------------------------------------------------------------------------
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    git clone --filter=blob:none --no-checkout \
        https://github.com/open-edge-platform/edge-ai-libraries.git \
        /src/edge-ai-libraries && \
    cd /src/edge-ai-libraries && \
    git sparse-checkout set libraries/kv-quant-offload && \
    git checkout "${KVWEAVE_COMMIT}" && \
    git apply --check /tmp/kvweave_torch_2_14.patch && \
    git apply /tmp/kvweave_torch_2_14.patch && \
    git diff --check

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/edge-ai-libraries/libraries/kv-quant-offload && \
    sed -i 's/c++17/c++20/g' setup.py && \
    sed -i '/"-D_FORTIFY_SOURCE=3"/i\        "-U_FORTIFY_SOURCE",' setup.py && \
    (MAX_JOBS=8 KVWEAVE_ISA=avx2 KVWEAVE_XPU=0 \
        pip wheel --verbose --no-build-isolation --no-deps . -w ${WHEELS} \
        > ${WHEELS}/build-logs/kvweave.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/kvweave.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/kvweave.log && \
    rm -rf build

# ---------------------------------------------------------------------------
# 2e. LMCache v0.4.7 + MP hybrid/KVWeave patch  ->  SYCL wheel
# ---------------------------------------------------------------------------
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    git clone --filter=blob:none https://github.com/LMCache/LMCache.git \
        /src/LMCache && \
    cd /src/LMCache && \
    git checkout "${LMCACHE_COMMIT}" && \
    git apply --check /tmp/lmcache_kvweave.patch && \
    git apply /tmp/lmcache_kvweave.patch && \
    git diff --check

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/LMCache && \
    sed -i 's/c++17/c++20/g' setup.py && \
    uv pip install -r requirements/build.txt && \
    (BUILD_WITH_SYCL=1 MAX_JOBS=8 \
        pip wheel --verbose --no-build-isolation --no-deps . -w ${WHEELS} \
        > ${WHEELS}/build-logs/lmcache.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/lmcache.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/lmcache.log && \
    rm -rf build

# Wheel inventory (visible in the build log for sanity).
RUN ls -la ${WHEELS}/

# Final v0.26 compatibility patches are kept after the independent native
# builds so a Python compatibility fix does not invalidate their compilation.
# Backport the accepted-count ownership fix from vLLM#51599.
COPY ./patches/gemma4_graph_scalar.patch ./patches/xpu_memory_accounting.patch \
     ./patches/xpu_allreduce_functional.patch \
     ./patches/vllm_mtp_accepted_count_ownership.patch /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm && \
    git apply --check /tmp/gemma4_graph_scalar.patch && \
    git apply /tmp/gemma4_graph_scalar.patch && \
    git apply --check /tmp/xpu_memory_accounting.patch && \
    git apply /tmp/xpu_memory_accounting.patch && \
    git apply --check /tmp/xpu_allreduce_functional.patch && \
    git apply /tmp/xpu_allreduce_functional.patch && \
    git apply --check /tmp/vllm_mtp_accepted_count_ownership.patch && \
    git apply /tmp/vllm_mtp_accepted_count_ownership.patch && \
    git diff --check && \
    mkdir -p ${WHEELS}/initial-vllm && \
    for wheel in ${WHEELS}/vllm-*.whl; do mv "$wheel" ${WHEELS}/initial-vllm/; done && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    (pip wheel --verbose --no-build-isolation --no-deps . -w ${WHEELS} \
        > ${WHEELS}/build-logs/vllm-compatibility.log 2>&1 || \
        { tail -200 ${WHEELS}/build-logs/vllm-compatibility.log; exit 1; }) && \
    tail -40 ${WHEELS}/build-logs/vllm-compatibility.log && \
    rm -rf build

# =========================================================================
# Stage 3: runtime -- production wheel install on OMIX with torch 2.14
# =========================================================================
FROM python-base AS runtime

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_QUANTIZE_Q40_LIB="/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so" \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    LD_LIBRARY_PATH="/opt/venv/lib:/usr/local/lib"

WORKDIR /llm-scaler/vllm

# Use the same shared-package constraints in every runtime installation,
# including vLLM's initial dependency resolution. Do not force a late downgrade.
# The shared python-base applies the same constraints and runtime overrides
# to both environments before any package resolution.

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsndfile1 libsm6 libxext6 libgl1 numactl curl && \
    rm -rf /var/lib/apt/lists/* && \
    uv venv --python "${PYTHON_VERSION}" \
        --seed ${VIRTUAL_ENV} && \
    uv pip install --upgrade pip wheel setuptools && \
    uv pip install torch==2.14.0+xpu torchaudio==2.11.0+xpu \
        torchvision==0.29.0+xpu triton==3.8.0+xpu && \
    uv pip install -r /opt/intel-runtime.overrides.txt

# --- patched source trees (vLLM, vllm-xpu-kernels, and the ESIMD kernels) kept
#     in the shipped image for reference / editable-install dev workflows.
#     These are plain COPY (not bind-mounted): unlike the wheels
#     below, they're meant to persist in the final image. ---
COPY --from=builder /src/vllm ./vllm
COPY --from=builder /src/vllm-xpu-kernels ./vllm-xpu-kernels
COPY --from=builder /src/custom-esimd-kernels-vllm ./custom-esimd-kernels-vllm
COPY ./docker/check_runtime_dependencies.py ./docker/check_runtime_observability.py /opt/runtime-checks/

# --- thin runtime OS deps, vLLM + kernel wheels, lean-core serving deps, and the
#     Triton shrink all in ONE layer: cleanup must happen in the same RUN
#     that creates the bloat, otherwise a later layer can't reclaim the space.
#     Wheels are bind-mounted (not COPY'd) from the builder stage: a COPY would
#     bake them into their own image layer that a later `rm` could never reclaim
#     (union FS layers are immutable); the bind mount makes them visible only for
#     this RUN's lifetime, so nothing from /wheels ever lands in the final image. ---
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -e && \
    uv pip install /wheels/vllm-*.whl && \
    uv pip install --no-deps --reinstall \
        /wheels/custom_esimd_kernels_vllm-*.whl \
        /wheels/vllm_xpu_kernels-*.whl && \
    uv pip install transformers==5.8.0 accelerate hf_transfer 'modelscope!=1.15.0' && \
    uv pip install librosa soundfile decord ijson && \
    uv pip install bigdl-core==2.4.0b2 && \
    pip install --require-hashes --no-deps -r /opt/lmcache_xpu_runtime_requirements.txt && \
    uv pip install --no-deps --reinstall \
        /wheels/kvweave-*.whl \
        /wheels/lmcache-*.whl && \
    python -c 'import lmcache.xpu_ops, kvweave.kvweave_quant; print(lmcache.xpu_ops.__file__); print(kvweave.kvweave_quant.__file__)' && \
    python -c 'from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import LMCacheMPConnector; assert LMCacheMPConnector.__module__ == "lmcache.integration.vllm.lmcache_mp_connector"; print(LMCacheMPConnector)' && \
    python -c 'import importlib.metadata as m; assert m.version("torch") == "2.14.0+xpu"; assert m.version("triton") == "3.8.0+xpu"; assert m.version("triton-xpu") == "3.8.0"; assert m.version("oneccl") == m.version("oneccl-devel") == "2022.1.2"' && \
    python /opt/runtime-checks/check_runtime_dependencies.py \
        --constraints /opt/runtime-xpu.constraints.txt \
        --constraints /opt/lmcache_xpu_runtime_requirements.txt && \
    python /opt/runtime-checks/check_runtime_observability.py && \
    TRITON=/opt/venv/lib/python3.12/site-packages/triton && \
    strip --strip-debug ${TRITON}/_C/libtriton.so

# Keep wheel runtime libraries ahead of the matching OMIX system installation.
ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && export LD_LIBRARY_PATH=\"/opt/venv/lib:${LD_LIBRARY_PATH:-}\" && exec vllm serve \"$@\"", "--"]
