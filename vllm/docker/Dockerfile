# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# OMIX build of llm-scaler-vllm (Intel XPU).
#
#   rust-build      : ubuntu:22.04 -- builds vLLM Rust extensions.
#   *-source        : clone + patch each component; no compilation.
#   build-base      : intel/omix:0.4.0-devel-ubuntu26.04 -- oneAPI 2026.1.1 DPC++
#                     compiler + torch 2.14.0+xpu build venv.  NOT shipped.
#   *-build         : one native wheel per stage; independent stages build in parallel.
#   *-wheel, wheels : export targets (`--output type=local,dest=...`).
#   runtime-lock    : resolves docker/runtime-requirements.txt (no compilation).
#   runtime         : torch 2.14.0+xpu + GPU UMD driver in /opt/venv.  Shipped.
#
# vLLM v0.31.0 requires separately built Rust extensions. The Intel PyTorch
# runtime is installed into a Python 3.12.14 isolated venv. OMIX supplies
# the compiler and GPU UMD; its default Python 3.14 is not the wheel interpreter.
#
# Scope: vLLM v0.31.0 + vllm-xpu-kernels with Qwen3.8-Flash-Next and
# MiniCPM-V 4.7 support + core serving deps.
#
# The runtime installs the native wheels from the `wheels` stage. Replace
# that stage with `--build-context wheels=<dir>` to reuse wheels exported from
# the same checkout; their manifest must match this build's inputs.
#
# Build context root MUST be the `vllm/` dir (so ./patches and ./docker resolve).

ARG VLLM_VERSION=v0.31.0
# vllm-xpu-kernels base of the 0.1.15.4 release required by vLLM v0.31.0.
ARG VLLM_XPU_KERNELS_COMMIT=ddf336d86e3c8602888572a3502f951abd51df12
ARG PYTHON_VERSION=3.12.14

# =========================================================================
# rust-build -- compile vLLM Rust extensions
# =========================================================================
FROM ubuntu:22.04 AS rust-build

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG VLLM_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_BUILD_JOBS=4 \
    VLLM_RS_BUILD_VERSION=0.31.0+xpu.llmscaler

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential unzip python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm.git /workspace/vllm

WORKDIR /workspace/vllm

RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    python3 -m pip install --no-cache-dir -r requirements/build/rust.txt

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    bash tools/build_rust.sh

# =========================================================================
# python-base -- uv, Python and the shared XPU version policy
# =========================================================================
FROM intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f AS python-base

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PYTHON_VERSION
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
# Builds see only the shared ABI/version policy. The runtime lock is applied
# in the runtime stage, so changing it does not invalidate native builds.
COPY ./docker/runtime-xpu.constraints.txt ./docker/intel-runtime.overrides.txt /opt/
RUN sed -n -E 's/^([A-Za-z0-9_.-]+[=<>!~]+[^ ]+).*/\1/p' \
        /opt/runtime-xpu.constraints.txt > /opt/xpu.constraints.txt

# =========================================================================
# Source stages -- clone and patch; no compilation
# =========================================================================
FROM python-base AS source-base

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing git ca-certificates

# vLLM v0.31.0 + Qwen3.8-Flash-Next, MiniCPM-V 4.7, Transformers 5.18 and
# compiled GDN patches.
FROM source-base AS vllm-source
ARG VLLM_VERSION
COPY ./patches/v0.31.0/0001-vllm-qwen38.patch ./patches/v0.31.0/0003-vllm-minicpmv47.patch \
     ./patches/v0.31.0/0005-vllm-runtime-compat.patch ./patches/v0.31.0/0008-vllm-gdn-compile.patch \
     /tmp/patches/
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    git clone --depth 1 -b "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git /src/vllm && \
    cd /src/vllm && \
    for patch in 0001-vllm-qwen38 0003-vllm-minicpmv47 0005-vllm-runtime-compat \
                 0008-vllm-gdn-compile; do \
        git apply --check "/tmp/patches/${patch}.patch" && \
        git apply "/tmp/patches/${patch}.patch" || exit 1; \
    done && \
    git diff --check

# vllm-xpu-kernels (pinned commit + Qwen3.8 and MiniCPM-V 4.7 kernels).
FROM source-base AS xpu-kernels-source
ARG VLLM_XPU_KERNELS_COMMIT
COPY ./patches/v0.31.0/0002-kernels-qwen38.patch ./patches/v0.31.0/0004-kernels-minicpmv47.patch \
     /tmp/patches/
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git /src/vllm-xpu-kernels && \
    cd /src/vllm-xpu-kernels && \
    git checkout ${VLLM_XPU_KERNELS_COMMIT} && \
    for patch in 0002-kernels-qwen38 0004-kernels-minicpmv47; do \
        git apply --check "/tmp/patches/${patch}.patch" && \
        git apply "/tmp/patches/${patch}.patch" || exit 1; \
    done && \
    git diff --check && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i '/^torch==/s/^/# /' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt

# =========================================================================
# build-base -- oneAPI toolchain + torch 2.14.0+xpu build venv
# =========================================================================
FROM python-base AS build-base

ENV VIRTUAL_ENV=/opt/buildenv \
    PATH="/root/.local/bin:/opt/buildenv/bin:${PATH}" \
    VLLM_TARGET_DEVICE=xpu

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

RUN mkdir -p /out /build-logs

# =========================================================================
# Native wheel builds (each writes one wheel to /out)
# =========================================================================
FROM build-base AS vllm-build
COPY --from=vllm-source /src/vllm /src/vllm
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind
COPY --from=rust-build /workspace/vllm/vllm/vllm-rs /src/vllm/vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/vllm/_rust_*.so /src/vllm/vllm/
RUN set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    (VLLM_VERSION_OVERRIDE=0.31.0+xpu.llmscaler \
        pip wheel --verbose --no-build-isolation --no-deps . -w /out \
        > /build-logs/vllm.log 2>&1 || \
        { tail -200 /build-logs/vllm.log; exit 1; }) && \
    tail -40 /build-logs/vllm.log

# Xe2 (BMG) and default targets with the Qwen3.8 kernels; Xe3P is not built.
FROM build-base AS xpu-kernels-build
COPY --from=xpu-kernels-source /src/vllm-xpu-kernels /src/vllm-xpu-kernels
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    source /opt/intel/oneapi/setvars.sh --force && \
    cd /src/vllm-xpu-kernels && \
    uv pip install -r requirements.txt && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH:-}" && \
    (MAX_JOBS=16 VLLM_XPU_ENABLE_XE2=ON VLLM_XPU_ENABLE_XE3P=OFF \
        VLLM_XPU_ENABLE_XE_DEFAULT=ON QWEN38_KERNELS_ENABLED=ON \
        VLLM_VERSION_OVERRIDE=0.1.15.4+llmscaler.1 \
        pip wheel --verbose --no-build-isolation --no-deps . -w /out \
        > /build-logs/vllm-xpu-kernels.log 2>&1 || \
        { tail -200 /build-logs/vllm-xpu-kernels.log; exit 1; }) && \
    tail -40 /build-logs/vllm-xpu-kernels.log

# =========================================================================
# Wheel export targets
# =========================================================================
FROM scratch AS vllm-wheel
COPY --from=vllm-build /out/*.whl /

FROM scratch AS vllm-xpu-kernels-wheel
COPY --from=xpu-kernels-build /out/*.whl /

# Inputs that determine the wheel set; recorded in, and checked against, manifest.json.
FROM python-base AS build-inputs
ARG VLLM_VERSION
ARG VLLM_XPU_KERNELS_COMMIT
COPY ./docker/wheel_manifest.py ./docker/runtime-requirements.txt /opt/build-inputs/
COPY ./patches/v0.31.0/*.patch /opt/build-inputs/patches/
RUN cd /opt/build-inputs && \
    VLLM_VERSION=${VLLM_VERSION} VLLM_XPU_KERNELS_COMMIT=${VLLM_XPU_KERNELS_COMMIT} \
    PYTHON_VERSION=${PYTHON_VERSION} \
    uv run --no-project --python "${PYTHON_VERSION}" wheel_manifest.py inputs \
        $(printf -- '--patch %s ' patches/*.patch) \
        --lock runtime-requirements.txt \
        --constraints /opt/runtime-xpu.constraints.txt > /build-inputs.json

FROM python-base AS wheels-manifest
COPY --from=vllm-build /out/*.whl /wheels/
COPY --from=xpu-kernels-build /out/*.whl /wheels/
COPY --from=build-inputs /build-inputs.json /opt/build-inputs/wheel_manifest.py /opt/build-inputs/
RUN uv run --no-project --python "${PYTHON_VERSION}" /opt/build-inputs/wheel_manifest.py create \
        --inputs /opt/build-inputs/build-inputs.json --wheels /wheels \
        --output /wheels/manifest.json

# All native wheels, their manifest and the runtime lock they were built for.
FROM scratch AS wheels
COPY --from=wheels-manifest /wheels/ /
COPY ./docker/runtime-requirements.txt /

# =========================================================================
# runtime-lock -- resolve docker/runtime-requirements.txt (see
# docker/compile_runtime_requirements.sh); not used by image builds
# =========================================================================
FROM python-base AS runtime-lock
ARG LOCK_COMPILE_ARGS=
WORKDIR /lock
COPY ./docker/runtime_requirements.in ./docker/runtime-requirements.txt /lock/
COPY --from=vllm-source /src/vllm/requirements /lock/vllm-requirements
RUN --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    cp /opt/intel-runtime.overrides.txt /lock/ && \
    sed -i '/^vllm_xpu_kernels[ =]/d' vllm-requirements/xpu.txt && \
    uv pip compile runtime_requirements.in \
        --python "${PYTHON_VERSION}" --only-binary :all: --generate-hashes \
        --custom-compile-command 'bash docker/compile_runtime_requirements.sh' \
        --output-file runtime-requirements.txt ${LOCK_COMPILE_ARGS}

FROM scratch AS runtime-lock-export
COPY --from=runtime-lock /lock/runtime-requirements.txt /

# =========================================================================
# runtime -- production wheel install on OMIX with torch 2.14
# =========================================================================
FROM python-base AS runtime

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    LD_LIBRARY_PATH="/opt/venv/lib:/usr/local/lib"

WORKDIR /llm-scaler/vllm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        ffmpeg libsndfile1 libsm6 libxext6 libgl1 numactl curl && \
    rm -rf /var/lib/apt/lists/* && \
    uv venv --python "${PYTHON_VERSION}" ${VIRTUAL_ENV}

# --- patched source trees (vLLM and vllm-xpu-kernels) kept in the shipped
#     image for reference / editable-install dev workflows. Copied from the
#     source stages, so reusing existing wheels compiles nothing. ---
COPY --from=vllm-source /src/vllm ./vllm
COPY --from=xpu-kernels-source /src/vllm-xpu-kernels ./vllm-xpu-kernels
COPY ./docker/check_runtime_dependencies.py ./docker/wheel_manifest.py /opt/runtime-checks/
COPY --from=build-inputs /build-inputs.json /opt/runtime-checks/
COPY ./docker/runtime-requirements.txt /opt/
# Later installs into the image keep the locked runtime versions.
RUN sed -n -E 's/^([A-Za-z0-9_.-]+[=<>!~]+[^ ]+).*/\1/p' \
        /opt/runtime-xpu.constraints.txt /opt/runtime-requirements.txt \
        > /opt/xpu.constraints.txt

# --- locked dependencies, the native wheels, validation and the Triton
#     shrink all in ONE layer: cleanup must happen in the same RUN that
#     creates the bloat, otherwise a later layer can't reclaim the space.
#     Wheels are bind-mounted (not COPY'd): a COPY would bake them into their
#     own image layer that a later `rm` could never reclaim. The uv cache is a
#     cache mount for the same reason (UV_LINK_MODE=copy keeps the venv
#     independent of it). Wheels cannot carry symlinks, so identical copies of
#     versioned shared libraries are relinked after installation. ---
RUN --mount=type=bind,from=wheels,target=/wheels \
    --mount=type=bind,source=docker/dedupe_shared_libs.py,target=/tmp/dedupe_shared_libs.py \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=build-ca,target=/etc/ssl/certs/ca-certificates.crt \
    set -eo pipefail && \
    python /opt/runtime-checks/wheel_manifest.py verify \
        --inputs /opt/runtime-checks/build-inputs.json --wheels /wheels && \
    env -u UV_CONSTRAINT -u UV_OVERRIDE \
        uv pip sync --require-hashes --only-binary :all: /opt/runtime-requirements.txt && \
    uv pip install --no-deps --only-binary :all: \
        /wheels/vllm-*.whl \
        /wheels/vllm_xpu_kernels-*.whl && \
    python /tmp/dedupe_shared_libs.py ${VIRTUAL_ENV} && \
    python -c 'import vllm_xpu_kernels._C, vllm_xpu_kernels._moe_C, vllm_xpu_kernels._qwen38_C; print(vllm_xpu_kernels._qwen38_C.__file__)' && \
    python -c 'import importlib.metadata as m; assert m.version("vllm") == "0.31.0+xpu.llmscaler"; assert m.version("vllm-xpu-kernels") == "0.1.15.4+llmscaler.1"; assert m.version("torch") == "2.14.0+xpu"; assert m.version("triton") == "3.8.0+xpu"; assert m.version("triton-xpu") == "3.8.0"; assert m.version("transformers") == "5.18.0"; assert m.version("oneccl") == m.version("oneccl-devel") == "2022.1.2"' && \
    python /opt/runtime-checks/check_runtime_dependencies.py \
        --constraints /opt/runtime-xpu.constraints.txt \
        --constraints /opt/runtime-requirements.txt && \
    TRITON=/opt/venv/lib/python3.12/site-packages/triton && \
    strip --strip-debug ${TRITON}/_C/libtriton.so

# Keep wheel runtime libraries ahead of the matching OMIX system installation.
ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && export LD_LIBRARY_PATH=\"/opt/venv/lib:${LD_LIBRARY_PATH:-}\" && exec vllm serve \"$@\"", "--"]
