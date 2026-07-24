# Dockerfile for SGLang on Intel BMG (Battlemage)

# ======== Build Stage ========
# Pinned to 0.1.0 instead of latest (same digest as latest at time of writing,
# but avoids the tag silently drifting under us on rebuilds).
FROM intel/omix:0.1.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1,::1
# AOT-compile ESIMD kernels for BMG only (skip multi-arch fat binaries
# that would pull in dpas v128 paths unsupported on XeLPG).
ARG DPCPP_SYCL_TARGET=bmg
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy}
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}
ENV OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET}
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV MAX_JOBS=1

# Pinned to upstream (sgl-project) releases/commits instead of the
# analytics-zoo dev-bmg forks. All BMG-specific changes are carried as
# patches under ./patches/ (see "Build sgl-kernel-xpu wheel" and "Build
# sglang wheel" below) and applied with `git apply` after checkout, so the
# exact upstream base is pinned and reproducible.
ARG SGLANG_UPSTREAM_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_UPSTREAM_TAG=v0.5.13
ARG SGL_KERNEL_UPSTREAM_REPO=https://github.com/sgl-project/sgl-kernel-xpu.git
# sgl-kernel-xpu has no release tags; pin to the exact upstream/main commit
# the analytics-zoo/dev-bmg fork was branched from.
ARG SGL_KERNEL_UPSTREAM_COMMIT=ea5c70f0909bcd55ceaf1803302651fc0593b64d

WORKDIR /llm-scaler/sglang
COPY . .

# --- System packages + pip/uv base setup ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip python3-dev git ninja-build \
 && python3 -m pip config set global.break-system-packages true \
 && pip3 install uv \
 && uv pip install --system ninja \
 && uv pip install --system torch==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build sgl-kernel-xpu wheel ---
# Pinned upstream commit + our BMG patch (mha_fwd init, gdn_attention,
# rms_norm_gated bf16/fp16, fp8 per-tensor quant graph-capture fix, ...),
# applied on top instead of pulling the analytics-zoo/dev-bmg fork directly.
RUN git clone ${SGL_KERNEL_UPSTREAM_REPO} /root/sk \
 && cd /root/sk \
 && git checkout ${SGL_KERNEL_UPSTREAM_COMMIT} \
 && git apply --whitespace=nowarn /llm-scaler/sglang/patches/sgl_kernel_xpu.patch \
 && pip3 wheel -v /root/sk --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET} --wheel-dir /root/wheels --no-deps \
 && rm -rf /root/sk \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build sglang wheel ---
# This is intentionally the last mutable layer before the wheel build. The
# build script passes a cache-bust arg for this step only so the rest of the
# image can stay cached while the pinned base + patch are refreshed on each
# build.
# Patch carries ESIMD fast-paths (decode attn, MoE fp8, fp8 GEMM,
# QKV split-norm-rope, GDN conv fused_seq, RMSNormGated SYCL, prefill DPAS),
# fp8 e5m2 weight dtype, XPU Graph scaffold + Qwen3.5/3.6 capture wiring —
# applied on top of the pinned upstream release instead of pulling the
# analytics-zoo/dev-bmg fork directly.
ARG SGLANG_CACHEBUST=0
RUN echo "Refreshing sglang ${SGLANG_UPSTREAM_TAG} + BMG patch (cache bust ${SGLANG_CACHEBUST})" && \
    git clone --depth 1 -b ${SGLANG_UPSTREAM_TAG} ${SGLANG_UPSTREAM_REPO} /root/sglang \
 && cd /root/sglang \
 && git apply --whitespace=nowarn /llm-scaler/sglang/patches/sglang_for_multi_arc.patch \
 && cd /root/sglang/python \
 && cp pyproject_xpu.toml pyproject.toml \
 && sed -i '/sgl-kernel/d;/torch==/d;/torchaudio==/d;/torchvision==/d;/torchao==/d' pyproject.toml \
 && pip3 wheel -v . --wheel-dir /root/wheels --no-deps \
 && rm -rf /root/sglang \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build custom ESIMD kernels wheel ---
# Single merged package `custom_esimd_kernels_sglang` — folds in what used to
# be two separate flavours (custom-esimd-kernels + custom-esimd-kernels-sglang):
#   * decode SDPA, FP8 GEMM, FP8 MoE silu, FP8 MoE prefill, prefill DPAS (was v2)
#   * Qwen3.5/3.6 fused QKV, GDN conv fused_seq, RMSNormGated (was -sglang)
RUN . /opt/intel/oneapi/setvars.sh --force \
 && cd /llm-scaler/sglang/custom-esimd-kernels \
 && CXX=icpx pip3 wheel -v . --wheel-dir /root/wheels --no-deps --no-build-isolation \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# ======== Runtime Stage ========
FROM intel/pytorch:xpu-2.11.0-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1,::1,intel.com,.intel.com
ARG DPCPP_SYCL_TARGET=bmg
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy}
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}
ENV OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET}

WORKDIR /llm-scaler/sglang
COPY . .

# Single RUN: install UMD driver (BMG) + all pre-built wheels via bind-mounted /root/wheels
# (no COPY layer for wheels -> ~400 MB saved), then strip + prune caches/bytecode.
RUN --mount=type=bind,from=builder,source=/root/wheels,target=/tmp/wheels \
    export no_proxy=localhost,127.0.0.1,::1 \
 # --- Intel BMG user-mode driver (pytorch base does not bundle it) ---
 && apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common python3-pip \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc \
        libze-dev \
 # --- Python deps: pre-built wheels + explicit runtime deps sglang needs ---
 && python3 -m pip config set global.break-system-packages true \
 && pip3 install uv \
 && uv pip install --python /opt/venv/bin/python \
        /tmp/wheels/*.whl \
        --extra-index-url https://download.pytorch.org/whl/xpu \
 && uv pip install --python /opt/venv/bin/python \
        --no-deps xgrammar==0.1.33 || true \
 # --- runtime helpers (eval scripts: gsm8k, etc.) ---
 && uv pip install --python /opt/venv/bin/python \
        pytest expecttest ray huggingface_hub \
 && chmod +x /llm-scaler/sglang/scripts/*.sh \
 # --- Size cleanup: drop caches + bytecode ---
 && find /opt/venv /usr/local/lib/python3* -name "*.pyc" -delete 2>/dev/null || true \
 && find /opt/venv /usr/local/lib/python3* -name "__pycache__" -type d \
        -exec rm -rf {} + 2>/dev/null || true \
 && find /tmp -mindepth 1 -maxdepth 1 ! -name wheels -exec rm -rf {} + \
 && rm -rf /root/.cache /root/.cmake /var/tmp/* /var/lib/apt/lists/*

WORKDIR /llm-scaler/sglang
ENTRYPOINT ["bash", "-c", "exec sglang serve \"$@\"", "--"]
