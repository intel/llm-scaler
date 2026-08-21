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

# Pinned to upstream (sgl-project) releases/commits. The patches under
# ./patches are snapshots of the merged analytics-zoo dev-bmg branches:
# sglang@3a7753f4b and sgl-kernel-xpu@fd89d0f.
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
# Pinned upstream commit + merged dev-bmg patch, including the BMG,
# Gemma4, and KVCacheIO changes.
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
# Patch carries the merged dev-bmg BMG, Qwen3.6, and Gemma4 changes on top of
# the pinned upstream release.
ARG SGLANG_CACHEBUST=0
RUN echo "Refreshing sglang ${SGLANG_UPSTREAM_TAG} + merged dev-bmg patch (cache bust ${SGLANG_CACHEBUST})" && \
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
#   * GQA2/split-K attention, FP16 GEMV, fused norms, and KV scatter
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
ARG ONECCL_INSTALLER="intel-oneccl-2021.15.9.14_offline.sh"
ARG ONECCL_INSTALLER_SHA256="f7ab81b6ed1b10dd35fadec366a78046d8af214888dfd625047ce8953d5aa4ef"
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy}
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}
ENV OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET}

WORKDIR /llm-scaler/sglang
COPY . .
COPY --from=builder /opt/intel/oneapi/2025.3/lib/libdnnl.so* /opt/intel/oneapi/2025.3/lib/

# Single RUN: install UMD driver (BMG) + all pre-built wheels via bind-mounted /root/wheels
# (no COPY layer for wheels -> ~400 MB saved), then strip + prune caches/bytecode.
RUN --mount=type=bind,from=builder,source=/root/wheels,target=/tmp/wheels \
    export no_proxy=localhost,127.0.0.1,::1 \
 # --- Intel BMG user-mode driver (pytorch base does not bundle it) ---
 && apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common python3-pip curl \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
       # Keep the production GPU runtime aligned with intel/omix:0.1.0-devel-ubuntu24.04. \
       # Pin direct packages and version-flexible driver dependencies so a later PPA \
       # update cannot silently change the Level Zero/OpenCL runtime. \
        libze-intel-gpu1=26.14.37833.4-1~24.04~ppa1 \
        libze1=1.28.2-1~24.04~ppa1 \
        libze-dev=1.28.2-1~24.04~ppa1 \
        libze-intel-gpu-raytracing=1.2.2-1~24.04~ppa1 \
        intel-opencl-icd=26.14.37833.4-1~24.04~ppa1 \
        intel-metrics-discovery=1.14.183-1~24.04~ppa1 \
        intel-gsc=0.9.5-1~24.04~ppa2 \
        libigdgmm12=22.10.0-1~24.04~ppa1 \
        libigc2=2.32.7-1256~24.04 \
        libigdfcl2=2.32.7-1256~24.04 \
        libigsc0=0.9.5-1~24.04~ppa2 \
        libmetee5=5.0.0-1~24.04~ppa2 \
        ocl-icd-libopencl1=2.3.2-1build1 \
        clinfo \
 # --- oneCCL 2021.15.9 runtime required by the BMG TP=2 path ---
 && curl -fL --retry 3 -o "${ONECCL_INSTALLER}" \
        "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.9/${ONECCL_INSTALLER}" \
 && printf "%s  %s\n" "${ONECCL_INSTALLER_SHA256}" "${ONECCL_INSTALLER}" \
        > /tmp/oneccl.sha256 \
 && sha256sum -c /tmp/oneccl.sha256 \
 && rm -f /tmp/oneccl.sha256 \
 && bash "${ONECCL_INSTALLER}" -s -x -f /tmp \
 && ONECCL_CUP="/tmp/${ONECCL_INSTALLER%.sh}/packages/intel.oneapi.lin.ccl.runtime,v=2021.15.9+13/cupPayload.cup" \
 && ONECCL_DST="/tmp/oneccl-runtime" \
 && python3 -m zipfile -e "${ONECCL_CUP}" "${ONECCL_DST}" \
 && ONECCL_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib" \
 && rm -f "${ONECCL_LIB}/libccl.so" "${ONECCL_LIB}/libccl.so.1" \
        "${ONECCL_LIB}/libccl_openmp.so" "${ONECCL_LIB}/libccl_openmp.so.0" \
 && ln -sf libccl.so.1 "${ONECCL_LIB}/libccl.so" \
 && ln -sf libccl.so.1.0 "${ONECCL_LIB}/libccl.so.1" \
 && ln -sf libccl_openmp.so.0 "${ONECCL_LIB}/libccl_openmp.so" \
 && ln -sf libccl_openmp.so.0.1 "${ONECCL_LIB}/libccl_openmp.so.0" \
 && ONECCL_CPU_LIB="${ONECCL_DST}/_installdir/ccl/2021.15/lib/ccl/cpu/lib" \
 && rm -f "${ONECCL_CPU_LIB}/libccl.so" "${ONECCL_CPU_LIB}/libccl.so.1" \
        "${ONECCL_CPU_LIB}/libccl_openmp.so" "${ONECCL_CPU_LIB}/libccl_openmp.so.0" \
 && ln -sf libccl.so.1 "${ONECCL_CPU_LIB}/libccl.so" \
 && ln -sf libccl.so.1.0 "${ONECCL_CPU_LIB}/libccl.so.1" \
 && ln -sf libccl_openmp.so.0 "${ONECCL_CPU_LIB}/libccl_openmp.so" \
 && ln -sf libccl_openmp.so.0.1 "${ONECCL_CPU_LIB}/libccl_openmp.so.0" \
 && mkdir -p /opt/intel/oneapi/ccl \
 && rm -rf /opt/intel/oneapi/ccl/2021.15 \
 && cp -a "${ONECCL_DST}/_installdir/ccl/2021.15" /opt/intel/oneapi/ccl/ \
 && rm -f /opt/intel/oneapi/ccl/latest \
 && ln -s /opt/intel/oneapi/ccl/2021.15 /opt/intel/oneapi/ccl/latest \
 && rm -f "${ONECCL_INSTALLER}" \
 && rm -rf "/tmp/${ONECCL_INSTALLER%.sh}" "${ONECCL_DST}" \
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
 # --- Replace the torch-bundled oneCCL in /opt/venv/lib with links to
 #     oneCCL 2021.15 (libtorch_xpu.so resolves libccl.so.1 through its
 #     RPATH into /opt/venv/lib). ---
 && uv pip uninstall --python /opt/venv/bin/python oneccl oneccl-devel || true \
 && rm -f /opt/venv/lib/libccl.so /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 \
        /opt/venv/lib/libccl_openmp.so /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so.0.1 \
 && ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0 /opt/venv/lib/libccl.so.1 \
 && ln -sf /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so \
 && ln -sf /opt/intel/oneapi/ccl/2021.15/lib/libccl_openmp.so.0.1 /opt/venv/lib/libccl_openmp.so.0 \
 && ln -sf /opt/venv/lib/libccl_openmp.so.0 /opt/venv/lib/libccl_openmp.so \
 && chmod +x /llm-scaler/sglang/scripts/*.sh \
 # --- Size cleanup: drop caches + bytecode ---
 && find /opt/venv /usr/local/lib/python3* -name "*.pyc" -delete 2>/dev/null || true \
 && find /opt/venv /usr/local/lib/python3* -name "__pycache__" -type d \
        -exec rm -rf {} + 2>/dev/null || true \
 && find /tmp -mindepth 1 -maxdepth 1 ! -name wheels -exec rm -rf {} + \
 && rm -rf /root/.cache /root/.cmake /var/tmp/* /var/lib/apt/lists/*

WORKDIR /llm-scaler/sglang

# NOTE: no global LD_PRELOAD for oneCCL. The 2021.15 libccl.so.1 carries no
# RPATH/RUNPATH, so preloading it into *every* process (including /bin/bash)
# fails with "libsvml.so: cannot open shared object file" — its oneAPI runtime
# deps only live in /opt/venv/lib, which is not on the loader search path.
# The /opt/venv/lib/libccl.so.1 -> 2021.15 symlink above is what makes torch
# pick up this oneCCL: libtorch_xpu.so has DT_RPATH=$ORIGIN/../../../..
# (= /opt/venv/lib), which is inherited by its transitive dependencies.
ENTRYPOINT ["bash", "-c", "exec sglang serve \"$@\"", "--"]
