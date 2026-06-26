# Dockerfile for SGLang on Intel BMG (Battlemage)
# Targets Qwen3.6-35B-A3B online fp8 inference.
#
# Build:
#   docker build -t llm-scaler-sgl:bmg \
#     --build-arg http_proxy=$http_proxy \
#     --build-arg https_proxy=$https_proxy \
#     llm-scaler/sgl
#
# Run (TP=4 example):
#   docker run --rm -it --device=/dev/dri --shm-size=16g \
#     -v /path/to/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro \
#     -p 30000:30000 \
#     llm-scaler-sgl:bmg \
#     /workspace/scripts/run_qwen3_6.sh

FROM amr-registry.caas.intel.com/intelanalytics/llm-scaler-platform:26.24.9.1

ENV DEBIAN_FRONTEND=noninteractive

ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1,::1
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# AOT-compile ESIMD kernels for BMG only (skip multi-arch fat binaries
# that would pull in dpas v128 paths unsupported on XeLPG).
ARG DPCPP_SYCL_TARGET=bmg
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}

ARG SGLANG_REPO=https://github.com/analytics-zoo/sglang.git
ARG SGLANG_BRANCH=dev-bmg
ARG SGL_KERNEL_REPO=https://github.com/analytics-zoo/sgl-kernel-xpu.git
ARG SGL_KERNEL_BRANCH=dev-bmg

# ----- system + python -----
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip config set global.break-system-packages true

RUN pip3 install torch==2.12.0+xpu torchvision==0.27.0+xpu torchaudio==2.11.0+xpu \
      triton-xpu==3.7.1 \
      --index-url https://download.pytorch.org/whl/xpu

# ----- sgl-kernel-xpu -----
# dev-bmg already carries the BMG fixes (mha_fwd init, gdn_attention,
# rms_norm_gated bf16/fp16, fp8 per-tensor quant graph-capture fix, ...).
RUN git clone -b ${SGL_KERNEL_BRANCH} ${SGL_KERNEL_REPO} /root/sgl-kernel-xpu && \
    cd /root/sgl-kernel-xpu && \
    pip3 install -v . \
        --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET} && \
    rm -rf /root/sgl-kernel-xpu

# ----- sglang (dev-bmg) -----
# Carries ESIMD fast-paths (decode attn, MoE fp8, fp8 GEMM, QKV split-norm-rope,
# GDN conv fused_seq, RMSNormGated SYCL), fp8 e5m2 weight dtype, XPU Graph
# scaffold + Qwen3.5/3.6 capture wiring.
RUN git clone -b ${SGLANG_BRANCH} ${SGLANG_REPO} /root/sglang && \
    cd /root/sglang/python && \
    cp pyproject_xpu.toml pyproject.toml && \
    sed -i '/sgl-kernel/d' pyproject.toml && \
    sed -i '/torch==/d' pyproject.toml && \
    sed -i '/torchaudio==/d' pyproject.toml && \
    sed -i '/torchvision==/d' pyproject.toml && \
    sed -i '/torchao==/d' pyproject.toml && \
    pip3 install . --extra-index-url https://download.pytorch.org/whl/xpu && \
    pip3 install --no-deps xgrammar==0.1.33 || true && \
    rm -rf /root/sglang

# ----- ESIMD kernel packages -----
# Two flavours, both required:
#   * custom-esimd-kernels        — sglang decode attn, fp8 GEMM, fp8 MoE silu
#   * custom-esimd-kernels-sglang — Qwen3.5/3.6 fused QKV, GDN conv fused_seq,
#                                   RMSNormGated ESIMD variant
COPY custom-esimd-kernels        /workspace/custom-esimd-kernels
COPY custom-esimd-kernels-sglang /workspace/custom-esimd-kernels-sglang

# Build is single-threaded on purpose: the three SyclExtension targets share
# one build.ninja under build/temp.../ and parallel writes corrupt it.
RUN cd /workspace/custom-esimd-kernels && \
    CXX=icpx python3 setup.py build_ext --inplace -j 1 && \
    pip3 install -e . --no-build-isolation

RUN cd /workspace/custom-esimd-kernels-sglang && \
    CXX=icpx python3 setup.py build_ext --inplace -j 1 && \
    pip3 install -e . --no-build-isolation

# ----- runtime helpers -----
RUN pip3 install pytest expecttest ray huggingface_hub

COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/*.sh

WORKDIR /workspace
CMD ["bash"]
