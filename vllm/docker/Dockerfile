# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:26.13.7.1 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics

RUN apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libaio-dev \
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev \
    # Install Intel GPU runtime packages
        intel-oneapi-dpcpp-ct=2025.2.0-517 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Configure environment to source oneAPI
# RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
#     echo "source /opt/intel/oneapi/ccl/2021.15.7-down.1/env/vars.sh --force" >> /root/.bashrc
# RUN   echo "source /opt/intel/oneapi/ccl/2021.15.7-down.1/env/vars.sh --force" >> /root/.bashrc

WORKDIR /llm
COPY ./patches/vllm_for_multi_arc.patch /tmp/
COPY ./patches/fix_registry_subprocess_env.patch /tmp/
# COPY ./patches/miner-u.patch /tmp/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# suppress the python externally managed environment error
RUN python3 -m pip config set global.break-system-packages true

# Clone + patch vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone -b v0.14.0 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    git apply /tmp/fix_registry_subprocess_env.patch && \
    pip install -r requirements/xpu.txt && \
    pip install arctic-inference==0.1.1 && \
    # pip install --no-cache-dir arctic-inference==0.1.1 && \
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.2/include/:${CPATH} && \
    MAX_JOBS=2 pip install --no-build-isolation .

# Clone + patch miner-U
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone -b release-2.6.2 https://github.com/opendatalab/MinerU.git && \
    cd MinerU && \
    pip install -e .[core] && \
    # pip install mineru_vl_utils==0.1.14 gradio gradio-client gradio-pdf && \
    sed -i 's/kwargs.get("max_concurrency", 100)/kwargs.get("max_concurrency", 200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py && \
    sed -i 's/kwargs.get("http_timeout", 600)/kwargs.get("http_timeout", 1200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py


# Install pypi dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install bigdl-core==2.4.0b2

# Cleanup patch file
RUN rm -rf /tmp/*



SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

COPY ./examples/offline_inference.py /llm/

ARG http_proxy
ARG https_proxy

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'


# Pin transformers version to avoid conflict in vLLM
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install librosa soundfile decord && \
    pip install git+https://github.com/huggingface/transformers.git && \
    pip install ijson

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /llm && \
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git && \
    cd vllm-xpu-kernels && \
    git checkout 4c83144 && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i 's|^torch==2.10.0+xpu|# torch==2.10.0+xpu|' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt && \
    pip install -r requirements.txt && \
    pip install --no-build-isolation .

# FIX triton
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall triton triton-xpu -y && \
    pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu


# Set additional environment for production usage
ENV VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"
ENV VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


# install development dependencies (for testing)
RUN cd /llm/vllm && \
    python3 -m pip install -e tests/vllm_test_utils
    
RUN pip uninstall oneccl oneccl-devel -y

RUN rm /usr/lib/python3/dist-packages/PyJWT-2.7.0.dist-info/ -rf

RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc

ENTRYPOINT ["bash", "-c", "vllm serve"]

