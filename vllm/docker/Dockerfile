# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.50.6.0 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics

# Install dependencies and Python 3.10
RUN apt-get update -y && \
    # apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
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
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev && \
    # Install Intel GPU runtime packages
    apt-get install -y intel-oneapi-dpcpp-ct=2025.2.0-517 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llm
COPY ./patches/vllm_for_multi_arc.patch /tmp/
COPY ./patches/miner-u.patch /tmp/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# suppress the python externally managed environment error
RUN python3 -m pip config set global.break-system-packages true

# Clone + patch vllm
RUN git clone -b v0.10.2 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    pip install --no-cache-dir -r requirements/xpu.txt && \
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.2/include/:${CPATH} && \
    python3 setup.py install

# Clone + patch miner-U
RUN git clone -b release-2.6.2 https://github.com/opendatalab/MinerU.git && \
    cd MinerU && \
    pip install -e .[core] && \
    # pip install mineru_vl_utils==0.1.14 gradio gradio-client gradio-pdf && \
    sed -i 's/kwargs.get("max_concurrency", 100)/kwargs.get("max_concurrency", 200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py && \
    sed -i 's/kwargs.get("http_timeout", 600)/kwargs.get("http_timeout", 1200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py


# Install pypi dependencies
RUN pip install bigdl-core==2.4.0b2

# Cleanup patch file
RUN rm -rf /tmp/*

# Configure environment to source oneAPI
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc

SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

COPY ./examples/offline_inference.py /llm/

ARG http_proxy
ARG https_proxy

# install additional dependencies for openai api server
RUN pip install accelerate hf_transfer 'modelscope!=1.15.0'


# Pin transformers version to avoid conflict in vLLM
RUN pip install "transformers==4.57.0" && \
    pip install librosa soundfile decord


# Set additional environment for production usage
ENV VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"
ENV VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ENV UR_L0_USE_IMMEDIATE_COMMANDLISTS=0

RUN pip uninstall oneccl oneccl-devel -y

# install development dependencies (for testing)
RUN cd /llm/vllm && \
    python3 -m pip install -e tests/vllm_test_utils
    
ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && python3 -m vllm.entrypoints.openai.api_server"]

