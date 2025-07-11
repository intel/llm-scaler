# llm-scaler-vllm

llm-scaler-vllm is an extended and optimized version of vLLM, specifically adapted for Intel’s Multi-BMG platform. This project enhances vLLM’s core architecture with Intel-specific performance optimizations, advanced features, and tailored support for customer use cases.

---

## Table of Contents

1. [Getting Started and Usage](#1-getting-started-and-usage)  
   1.1 [Install Native Environment](#11-install-native-environment)  
   1.2 [Pulling and Running the Docker Container](#11-pulling-and-running-the-docker-container)  
   1.3 [Launching the Serving Service](#12-launching-the-serving-service)  
   1.4 [Benchmarking the Service](#13-benchmarking-the-service)  
2. [Advanced Features](#2-advanced-features)  
   2.1 [CCL Support (both P2P & USM)](#21-ccl-support-both-p2p--usm)  
   2.2 [INT4 and FP8 Quantized Online Serving](#22-int4-and-fp8-quantized-online-serving)  
   2.3 [Embedding and Reranker Model Support](#23-embedding-and-reranker-model-support)  
   2.4 [Multi-Modal Model Support](#24-multi-modal-model-support)  
   2.5 [Data Parallelism (DP)](#25-data-parallelism-dp)  
   2.6 [Maximum Context Length Support](#26-maximum-context-length-support)  
3. [Supported Models](#3-supported-models)  
4. [Troubleshooting](#4-troubleshooting)  

---

## 1. Getting Started and Usage

### 1.1 Install Native Environment

#### 1.1.1 Execute the Script

First, install a standard Ubuntu 25.04
- [Ubuntu 25.04 Desktop](https://releases.ubuntu.com/25.04/ubuntu-25.04-desktop-amd64.iso) (for Xeon-W)
- [Ubuntu 25.04 Server](https://releases.ubuntu.com/25.04/ubuntu-25.04-live-server-amd64.iso) (for Xeon-SP).

Then, update the proxy configuration in [native_bkc_setup.sh](https://github.com/intel/llm-scaler/blob/main/vllm/tools/native_bkc_setup.sh). 

```bash
export https_proxy=http://your-proxy.com:port
export http_proxy=http://your-proxy.com:port
export no_proxy=127.0.0.1,*.intel.com
````
Make sure your system has internet access and also the proxy can access [Ubuntu intel-graphics PPA](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics) since native_bkc_setup.sh will get packages from there.

Switch to root user, run this script.

```bash
sudo su -
cd vllm/tools/
chmod +x native_bkc_setup.sh
./native_bkc_setup.sh
```` 

If everything is ok, you can see below installation completion message. Depending on your network speed, the execution may require 30 mins or longer time. 

```bash
Tools and scripts are located at /root/multi-arc.
✅ [DONE] Environment setup complete. Please reboot your system to apply changes.
````

#### 1.1.2 Check the Installation

Since it update the GPU firmware and initramfs, you need to reboot to make the changes taking effect. 

After reboot, you can use lscpi and sycl-ls to check if all the drivers are installed correctly.

For Intel B60 GPU, it's device ID is e211, you can grep this ID in lspci to make sure the KMD (kernel mode driver) workable. 

```bash
root@edgeaihost19:~# lspci -tv | grep -i e211
 |           \-01.0-[17-1a]----00.0-[18-1a]--+-01.0-[19]----00.0  Intel Corporation Device e211
 |           \-05.0-[4e-51]----00.0-[4f-51]--+-01.0-[50]----00.0  Intel Corporation Device e211
 |           \-01.0-[85-88]----00.0-[86-88]--+-01.0-[87]----00.0  Intel Corporation Device e211
 |           \-01.0-[bc-bf]----00.0-[bd-bf]--+-01.0-[be]----00.0  Intel Corporation Device e211
````

If you also see e211 device recognized by sycl-ls, then GPGPU UMD (user mode driver) working properly.

```bash
root@edgeaihost19:~# source /opt/intel/oneapi/setvars.sh

:: initializing oneAPI environment ...
   -bash: BASH_VERSION = 5.2.37(1)-release
:: advisor -- latest
:: ccl -- latest
:: compiler -- latest
:: dal -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: dnnl -- latest
:: dpcpp-ct -- latest
:: dpl -- latest
:: ipp -- latest
:: ippcp -- latest
:: mkl -- latest
:: mpi -- latest
:: pti -- latest
:: tbb -- latest
:: umf -- latest
:: vtune -- latest
:: oneAPI environment initialized ::

root@edgeaihost19:~# sycl-ls
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Graphics [0xe211] 20.1.0 [1.6.33944+12]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Graphics [0xe211] 20.1.0 [1.6.33944+12]
[level_zero:gpu][level_zero:2] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Graphics [0xe211] 20.1.0 [1.6.33944+12]
[level_zero:gpu][level_zero:3] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Graphics [0xe211] 20.1.0 [1.6.33944+12]
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) w5-2545 OpenCL 3.0 (Build 0) [2025.20.6.0.04_224945]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Graphics [0xe211] OpenCL 3.0 NEO  [25.22.33944]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Graphics [0xe211] OpenCL 3.0 NEO  [25.22.33944]
[opencl:gpu][opencl:3] Intel(R) OpenCL Graphics, Intel(R) Graphics [0xe211] OpenCL 3.0 NEO  [25.22.33944]
[opencl:gpu][opencl:4] Intel(R) OpenCL Graphics, Intel(R) Graphics [0xe211] OpenCL 3.0 NEO  [25.22.33944]
````

#### 1.1.3 Using Tools

This script installs 2 tools for GPU development: 

- level-zero-tests: GPU P2P & memory bandwidth benchmark
- xpu-smi: GPU profiling and management

level-zero-tests is located /root/multi-arc/level-zero-tests. It's already built, you can directly run the binary in
- level-zero-tests/build/perf_tests/ze_peer/ze_peer for P2P benchmark
- level-zero-tests/build/perf_tests/ze_peak/ze_peak for memory bandwidth benchmark including H2D, D2H, D2D

xpu-smi already installed in system, you can directly run.

```bash
xpu-smi
````

Intel also offers 2 other tools which are not publicly available for current Pre-PV release. 
- 1ccl tool for collective communication benchmark
- gemm tool for compute capability benchmark

To get these 2 tools or detailed user guide for all tools, please contact your Intel support team for help.

We also provide a script to set the CPU/GPU to performance mode, you can run it before running the workload

```bash 
cd /root/multi-arc
./setup_perf.sh
````

### 1.2 Pulling and Running the Docker Container

First, pull the image:

```bash
docker pull intel/llm-scaler-vllm:latest
````

Then, run the container:

```bash
sudo docker run -td \
    --privileged \
    --net=host \
    --device=/dev/dri \
    --name=lsv-container \
    -v /home/intel/LLM:/llm/models/ \
    -e no_proxy=localhost,127.0.0.1 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    intel/llm-scaler-vllm:latest
```

Enter the container:

```bash
docker exec -it lsv-container bash
```

---

### 1.3 Launching the Serving Service

```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=1 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    --quantization fp8 \
    -tp=1
```

---

### 1.4 Benchmarking the Service

```bash
python3 /llm/vllm/benchmarks/benchmark_serving.py \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dataset-name random \
    --random-input-len=1024 \
    --random-output-len=512 \
    --ignore-eos \
    --num-prompt 10 \
    --trust_remote_code \
    --request-rate inf \
    --backend vllm \
    --port=8000
```

---

## 2. Advanced Features

### 2.1 CCL Support (both P2P & USM)

The image includes OneCCL with automatic fallback between P2P and USM memory exchange modes.

* To manually switch modes, use:

```bash
export CCL_TOPO_P2P_ACCESS=1  # P2P mode
export CCL_TOPO_P2P_ACCESS=0  # USM mode
```

* Performance notes:

  * Small batch sizes show minimal difference.
  * Large batch sizes (e.g., batch=30) typically see around 15% higher throughput with P2P mode compared to USM.

---

### 2.2 INT4 and FP8 Quantized Online Serving
To enable online quantization using `llm-scaler-vllm`, specify the desired quantization method with the `--quantization` option when starting the service.

The following example shows how to launch the server with `sym_int4` quantization:

```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=1 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    --quantization sym_int4 \
    -tp=1
```

To use fp8 quantization, simply replace `--quantization sym_int4` with:

```bash
--quantization fp8
```
---

### 2.3 Embedding and Reranker Model Support

#### Start service using V0 engine
```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=0 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/bge-reranker-large \
    --served-model-name bge-reranker-large \
    --task embed \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=2048 \
    --disable-log-requests \
    --max-model-len=2048 \
    --block-size 16 \
    --quantization fp8 \
    -tp=1
```

After starting the vLLM service, you can follow these two links to use it.
#### [Rerank api](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#re-rank-api)

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "bge-reranker-large",
  "query": "What is the capital of France?",
  "documents": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Horses and cows are both animals.",
    "The French have a rich tradition in engineering."
  ]
}'
```

#### [Embedding api](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api_1)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["需要嵌入文本1","这是第二个句子"],
    "model": "bge-m3",
    "encoding_format": "float"
  }'
```
---

### 2.4 Multi-Modal Model Support

#### Start service using V1 engine
```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=1 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/Qwen2.5-VL-7B-Instruct \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --allowed-local-media-path /llm/models/test \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=5120 \
    --disable-log-requests \
    --max-model-len=5120 \
    --block-size 16 \
    --quantization fp8 \
    -tp=1
```

After starting the vLLM service, you can follow this link to use it

#### [Multimodal input](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#online-serving)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
```
---

### 2.5 Data Parallelism (DP)

Supports data parallelism on Intel XPU with near-linear scaling.

Example throughput measurements with Qwen-7B model, tensor parallelism (tp) = 1:

| DP Setting | Batch Size | Throughput Ratio |
| ---------- | ---------- | ---------------- |
| 1          | 10         | 1x               |
| 2          | 20         | 1.9x             |
| 4          | 40         | 3.58x            |

To enable data parallelism, add:

```bash
--dp 2
```

---

### 2.6 Maximum Context Length Support
When using the `V1` engine, the system automatically logs the maximum supported context length during startup based on the available GPU memory and KV cache configuration.

#### Example: Successful Startup

The following log output shows the service successfully started with sufficient memory, and a GPU KV cache size capable of handling up to `114,432` tokens:

```
INFO 07-11 06:18:32 [kv_cache_utils.py:646] GPU KV cache size: 114,432 tokens
INFO 07-11 06:18:32 [kv_cache_utils.py:649] Maximum concurrency for 18,000 tokens per request: 6.36x
```
This indicates that the model can support requests with up to `114,432` tokens per sequence.

To fully utilize this capacity, you can set the following option at startup:
```bash
--max-model-len 114432
```


#### Example: Exceeding Memory Capacity

If the requested context length exceeds the available KV cache memory, the service will fail to start and suggest the `maximum supported value`. For example:


```
ERROR 07-11 06:23:05 [core.py:390] ValueError: To serve at least one request with the models's max seq len (118000), (6.30 GiB KV cache is needed, which is larger than the available KV cache memory (6.11 GiB). Based on the available memory, the estimated maximum model length is 114432. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```
In this case, you should adjust the launch command with:

```bash
--max-model-len 114432
```

---

## 3. Supported Models

| Model Name        | Category         | Notes                          |
|-------------------|------------------|---------------------------------|
|       DeepSeek-R1-0528-Qwen3-8B   |        language model             |                                 |
|       DeepSeek-R1-Distill-1.5B/7B/8B/14B/32B/70B             |         language model         |                                 |
|       Qwen3-8B/14B/32B            |        language model             |                                 |
|       QwQ-32B                     |        language model             |                                 |
|       Ministral-8B                |        language model             |                                 |
|       Llama3.1-8B/Llama3.1-70B    |        language model             |                                 |
|       Baichuan2-7B/13B            |        language model             |                                 |
|       codegeex4-all-9b            |        language model             |                                 |
|       DeepSeek-Coder-33B          |        language model             |                                 |
|       Qwen3-30B-A3B               |        language model             |                                 |
|       Qwen2-VL-7B-Instruct        |        multimodal model           |                                 |
|       MiniCPM-V-2.6               |        multimodal model           |                                 |
|       Qwen2.5-VL 7B/32B/72B       |        multimodal model           | pip install transformers==4.52.4       |
|       UI-TARS-7B-DPO              |        multimodal model           | pip install transformers==4.49.0       |
|       Gemma-3-12B                 |        multimodal model           | only can run bf16 with no quantization |
|       GLM-4V-9B                   |        multimodal model           | only can run with four cards           |
|       Qwen3-Embedding             |        Embedding                  |                                 |
|       bge-large, bge-m3           |        Embedding                  |                                 |
|       Qwen3-Reranker              |        Rerank                     |                                 |
|       bge-reranker-large, bge-reranker-v2-m3 |  Rerank                |                                 |
--- 

## 4. Troubleshooting




