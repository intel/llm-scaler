# llm-scaler-vllm

llm-scaler-vllm is an extended and optimized version of vLLM, specifically adapted for Intel’s Multi GPU platform. This project enhances vLLM’s core architecture with Intel-specific performance optimizations, advanced features, and tailored support for customer use cases.

---

## Table of Contents

1. [Getting Started and Usage](#1-getting-started-and-usage)  
   1.1 [Install Bare Metal Environment](#11-install-bare-metal-environment)  
   1.2 [Run Platform Evaluation](#12-run-platform-evaluation)  
   1.3 [Pulling and Running the vllm Docker Container](#13-pulling-and-running-the-vllm-docker-container)  
   1.4 [Launching the Serving Service](#14-launching-the-serving-service)  
   1.5 [Benchmarking the Service](#15-benchmarking-the-service)  
   1.6 [(Optional) Monitoring the Service with Prometheus and Grafana](#16-optional-monitoring-the-service-with-prometheus-and-grafana)
2. [Advanced Features](#2-advanced-features)  
   2.1 [CCL Support (both P2P & USM)](#21-ccl-support-both-p2p--usm)  
   2.2 [INT4 and FP8 Quantized Online Serving](#22-int4-and-fp8-quantized-online-serving)  
   2.3 [Embedding and Reranker Model Support](#23-embedding-and-reranker-model-support)  
   2.4 [Multi-Modal Model Support](#24-multi-modal-model-support)  
   2.5 [Omni Model Support](#25-omni-model-support)  
   2.6 [Data Parallelism (DP)](#26-data-parallelism-dp)  
   2.7 [Finding maximum Context Length](#27-finding-maximum-context-length)   
   2.8 [Multi-Modal Webui](#28-multi-modal-webui)  
   2.9 [Multi-node Distributed Deployment (PP/TP)](#29-multi-node-distributed-deployment-pptp)  
   2.10 [BPE-Qwen Tokenizer](#210-bpe-qwen-tokenizer)  
   2.11 [Load Balancer Solution](#211-load-balancer-solution)
4. [Supported Models](#3-supported-models)  
5. [Troubleshooting](#4-troubleshooting)
6. [Performance tuning](#5-performance-tuning)

---

## 1. Getting Started and Usage

We provide two offerings to setup the environment and run evaluation:

- Offline Installer for Bare Metal Environment Setup  
Maintained on Intel RDC website. It will include all necessary components such as Linux kernel, GPU firmware, graphics driver, tools, the update of system configuration and often used scripts without internet requirement. 
This installer can run in either bare metal or docker environments. In docker environment, it will skip the installation of Linux kernel, GPU firmware and the update of system level configuration.

- vllm Inference Docker Image (llm-scaler-vllm)  
Maintained on Dockerhub. It already uses above offline installer to align the base platform environment. Meanwhile includes the components for LLM inference such as vllm/IPEX.

The diagram below depicts the components of each offering and how they relate to each other.
<img width="2521" height="1015" alt="image" src="https://github.com/user-attachments/assets/63849b79-7c20-4b53-879b-8c13f9109ec4" />

Typically, users have below two use cases:

| Use Case | Description | Required Steps |
| -------- | ----------- | -------------- |
| **Platform Evaluation** | For evaluating platform capabilities only, with no intention to run vLLM inference. | 1. Install **Ubuntu 25.04** <br> 2. Download and run offline installer <br> 3. Run platform evaluation script after the installation in bare metal environment |
| **vLLM Inference Benchmark** | For running inference benchmarks based on vLLM/IPEX. | 1. Install **Ubuntu 25.04** <br> 2. Download and run offline installer  <br> 3. Pull the **vLLM Docker image** from Docker Hub <br> 4. Download the target model <br> 5. Run **vLLM-based inference performance tests** |

Currently, we include the following features for basic platform evaluation such as GPU memory bandwidth, P2P/collective communication cross GPUs and GeMM (generic matrix multiply) compute.

**Note: Both offline installer and docker image are intended for demo purposes only and not intended for production use. For production, please refer to our docker file to generate your own image**
- [vllm docker file](https://github.com/intel/llm-scaler/blob/main/vllm/docker/Dockerfile)
- [platform_docker_file](https://github.com/intel/llm-scaler/blob/main/vllm/docker/Dockerfile.platform)

### 1.1 Install Bare Metal Environment

First, install a standard Ubuntu 25.04 from the following link. 
- [Ubuntu 25.04 Desktop](https://releases.ubuntu.com/25.04/ubuntu-25.04-desktop-amd64.iso) (for Xeon-W)
- [Ubuntu 25.04 Server](https://releases.ubuntu.com/25.04/ubuntu-25.04-live-server-amd64.iso) (for Xeon-SP).

Download Offline Installer from Intel RDC webiste. This can be download directly without registration requirement. 
[RDC Download Link](https://cdrdv2.intel.com/v1/dl/getContent/873591/873592)

Switch to root user, extract and installer and run installation script.

```bash
sudo su -
cd the_path_of_multi-arc-bmg-offline-installer-x.x.x.x 
./installer.sh
```` 

If everything is ok, you can see below installation completion message. Then please reboot to apply changes.

```bash
[INFO] Intel Multi-ARC base platform installation complete.
[INFO] Please reboot the system to apply changes.

Tools installed: gemm / 1ccl / xpu-smi in /usr/bin
level-zero-tests: ./tools/level-zero-tests
Support scripts: ./scripts
Installation log: ./install_log_20260129_164959.log
````

### 1.2 Run Platform Evaluation

After the reboot, go to /opt/intel/multi-arc directory, tools/scripts are there.

```bash
(base) root@intel:~/multi-arc-bmg-offline-installer-26.5.6.1# ll
total 48
drwxrwxr-x  6 intel intel 4096 Jan 26 16:16 ./
drwx------ 19 root  root  4096 Jan 29 16:55 ../
-rwxrwxr-x  1 intel intel 5769 Jan 23 11:21 install*
-rwxrwxr-x  1 intel intel 3439 Jan 23 11:31 installer.sh*
-rw-rw-r--  1 intel intel  818 Jan 23 11:31 README.md
drwxrwxr-x  2 intel intel 4096 Jan 20 19:58 results/
drwxrwxr-x  7 intel intel 4096 Jan 23 11:21 scripts/
drwxrwxr-x  3 intel intel 4096 Jan 20 19:58 tools/
drwxrwxr-x  9 intel intel 4096 Jan 19 15:31 ubuntu-25.04-desktop/
-rwxrwxr-x  1 intel intel  409 Jan 23 11:21 uninstall*
-rw-rw-r--  1 intel intel  587 Jan 23 11:31 VERSION
````

Please read the README.md firstly to understand all of our offerings. Then your may use scripts/evaluation/platform_basic_evaluation.sh
to perform a quick evaluation with report under results. We also provide a reference perf under results/

```bash
(base) root@intel:~/multi-arc-bmg-offline-installer-26.5.6.1# ls results/ -l
total 8
-rw-rw-r-- 1 intel intel 552 Jan 23 11:31 reference_perf_b60.csv
-rw-rw-r-- 1 intel intel 535 Jan 23 11:31 reference_perf_b70.csv
````

When you meet issue requiring our support, you can use below script to get necesary information of your system.
```bash
(base) root@intel:~/multi-arc-bmg-offline-installer-26.5.6.1# ll scripts/debug/collect_sysinfo.sh
-rwxrwxr-x 1 intel intel 2701 Jan 23 11:31 scripts/debug/collect_sysinfo.sh*
````

You can also check our FAQ and known issues for more details.
```bash
https://github.com/intel/llm-scaler/blob/main/vllm/FAQ.md
https://github.com/intel/llm-scaler/blob/main/vllm/KNOWN_ISSUES.md
````

### 1.3 Pulling and Running the vllm Docker Container

First, pull the image for **Intel Arc B60 GPUs**:

> **⚠️ Important**
> Do **NOT** use the `latest` tag.
> Instead, go to the **Releases** page and pull the *exact* beta version:
> [https://github.com/intel/llm-scaler/blob/main/Releases.md/#latest-beta-release](https://github.com/intel/llm-scaler/blob/main/Releases.md/#latest-beta-release)
>
> This ensures you can precisely identify which version you are using.

Example:

```bash
# Replace <VERSION> with the latest beta release version from the link above
docker pull intel/llm-scaler-vllm:<VERSION>
```

**Notes:**
* `intel/llm-scaler-vllm:1.0` → PV release image (stable)
* `intel/llm-scaler-vllm:<VERSION>` → Recommended beta release (instead of `latest`)

**Supplement: For Intel Arc A770 GPUs**
```bash
docker pull intelanalytics/multi-arc-serving:latest
```
- Usage Instructions: [VLLM Docker Quickstart for A770](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/DockerGuides/vllm_docker_quickstart.md#3-start-the-docker-container)

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
    intel/llm-scaler-vllm:<VERSION>
```

Enter the container:

```bash
docker exec -it lsv-container bash
```

---

**Note — Mapping a Single GPU**
> If you need to map only a specific GPU into the container, remove both `--privileged` and `--device=/dev/dri` from the `docker run` command, and replace them with the following device and mount options (example for the first GPU):

```bash
--device /dev/dri/renderD128:/dev/dri/renderD128 \
--mount type=bind,source="/dev/dri/by-path/pci-0000:18:00.0-card",target="/dev/dri/by-path/pci-0000:18:00.0-card" \
--mount type=bind,source="/dev/dri/by-path/pci-0000:18:00.0-render",target="/dev/dri/by-path/pci-0000:18:00.0-render" \
-v /dev/dri/card0:/dev/dri/card0 \
```

This way, only the first GPU will be mapped into the Docker container.

---

**Note — Intel oneAPI Environment**
> How you start the container determines whether you need to manually source the Intel oneAPI environment (`source /opt/intel/oneapi/setvars.sh --force`):
>
> * **Interactive shell (`docker exec -it <container> bash`)**  
>   `/root/.bashrc` already sources the oneAPI environment. No manual action needed.
>
> * **Docker Compose, overridden ENTRYPOINT, or direct `docker run` without interactive bash**  
>   The environment is **not automatically loaded** if no shell is involved. Prepend your command with `source /opt/intel/oneapi/setvars.sh --force &&` to ensure proper GPU/XPU setup.
>   
>   ```yaml
>   entrypoint: >
>     entrypoint: source /opt/intel/oneapi/setvars.sh --force && vllm serve --model /llm/models/Qwen3-14B
>   ```
>
> **Summary:** Automated starts require sourcing the oneAPI script; interactive bash sessions are ready to use.

---



### 1.4 Launching the Serving Service

### 1.4.0 (Optional)
If you don't download hf models before, you can try to start by this way, But you may meet network error. We recommend to start service by download models first and mount models on staring docker container. And the project [hf-mirror](https://hf-mirror.com/) is recommended to solve model download network error.
```bash
HF_TOKEN="<your_api_token>"
HF_HOME="/llm/models"
vllm serve \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B
```

### 1.4.1 Start the Serving Service using local model

```bash
# Start the vLLM service, logging to both file /llm/vllm.log and Docker logs
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
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
    -tp=1 \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```
you can add the argument `--api-key xxx` for user authentication. Users are supposed to send their requests with request header bearing the API key.

---

### 1.5 Benchmarking the Service

```bash
vllm bench serve \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dataset-name random \
    --served-model-name DeepSeek-R1-Distill-Qwen-7B \
    --random-input-len=1024 \
    --random-output-len=512 \
    --ignore-eos \
    --num-prompt 10 \
    --trust_remote_code \
    --request-rate inf \
    --backend vllm \
    --port=8000
```
### 1.6 (Optional) Monitoring the Service with Prometheus and Grafana

Refer to [here](monitor/README.md) for details.

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
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
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
    -tp=1 \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```

To use fp8 online quantization, simply replace `--quantization sym_int4` with:

```bash
--quantization fp8
```

For those models that have been quantized before, such as AWQ-Int4/GPTQ-Int4/FP8 models, user do not need to specify the `--quantization` option.

---

### 2.3 Embedding and Reranker Model Support

#### Start service with embedding task
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/bge-m3 \
    --served-model-name bge-m3 \
    --task embed \
    --dtype=float16 \
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
    --block-size 64 \
    -tp=1
```

---
After starting the vLLM service, you can follow this link to use it.

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

#### Start service with classify task

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/bge-reranker-base \
    --served-model-name bge-reranker-base \
    --task score \
    --dtype=float16 \
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
    --block-size 64 \
    -tp=1 \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```
After starting the vLLM service, you can follow this link to use it.
#### [Rerank api](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#re-rank-api)

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "bge-reranker-base",
  "query": "What is the capital of France?",
  "documents": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Horses and cows are both animals.",
    "The French have a rich tradition in engineering."
  ]
}'
```


---

### 2.4 Multi-Modal Model Support

#### Start service using V1 engine
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/Qwen2.5-VL-7B-Instruct \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --allowed-local-media-path /llm/models/test \
    --dtype=float16 \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=5120 \
    --disable-log-requests \
    --max-model-len=5120 \
    --block-size 64 \
    --quantization fp8 \
    -tp=1 \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```

After starting the vLLM service, you can follow this link to use it

#### [Multimodal image input](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#image-inputs_1)

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

if want to process image in server local, you can `"url": "file:/llm/models/test/1.jpg"` to test.

---

### 2.4.1 Audio Model Support [Deprecated]

#### Install audio dependencies
```bash
pip install transformers==4.52.4 librosa
```

#### Start service using V0 engine
```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=0 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/whisper-medium \
    --served-model-name whisper-medium \
    --allowed-local-media-path /llm/models/test \
    --dtype=float16 \
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

#### [Multimodal audio input](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#audio-inputs_1)

```bash
curl http://localhost:8000/v1/audio/transcriptions \
-H "Content-Type: multipart/form-data" \
-F file="@/llm/models/test/output.wav" \
-F model="whisper-large-v3-turbo"
```
---

### 2.4.2 OCR Model Support

Refer to [here](OCR/README.md) for details.

### 2.5 Omni Model Support

#### Install audio dependencies
```bash
pip install librosa soundfile
```

#### Start service using V1 engine
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/Qwen2.5-Omni-7B \
    --served-model-name Qwen2.5-Omni-7B \
    --allowed-local-media-path /llm/models/test \
    --dtype=float16 \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=5120 \
    --disable-log-requests \
    --max-model-len=5120 \
    --block-size 64 \
    --quantization fp8 \
    -tp=1 \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```

After starting the vLLM service, you can follow this link to use it

#### [Qwen-Omni input](https://github.com/QwenLM/Qwen2.5-Omni?tab=readme-ov-file#vllm-serve-usage)

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"}},
        {"type": "text", "text": "What is the text in the illustration, and what is the sound in the audio?"}
    ]}
    ]
    }'
```

An example responce is listed below:
```json
{"id":"chatcmpl-xxx","object":"chat.completion","model":"Qwen2.5-Omni-7B","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":null,"content":"The text in the image is \"TONGYI Qwen\". The sound in the audio is a cough.","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":156,"total_tokens":180,"completion_tokens":24,"prompt_tokens_details":null},"prompt_logprobs":null,"kv_transfer_params":null}
```

For video input, one can input like this:

```bash
curl -sS http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen3-Omni-30B-A3B-Instruct",
    "temperature": 0,
    "max_tokens": 1024,
    "messages": [{
      "role": "user",
      "content": [
        { "type": "text", "text": "Please describe the video comprehensively as much as possible." },
        { "type": "video_url", "video_url": { "url": "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4" } }
      ]
    }]
  }'
```


---

### 2.6 Data Parallelism (DP)

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

> **Note**
> In addition to DP, a **load balancer–based deployment** is also supported as a drop-in alternative.
> It provides slightly better performance in some scenarios and supports periodic instance rotation for long-running services.
> See [Section 2.11 Load Balancer](#211-load-balancer-solution) for details.


---

### 2.7 Finding maximum Context Length
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

### 2.8 Multi-Modal Webui
The project provides two optimized interfaces for interacting with Qwen2.5-VL models:


#### 📌 Core Components
- **Inference Engine**: vLLM (Intel-optimized)
- **Interfaces**: 
  - Gradio (for rapid prototyping)
  - ComfyUI (for complex workflows)

#### 🚀 Deployment Options

#### Option 1: Gradio Deployment (Recommended for Most Users)
- check `/llm-scaler/vllm/webui/multi-modal-gradio/README.md` for implementation details

#### Option 2: ComfyUI Deployment (Advanced Workflows)
- check `/llm-scaler/vllm/webui/multi-modal-comfyui/README.md` for implementation details


#### 🔧 Configuration Guide

| Parameter | Effect | Recommended Value |
|-----------|--------|-------------------|
| `--quantization fp8` | XPU acceleration | Required |
| `-tp=2` | Tensor parallelism | Match GPU count |
| `--max-model-len` | Context window | 32768 (max) |

---


### 2.9 Multi-node Distributed Deployment (PP/TP)

Supports multi-node distributed deployment with **pipeline parallelism (PP)** and **tensor parallelism (TP)**, using **Docker Swarm, SSH, MPI, and Ray**. This enables scaling across multiple machines with coordinated communication.

---

#### **Step 1. Setup Docker Swarm**

On **Machine A (Node-1)**:

```bash
docker swarm init --advertise-addr <MachineA_IP>
```

On **Machine B (Node-2)** (use the join command printed from Machine A):

```bash
docker swarm join --token <token> <MachineA_IP>:2377
```

Create overlay network (on any node):

```bash
docker network create --driver overlay --attachable my-overlay
```

Check cluster:

```bash
docker node ls
docker network ls --filter driver=overlay
```

---

#### **Step 2. Start Containers**

**Node-1:**

```bash
sudo docker run -td \
    --privileged \
    --network=my-overlay \
    --device=/dev/dri \
    --name=node-1 \
    -v /model_path:/llm/models/ \
    -e no_proxy=localhost,127.0.0.1 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    intel/llm-scaler-vllm:0.10.0-b2
```

**Node-2:**

```bash
sudo docker run -td \
    --privileged \
    --network=my-overlay \
    --device=/dev/dri \
    --name=node-2 \
    -v /model_path:/llm/models/ \
    -e no_proxy=localhost,127.0.0.1 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    intel/llm-scaler-vllm:0.10.0-b2
```

Enter container:

```bash
docker exec -it node-1 bash
```

---

#### **Step 3. Configure Hostname and SSH**

Inside each container:

```bash
hostname node-1   # on Node-1
hostname node-2   # on Node-2
```

Install networking & SSH:

```bash
apt update && apt install -y iputils-ping openssh-client openssh-server net-tools
```

Start SSH:

```bash
mkdir -p /var/run/sshd
/usr/sbin/sshd
```

Enable root login (`/etc/ssh/sshd_config`):

```
PermitRootLogin yes
PasswordAuthentication yes
```

Restart SSH:

```bash
pkill sshd
/usr/sbin/sshd
```

Set root password:

```bash
passwd
# use: rootpass123
```

Verify SSH port:

```bash
netstat -tlnp | grep :22
```

Generate SSH key (Node-1):

```bash
ssh-keygen -t rsa -b 4096 -C "user@domain"
ssh-copy-id root@node-2
```

Test login:

```bash
ssh node-2
```

Repeat same setup on **Node-2**.

---

#### **Step 4. Run MPI Tests**

Using hostnames:

```bash
mpirun -np 2 -ppn 1 -hosts node-1,node-2 hostname
```

Benchmark test:

```bash
mpirun -np 2 -ppn 1 -hosts node-1,node-2 /llm/models/benchmark
```

---

#### **Step 5. Start Ray Cluster**

**On Node-1 (Head):**

```bash
export VLLM_HOST_IP=10.0.1.19
ray start --block --head --port=6379 --num-gpus=1 --node-ip-address=10.0.1.19
```

**On Node-2 (Worker):**

```bash
export VLLM_HOST_IP=10.0.1.20
ray start --block --address=10.0.1.19:6379 --num-gpus=1
```

---

#### **Step 6. Launch vLLM Service**

Run on **Node-1**:

```bash
export MODEL_NAME="/llm/models/Qwen2.5-7B-Instruct"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export CCL_ATL_TRANSPORT=ofi
export VLLM_HOST_IP=10.0.1.19

vllm serve \
    --model $MODEL_NAME \
    --dtype=float16 \
    --enforce-eager \
    --port 8005 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=20000 \
    --block-size 64 \
    --served-model-name test \
    -tp=2 -pp=1 \
    --distributed-executor-backend ray \
    2>&1 | tee /llm/vllm.log > /proc/1/fd/1 &

# Use tail to view logs in the current terminal
# If the user wants to see logs in real-time in the current terminal, 
# they can remove '> /proc/1/fd/1 &' and run in the foreground:
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ... 2>&1 | tee /llm/vllm.log
tail -f /llm/vllm.log
```


At this point, multi-node distributed inference with **PP + TP** is running, coordinated by **Ray** across Node-1 and Node-2.

---


### 2.10 BPE-Qwen Tokenizer

We have integrated the **bpe-qwen tokenizer** to accelerate tokenization for Qwen models.

**Note:** You need to install it first:
```
pip install bpe-qwen
```

To enable it when launching the API server, add:

```bash
--tokenizer-mode bpe-qwen
```

---

### 2.11 Load Balancer Solution

This document describes a **load balancer–based deployment** for vLLM using Docker Compose.
The load balancer routes traffic to multiple vLLM instances and exposes a single endpoint.

Once started, send requests to:

```
http://localhost:8000
```


#### Use Case 1: Drop-in Alternative to DP

Use this setup as a **drop-in alternative to DP**.

Compared to DP, the load balancer approach provides **slightly better performance** in our testing and does not require any DP-specific configuration.

Start the Load Balancer

```bash
cd vllm/docker-compose/load_balancer
docker compose up -d
```

You can view logs in real time to monitor service status:

```bash
docker compose logs -f
```

After startup, all requests can be sent directly to:

```
http://localhost:8000
```

Stop / clean up:
```
docker compose down
```

#### Use Case 2: Periodic vLLM Rotation (Long-Running Service)

Use this when running vLLM for a long time and you want to periodically restart instances (e.g., once per day) to avoid degradation, without service interruption.

Start with Rotation Enabled

```bash
cd vllm/docker-compose/load_balancer
chmod +x vllm_bootstrap_and_rotate.sh
bash vllm_bootstrap_and_rotate.sh
```

You can view logs in real time to monitor service status:

```bash
docker compose logs -f
```

Once started, requests continue to be served at:

```
http://localhost:8000
```

To stop the rotation and clean up resources:

```bash
docker compose down
crontab -l | grep -v "vllm_bootstrap_and_rotate.sh" | crontab -
```

> This will stop all containers and remove the cron job that triggers periodic rotation.

---

## 3. Supported Models


| Category             | Model Name                                 | FP16 | Dynamic Online FP8 | Dynamic Online Int4 | MXFP4 | Notes                     |
|----------------------|--------------------------------------------|------|--------------------|----------------------|-------|---------------------------|
| Language Model       | openai/gpt-oss-20b                         |      |                    |                      |   ✅   |                           |
| Language Model       | openai/gpt-oss-120b                        |      |                    |                      |   ✅   |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-Distill-Llama-70B  |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B      |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | deepseek-ai/DeepSeek-V2-Lite               |  ✅  |         ✅         |                      |       | export VLLM_MLA_DISABLE=1 |
| Language Model       | deepseek-ai/deepseek-coder-33b-instruct    |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | Qwen/Qwen3-8B                              |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | Qwen/Qwen3-14B                             |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | Qwen/Qwen3-32B                             |  ✅  |         ✅         |          ✅          |       |                           |
| Language MOE Model   | Qwen/Qwen3-30B-A3B                         |  ✅  |         ✅         |          ✅          |       |                           |
| Language MOE Model   | Qwen/Qwen3-235B-A22B                       |      |         ✅         |                      |       |                           |
| Language MOE Model   | Qwen/Qwen3-Coder-30B-A3B-Instruct          |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | Qwen/QwQ-32B                               |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | mistralai/Ministral-8B-Instruct-2410       |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | mistralai/Mixtral-8x7B-Instruct-v0.1       |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | meta-llama/Llama-3.1-8B                    |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | meta-llama/Llama-3.1-70B                   |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | baichuan-inc/Baichuan2-7B-Chat             |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| Language Model       | baichuan-inc/Baichuan2-13B-Chat            |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| Language Model       | THUDM/CodeGeex4-All-9B                     |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| Language Model       | zai-org/GLM-4-9B-0414                      |      |         ✅        |                      |       | use bfloat16 |
| Language Model       | zai-org/GLM-4-32B-0414                     |      |         ✅        |                      |       | use bfloat16 |
| Language MOE Model   | zai-org/GLM-4.5-Air                        |  ✅  |         ✅         |                      |       |                           |
| Language Model       | ByteDance-Seed/Seed-OSS-36B-Instruct       |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | miromind-ai/MiroThinker-v1.5-30B           |  ✅  |         ✅         |          ✅          |       |                           |
| Language Model       | tencent/Hunyuan-0.5B-Instruct              |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](#31-how-to-use-hunyuan-7b-instruct)   |
| Language Model       | tencent/Hunyuan-7B-Instruct                |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](#31-how-to-use-hunyuan-7b-instruct)   |
| Multimodal Model     | Qwen/Qwen2-VL-7B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | Qwen/Qwen2.5-VL-7B-Instruct                |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | Qwen/Qwen2.5-VL-32B-Instruct               |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | Qwen/Qwen2.5-VL-72B-Instruct               |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | Qwen/Qwen3-VL-4B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | Qwen/Qwen3-VL-8B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal MOE Model | Qwen/Qwen3-VL-30B-A3B-Instruct             |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | openbmb/MiniCPM-V-2_6                      |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | openbmb/MiniCPM-V-4                        |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | openbmb/MiniCPM-V-4_5                      |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | OpenGVLab/InternVL2-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | OpenGVLab/InternVL3-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | OpenGVLab/InternVL3_5-8B                   |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal MOE Model | OpenGVLab/InternVL3_5-30B-A3B              |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | rednote-hilab/dots.ocr                     |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | ByteDance-Seed/UI-TARS-7B-DPO              |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | google/gemma-3-12b-it                      |      |         ✅         |                      |       |  use bfloat16  |
| Multimodal Model     | google/gemma-3-27b-it                      |      |         ✅         |                      |       |  use bfloat16  |
| Multimodal Model     | THUDM/GLM-4v-9B                            |  ✅  |         ✅         |          ✅         |       |  with --hf-overrides and chat_template  |
| Multimodal Model     | zai-org/GLM-4.1V-9B-Base                   |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | zai-org/GLM-4.1V-9B-Thinking               |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | zai-org/Glyph                              |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | opendatalab/MinerU2.5-2509-1.2B            |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | baidu/ERNIE-4.5-VL-28B-A3B-Thinking        |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | zai-org/GLM-4.6V-Flash                     |  ✅  |         ✅         |          ✅          |       |   pip install transformers==5.0.0rc0 first            |
| Multimodal Model     | PaddlePaddle/PaddleOCR-VL                  |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](OCR/README.md#3-paddler-ocr-support)     |
| Multimodal Model     | deepseek-ai/DeepSeek-OCR                   |  ✅  |         ✅         |          ✅          |       |                           |
| Multimodal Model     | moonshotai/Kimi-VL-A3B-Thinking-2506       |  ✅  |         ✅         |          ✅          |       |                           |
| omni                 | Qwen/Qwen2.5-Omni-7B                       |  ✅  |         ✅         |          ✅          |       |                           |
| omni                 | Qwen/Qwen3-Omni-30B-A3B-Instruct           |  ✅  |         ✅         |          ✅          |       |                           |
| audio                | openai/whisper-medium                      |  ✅  |         ✅         |          ✅          |       |                           |
| audio                | openai/whisper-large-v3                    |  ✅  |         ✅         |          ✅          |       |                           |
| Embedding Model      | Qwen/Qwen3-Embedding-8B                    |  ✅  |         ✅         |          ✅          |       |                           |
| Embedding Model      | BAAI/bge-m3                                |  ✅  |         ✅         |          ✅          |       |                           |
| Embedding Model      | BAAI/bge-large-en-v1.5                     |  ✅  |         ✅         |          ✅          |       |                           |
| Reranker Model       | Qwen/Qwen3-Reranker-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| Reranker Model       | BAAI/bge-reranker-large                    |  ✅  |         ✅         |          ✅          |       |                           |
| Reranker Model       | BAAI/bge-reranker-v2-m3                    |  ✅  |         ✅         |          ✅          |       |                           |


--- 

### 3.1 how to use Hunyuan-7B-Instruct 
install new transformers version
```bash
pip install transformers==4.56.1
```

Need to use the followng format like [here](https://huggingface.co/tencent/Hunyuan-7B-Instruct#use-with-transformers), and you can decide to use `think` or not.
```bash
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
"model": "Hunyuan-7B-Instruct",
"messages": [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "/no_thinkWhat is AI?"}]
    }
],
"max_tokens": 128
}'
```

## 4. Troubleshooting

### 4.1 ModuleNotFoundError: No module named 'vllm.\_C'

If you encounter the following error:

```
ModuleNotFoundError: No module named 'vllm._C'
```

This may be caused by running your script from within the `/llm/vllm` directory.

To avoid this error, make sure to run your commands from the `/llm` root directory instead. For example:

```bash
cd /llm
python3 -m vllm.entrypoints.openai.api_server
```

### 4.2 Out-of-memory while online quantization

When the model size is very large, running FP8 online quantization may cause out-of-memory errors.

To avoid this issue, set the following environment variable before starting the service:

```bash
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
```


## 5. Performance tuning

To improve performance, you can optimize CPU affinity based on the GPU–NUMA topology.

For example, if your process uses two GPUs that are both connected to NUMA node 0, you can use lscpu to identify the CPU cores associated with that NUMA node:

```bash
edgeai@edgeaihost27:~$ lscpu
NUMA:
  NUMA node(s):           4
  NUMA node0 CPU(s):      0-17,72-89
  NUMA node1 CPU(s):      18-35,90-107
  NUMA node2 CPU(s):      36-53,108-125
  NUMA node3 CPU(s):      54-71,126-143
```

Then, launch the service by binding it to the relevant CPU cores:
```bash
numactl -C 0-17 YOUR_COMMAND
```

This ensures that the CPU threads serving your GPUs remain on the optimal NUMA node, reducing memory access latency and improving throughput.
