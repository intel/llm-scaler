# Docker Image Releases for Intel Arc B60 Pro GPU: 

## LLM-Scaler-vLLM

### Latest Beta Release 
* `intel/llm-scaler-vllm:0.10.2-b6` [11/25/2025]:
    - MoE-Int4 support for Qwen3-30B-A3B
    - Bpe-Qwen tokenizer support
    - Enable Qwen3-VL Dense/MoE models
    - Enable Qwen3-Omni models
    - MinerU 2.5 Support
    - Enable whisper transcription models
    - Fix minicpmv4.5 OOM issue and output error
    - Enable ERNIE-4.5-vl models
    - Enable Glyph based GLM-4.1V-9B-Base

### 1.2 Release 
* `intel/llm-scaler-vllm:1.2` [12/11/2025]: 
    - Same image as intel/llm-scaler-vllm:0.10.2-b6 

### Previous Releases
* `intel/llm-scaler-vllm:1.1-preview` [09/29/2025]:
    - Same image as intel/llm-scaler-vllm:0.10.0-b2

* `intel/llm-scaler-vllm:0.10.2-b5` [11/04/2025]:
    - Support gpt-oss models 

* `intel/llm-scaler-vllm:0.10.0-b3` [09/23/2025]:
    - Support Seed-oss model
    - Support miner-U
    - Support MiniCPM-V-4_5
    - Fix internvl_3_5 and deepseek-v2-lite error

* `intel/llm-scaler-vllm:0.10.0-b2` [09/02/2025]:
    - Bug fix for sym_int4 online quantization on Multi-modal models
    
* `intel/llm-scaler-vllm:0.10.0-b1` [08/29/2025]:
    - Upgrade vLLM to 0.10.0 version
    - Support async scheduling with option --async-scheduling
    - Change to V1 engine for embedding/reranker models
    - Support pipeline parallelism with mp/ray backend
    - Support internvl3-8b
    - Support MiniCPM-v-4
    - Support InternVL3_5-8B

* `intel/llm-scaler-vllm:0.9.0-b3` [08/21/2025]:
    - Support Whisper
    - Support GLM-4.5-Air
    - Support dots.ocr
    - Support GLM-4.1V-9B-Thinking for image input
    - Optimize vLLM memory usage by updating profile_run logic
    - Enable/Optimize pipeline parallelism with Ray backend
   
* `intel/llm-scaler-vllm:1.0` [08/10/2025]: 
    - Same image as intel/llm-scaler-vllm:0.2.0-b2 

* `intel/llm-scaler-vllm:0.2.0-b2` [07/25/2025]:
    - Support by-layer online quantization to reduce the required GPU memory
    - Support embedding, rerank models
    - Enhance the support to multi-modal models
    - Maximum length auto-detecting
    - Support data parallelism
    - Support pipeline parallelism (experimental)
    - Support torch.compile (experimental)
    - Support speculative decoding (experimental)
    - Performance improvements
    - Bug fixes

## LLM-Scaler-Omni

### Latest Beta Release 
* `intel/llm-scaler-omni:0.1.0-b4` [12/10/2025]:
    - More workflows support:
        - Z-Image-Turbo
        - Hunyuan-Video-1.5 T2V/I2V with multi-XPU support
    - Initial support for SGLang Diffusion. 10% perf improvement compared to ComfyUI in 1*B60 scenario.

### Previous Releases
* `intel/llm-scaler-omni:0.1.0-b3` [11/19/2025]:
    - More workflows support:
        - Hunyuan 3D 2.1
        - Controlnet on Stable Diffusion 3.5, FLUX.1
        - Multi XPU support for Wan 2.2 I2V 14B rapid aio
        - AnimateDiff lightning
    - Add Windows installation

* `intel/llm-scaler-omni:0.1.0-b2` [10/17/2025]:
    - Fix issues:
        - Fix ComfyUI interpolate issue
        - Fix Xinference XPU index selection issue
    - Support more workflows:
        - ComfyUI
        - Wan2.2-Animate-14B basic workflow
        - Qwen-Image-Edit 2509 workflow
        - VoxCPM workflow
    - Xinference:
        - Kokoro-82M-v1.1-zh
  
* `intel/llm-scaler-omni:0.1.0-b1` [09/29/2025]:
    - Integrate ComfyUI on XPU and provide sample workflows for:
        - Wan2.2 TI2V 5B
        - Wan2.2 T2V 14B (multi-XPU supported)
        - FLUX.1 dev
        - FLUX.1 Kontext dev
        - Stable Diffusion 3.5 large
        - Qwen Image, Qwen Image Edit
    - Add support for xDit, Yunchang, and Raylight usages on XPU
    - Integrate Xinference with OpenAI-compatible APIs for:
        - Kokoro 82M
        - Whisper Large v3
        - Stable Diffusion 3.5 Medium
