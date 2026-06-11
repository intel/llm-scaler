# LLM Scaler

LLM Scaler is an GenAI solution for text generation, image generation, video generation etc. running on Intel® Arc™ Pro B60 and B70 GPUs. LLM Scalar leverages standard frameworks such as vLLM, ComfyUI, SGLang Diffusion, Xinference etc and ensures the best performance for State-of-Art GenAI models running on Arc Pro B60/B70 GPUs.

---

## Latest Update
- 🔥[2026.06] We released `intel/llm-scaler-vllm:0.14.0-b8.3.1` to enable FP8 KV Cache and fix bugs for Qwen3/Qwen3.5 models. 
- 🔥[2026.05] We released `intel/llm-scaler-vllm:0.14.0-b8.3` to improve performance for Qwen3.5/3.6 series and Qwen3-Coder-Next, and enabled model streaming load to reduce peak memory. 
- [2026.05] We released `intel/llm-scaler-vllm:1.4` (or, `intel/llm-scaler-vllm:0.14.0-b8.2.1`) with new platform image and support Intel® Arc™ Pro B70 GPU. 
- [2026.05] We released `intel/llm-scaler-omni:0.1.0-b7` for more model workflows and performance improvments. 
- [2026.03] We released `intel/llm-scaler-vllm:0.14.0-b8.1` to support Qwen3.5-27B, Qwen3.5-35B-A3B and Qwen3.5-122B-A10B (FP8/INT4 online quantization, GPTQ)
- [2026.03] We released `intel/llm-scaler-omni:0.1.0-b6` for ComfyUI to support CacheDiT and torch.compile(), ComfyUI-GGUF, and more model workflows, and support FP8 for SGLang Diffusion.
- [2026.03] We released `intel/llm-scaler-vllm:0.14.0-b8` for vLLM 0.14.0 and PyTorch 2.10 support, various new models support and performance improvement. 
- [2026.01] We released `intel/llm-scaler-vllm:1.3` (or, `intel/llm-scaler-vllm:0.11.1-b7`) for vLLM 0.11.1 and PyTorch 2.9 support, various new models support and performance improvement.
- [2026.01] We released `intel/llm-scaler-omni:0.1.0-b5` for Python 3.12 and PyTorch 2.9 support, various ComfyUI workflows and more SGLang Diffusion support.
- [2025.12] We released `intel/llm-scaler-vllm:1.2`, same image as `intel/llm-scaler-vllm:0.10.2-b6`. 
- [2025.12] We released `intel/llm-scaler-omni:0.1.0-b4` to support ComfyUI workflows for Z-Image-Turbo, Hunyuan-Video-1.5 T2V/I2V with multi-XPU, and experimentially support SGLang Diffusion. 
- [2025.11] We released `intel/llm-scaler-vllm:0.10.2-b6` to support Qwen3-VL (Dense/MoE), Qwen3-Omni, Qwen3-30B-A3B (MoE Int4), MinerU 2.5, ERNIE-4.5-vl etc. 
- [2025.11] We released `intel/llm-scaler-vllm:0.10.2-b5` to support gpt-oss models and released `intel/llm-scaler-omni:0.1.0-b3` to support more ComfyUI workflows, and Windows installation.
- [2025.10] We released `intel/llm-scaler-omni:0.1.0-b2` to support more models with ComfyUI workflows and Xinference.
- [2025.09] We released `intel/llm-scaler-vllm:0.10.0-b3` to support more models (MinerU, MiniCPM-v-4.5 etc), and released `intel/llm-scaler-omni:0.1.0-b1` to enable first omni GenAI models using ComfyUI and Xinference on Arc Pro B60 GPU.
- [2025.08] We released `intel/llm-scaler-vllm:1.0`.



## LLM Scaler vLLM

`llm-scaler-vllm` supports running text generation models using vLLM, featuring: 

- ***CCL*** support (P2P or USM)
- ***INT4*** and ***FP8*** quantized online serving
- ***Embedding*** and ***Reranker*** model support
- ***Multi-Modal*** model support
- ***Omni*** model support
- ***Tensor Parallel***, ***Pipeline Parallel*** and ***Data Parallel***
- Finding maximum Context Length
- Multi-Modal WebUI
- BPE-Qwen tokenizer

Please follow the instructions in the [Getting Started](vllm/README.md/#1-getting-started-and-usage) to use `llm-scaler-vllm`. 

### Supported Models


| Model Name                                 | FP16 | Dynamic Online FP8 | Dynamic Online Int4 | MXFP4 | Notes                     |
|--------------------------------------------|------|--------------------|----------------------|-------|---------------------------|
| openai/gpt-oss-20b                         |      |                    |                      |   ✅   |                           |
| openai/gpt-oss-120b                        |      |                    |                      |   ✅   |                           |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B   |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B  |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B      |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-V2-Lite               |  ✅  |         ✅         |                      |       | export VLLM_MLA_DISABLE=1 |
| deepseek-ai/deepseek-coder-33b-instruct    |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-8B                              |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-14B                             |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-32B                             |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-30B-A3B                         |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-235B-A22B                       |      |         ✅         |                      |       |                           |
| Qwen/Qwen3-Coder-30B-A3B-Instruct          |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-Coder-Next                      |  ✅  |         ✅         |                    |       |                           |
| Qwen/Qwen3.5/3.6-27B                       |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3.5/3.6-35B-A3B                   |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3.5-122B-A10B                     |      |         ✅         |          ✅          |       |                           |
| Qwen/QwQ-32B                               |  ✅  |         ✅         |          ✅          |       |                           |
| mistralai/Ministral-8B-Instruct-2410       |  ✅  |         ✅         |          ✅          |       |                           |
| mistralai/Mixtral-8x7B-Instruct-v0.1       |  ✅  |         ✅         |          ✅          |       |                           |
| meta-llama/Llama-3.1-8B                    |  ✅  |         ✅         |          ✅          |       |                           |
| meta-llama/Llama-3.1-70B                   |  ✅  |         ✅         |          ✅          |       |                           |
| baichuan-inc/Baichuan2-7B-Chat             |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| baichuan-inc/Baichuan2-13B-Chat            |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| THUDM/CodeGeex4-All-9B                     |  ✅  |         ✅         |          ✅          |       | with chat_template        |
| zai-org/GLM-4-9B-0414                      |      |         ✅        |                      |       | use bfloat16 |
| zai-org/GLM-4-32B-0414                     |      |         ✅        |                      |       | use bfloat16 |
| zai-org/GLM-4.5-Air                        |  ✅  |         ✅         |                      |       |                           |
| zai-org/GLM-4.7-Flash                      |  ✅  |         ✅         |                      |       |                           |
| ByteDance-Seed/Seed-OSS-36B-Instruct       |  ✅  |         ✅         |          ✅          |       |                           |
| miromind-ai/MiroThinker-v1.5-30B           |  ✅  |         ✅         |          ✅          |       |                           |
| tencent/Hunyuan-0.5B-Instruct              |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](./vllm/README.md#31-how-to-use-hunyuan-7b-instruct)   |
| tencent/Hunyuan-7B-Instruct                |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](./vllm/README.md#31-how-to-use-hunyuan-7b-instruct)   |
| Qwen/Qwen2-VL-7B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen2.5-VL-7B-Instruct                |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen2.5-VL-32B-Instruct               |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen2.5-VL-72B-Instruct               |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-VL-4B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-VL-8B-Instruct                  |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-VL-30B-A3B-Instruct             |  ✅  |         ✅         |          ✅          |       |                           |
| openbmb/MiniCPM-V-2_6                      |  ✅  |         ✅         |          ✅          |       |                           |
| openbmb/MiniCPM-V-4                        |  ✅  |         ✅         |          ✅          |       |                           |
| openbmb/MiniCPM-V-4_5                      |  ✅  |         ✅         |          ✅          |       |                           |
| OpenGVLab/InternVL2-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| OpenGVLab/InternVL3-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| OpenGVLab/InternVL3_5-8B                   |  ✅  |         ✅         |          ✅          |       |                           |
| OpenGVLab/InternVL3_5-30B-A3B              |  ✅  |         ✅         |          ✅          |       |                           |
| rednote-hilab/dots.ocr                     |  ✅  |         ✅         |          ✅          |       |                           |
| ByteDance-Seed/UI-TARS-7B-DPO              |  ✅  |         ✅         |          ✅          |       |                           |
| google/gemma-3-12b-it                      |      |         ✅         |                      |       |  use bfloat16  |
| google/gemma-3-27b-it                      |      |         ✅         |                      |       |  use bfloat16  |
| THUDM/GLM-4v-9B                            |  ✅  |         ✅         |          ✅         |       |  with --hf-overrides and chat_template  |
| zai-org/GLM-4.1V-9B-Base                   |  ✅  |         ✅         |          ✅          |       |                           |
| zai-org/GLM-4.1V-9B-Thinking               |  ✅  |         ✅         |          ✅          |       |                           |
| zai-org/Glyph                              |  ✅  |         ✅         |          ✅          |       |                           |
| opendatalab/MinerU2.5-2509-1.2B            |  ✅  |         ✅         |          ✅          |       |                           |
| baidu/ERNIE-4.5-VL-28B-A3B-Thinking        |  ✅  |         ✅         |          ✅          |       |                           |
| zai-org/GLM-4.6V-Flash                     |  ✅  |         ✅         |          ✅          |       |   pip install transformers==5.0.0rc0 first            |
| PaddlePaddle/PaddleOCR-VL                  |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](./vllm/README.md#32-how-to-use-paddleocr)     |
| deepseek-ai/DeepSeek-OCR                   |  ✅  |         ✅         |          ✅          |       |                           |
| deepseek-ai/DeepSeek-OCR-2                 |  ✅  |         ✅         |          ✅          |       |  There may be accuracy issues when using `--quantization fp8`             |
| moonshotai/Kimi-VL-A3B-Thinking-2506       |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen2.5-Omni-7B                       |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-Omni-30B-A3B-Instruct           |  ✅  |         ✅         |          ✅          |       |                           |
| openai/whisper-medium                      |  ✅  |         ✅         |          ✅          |       |                           |
| openai/whisper-large-v3                    |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-Embedding-8B                    |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen3-VL-Embedding-2B/8B                   |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](https://github.com/vllm-project/vllm/blob/2f4226fe5280b60c47b4f6f01d9b18ac9cda2038/examples/pooling/embed/vision_embedding_online.py)                    |
| BAAI/bge-m3                                |  ✅  |         ✅         |          ✅          |       |                           |
| BAAI/bge-large-en-v1.5                     |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen/Qwen3-Reranker-8B                     |  ✅  |         ✅         |          ✅          |       |                           |
| Qwen3-VL-Reranker-2B/8B                    |  ✅  |         ✅         |          ✅          |       |  follow the guide in [here](https://github.com/vllm-project/vllm/blob/2f4226fe5280b60c47b4f6f01d9b18ac9cda2038/examples/pooling/score/vision_rerank_api_online.py)                    |
| BAAI/bge-reranker-large                    |  ✅  |         ✅         |          ✅          |       |                           |
| BAAI/bge-reranker-v2-m3                    |  ✅  |         ✅         |          ✅          |       |                           |



--- 


## LLM Scaler Omni (experimental)

`llm-scaler-omni` supports running image/voice/video generation etc., featuring `Omni Studio` mode (using ComfyUI) and `Omni Serving` mode (via SGLang Diffusion or Xinference).  


Please follow the instructions in the [Getting Started](omni/README.md/#getting-started-with-omni-docker-image) to use `llm-scaler-omni`. 


### Omni Demos

| Qwen-Image | Multi B60 Wan2.2-T2V-14B |
|------------|--------------------------|
| ![Qwen Image Demo](./omni/assets/demo_qwen_image.gif) | ![Wan2.2 T2V Demo](./omni/assets/demo_wan2.2_14b_i2v_multi_xpu.gif) |


### Omni Studio (ComfyUI WebUI interaction)

`Omni Stuido` supports Image Generation/Edit, Video Generation, Audio Generation, 3D Generation etc.  


| Model Category | Model | Type | 
|----------------------|------------|---------------|
| **Image Generation** | Qwen-Image, Qwen-Image-Edit | Text-to-Image, Image Editing | 
| **Image Generation** | Stable Diffusion 3.5 | Text-to-Image, ControlNet | 
| **Image Generation** | Z-Image-Turbo | Text-to-Image | 
| **Image Generation** | Flux.1, Flux.1 Kontext dev | Text-to-Image, Multi-Image Reference, ControlNet | 
| **Image Generation** | FireRed-Image-Edit-1.1 | Image Editing | 
| **Video Generation** | Wan2.2 TI2V 5B, Wan2.2 T2V 14B, Wan2.2 I2V 14B | Text-to-Video, Image-to-Video | 
| **Video Generation** | Wan2.2 Animate 14B | Video Animation | 
| **Video Generation** | HunyuanVideo 1.5 8.3B | Text-to-Video, Image-to-Video | 
| **Video Generation** | LTX-2 | Text-to-Video, Image-to-Video | 
| **3D Generation** | Hunyuan3D 2.1 | Text/Image-to-3D | 
| **Audio Generation** | VoxCPM1.5, IndexTTS 2 | Text-to-Speech, Voice Cloning | 
| **Video Upscaling** | SeedVR2 | Video Restoration and Upscaling | 


Please check [ComfyUI Support](omni/README.md/#comfyui) for more details.

### Omni Serving (OpenAI-API compatible serving)

`Omni Serving` supports Image Generation, Audio Generation etc.

- Image Generation (`/v1/images/generations`): Stable Diffusion 3.5, Flux.1-dev
- Text to Speech (`/v1/audio/speech`): Kokoro 82M
- Speech to Text (`/v1/audio/transcriptions`): whisper-large-v3

Please check [Xinference Support](omni/README.md/#xinference) for more details. 

---
## Releases
- Please check out the Docker image releases for [llm-scaler-vllm](Releases.md/#llm-scaler-vllm) and [llm-scaler-omni](Releases.md/#llm-scaler-omni)

---
## Get Support
- Please report a bug or raise a feature request by opening a [Github Issue](https://github.com/intel/llm-scaler/issues)
