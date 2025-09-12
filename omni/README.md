# llm-scaler-omni

---

## Table of Contents

1. [Getting Started with Omni Docker Image](#getting-started-with-omni-docker-image)
2. [ComfyUI](#comfyui)
3. [XInference](#xinference)
4. [Stand-alone Examples](#stand-alone-examples)

---

## Getting Started with Omni Docker Image

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=intel/llm-scaler-omni:0.1-b1
export CONTAINER_NAME=comfyui
export MODEL_DIR=<your_model_dir>
export COMFYUI_MODEL_DIR=<your_comfyui_model_dir>
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        -v $MODEL_DIR:/llm/models/ \
        -v $COMFYUI_MODEL_DIR:/llm/ComfyUI/models \
        --shm-size="64g" \
        --entrypoint=/bin/bash \
        $DOCKER_IMAGE

docker exec -it comfyui bash
```

## ComfyUI:
```bash
cd /llm/ComfyUI

MODEL_PATH=<your_comfyui_models_path>
rm -rf /llm/ComfyUI/models
ln -s $MODEL_PATH /llm/ComfyUI/models
echo "Symbolic link created from $MODEL_PATH to /llm/ComfyUI/models"

export http_proxy=<your_proxy>
export https_proxy=<your_proxy>
export no_proxy=localhost,127.0.0.1

python3 main.py
```

Then you can access the webUI at `http://<your_local_ip>:8188/`. On the left side, 

![workflow image](./assets/confyui_workflow.png)

### ComfyUI workflows

Currently, the following workflows are supported on B60:
- Qwen-Image (refer to https://raw.githubusercontent.com/Comfy-Org/example_workflows/main/image/qwen/image_qwen_image_distill.json)
- Qwen-Image-Edit (refer to https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/image_qwen_image_edit.json)
- Wan2.2-TI2V-5B (refer to https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_wan2_2_5B_ti2v.json)
- Wan2.2-T2V-14B with raylight (refer to https://github.com/komikndr/raylight/blob/main/example_workflows/WanT2V_Raylight.json)
- Flux.1 Kontext Dev(Basic) workflow in ComfyUI examples (refer to https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev)
- SD3.5 Simple in ComfyUI examples (refer to https://comfyanonymous.github.io/ComfyUI_examples/sd3/)

#### Qwen-Image

ComfyUI tutorial for qwen-image: https://docs.comfy.org/tutorials/image/qwen/qwen-image

Only `Qwen-Image Native Workflow Example` part is validated and there are some issues using LoRA. It's recommended to run the Distilled version for better performance.

#### Qwen-Image-Edit

ComfyUI tutorial for qwen-image-edit: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit

#### Wan2.2-TI2V-5B

ComfyUI tutorial for wan2.2: https://docs.comfy.org/tutorials/video/wan/wan2_2

Due to memory limit with single device, only `
Wan2.2 TI2V 5B Hybrid Version Workflow Example` is validated.

#### Wan2.2-T2V-14B with raylight

Currently using [WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) and [raylight](https://github.com/komikndr/raylight) as a faster solution with multi-XPU support. The model weights can get from [here](https://modelscope.cn/models/Phr00t/WAN2.2-14B-Rapid-AllInOne/files), and you may need to extract the unet part and VAE part seperately with `tools/extract.py`.

![wan_raylight](./assets/wan_raylight.png)

##### Follow the Steps to Complete the Workflow

1. Model Loading

- Ensure the `Load Diffusion Model (Ray)` node loads the diffusion model part from WAN2.2-14B-Rapid-AllInOne.
- Ensure the `Load VAE` node loads the VAE part from WAN2.2-14B-Rapid-AllInOne.
- Ensure the `Load CLIP` node loads `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

2. Ray configuration

Set the `GPU` and `ulysses_degree` in `Ray Init Actor` node to GPU nums you want to use.

3. Click the `Run` button or use the shortcut `Ctrl(cmd) + Enter` to run the workflow

## XInference

```bash
export ZE_AFFINITY_MASK=0 # In multi XPU environment, clearly select GPU index to avoid issues.
xinference-local --host 0.0.0.0 --port 9997
```
Supported models:
- Stable Diffusion 3.5 Medium
- Kokoro 82M
- whisper large v3

### WebUI Usage

#### 1. Access Xinference Web UI
![xinference_launch](./assets/xinference_launch.png)

#### 2. Select model and configure `model_path`
![xinference_model](./assets/xinference_configure.png)

#### 3. Find running model and launch Gradio UI for this model
![xinference_gradio](./assets/xinference_gradio.png)

#### 4. Generate within Gradio UI
![xinference_example](./assets/xinference_sd.png)

### OpenAI API Usage

> Visit http://127.0.0.1:9997/docs to inspect the API docs.

#### 1. Launch API service
You can select model and launch service via WebUI (refer to [here](#1-access-xinference-web-ui)) or by command:

```bash
export ZE_AFFINITY_MASK=0 # In multi XPU environment, clearly select GPU index to avoid issues.
xinference-local --host 0.0.0.0 --port 9997

xinference launch --model-name sd3.5-medium --model-type image --model-path /llm/models/stable-diffusion-3.5-medium/
```

#### 2. Post request in OpenAI API format

For TTS model (`Kokoro 82M` for example):
```bash
curl http://localhost:9997/v1/audio/speech   -H "Content-Type: application/json"   -d '{
    "model": "Kokoro-82M",
    "input": "kokoro, hello, I am kokoro." 
  }'   --output output.wav
```

For STT models (`whisper large v3` for example):
```bash
AUDIO_FILE_PATH=<your_audio_file_path>

curl -X 'POST' \
  "http://localhost:9997/v1/audio/translations" \
  -H 'accept: application/json' \
  -F "model=whisper-large-v3" \
  -F "file=@${AUDIO_FILE_PATH}"

{"text":" Cacaro's hello, I am Cacaro."}
```

For text-to-image models (`Stable Diffusion 3.5 Medium` for example):
```bash
curl http://localhost:9997/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sd3.5-medium",
    "prompt": "A Shiba Inu chasing butterflies on a sunny grassy field, cartoon style, with vibrant colors.",
    "n": 1,
    "size": "1024x1024",
    "quality": "standard",
    "response_format": "url"
  }'
```

## Stand-alone Examples 

> Notes: Stand-alone examples are excluded from `intel/llm-scaler-omni` image.

Supported models:
- Hunyuan3D 2.1
- Qwen Image
- Wan 2.1 / 2.2

