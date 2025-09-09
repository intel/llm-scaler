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
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        -v $MODEL_DIR:/llm/models/ \
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

```
xinference-local --host 0.0.0.0 --port 9997
```
Supported models:
- Stable Diffusion 3.5 Medium
- Kokoro 82M
- whisper large v3

## Stand-alone Examples 

> Notes: Stand-alone examples are excluded from `intel/llm-scaler-omni` image.

Supported models:
- Hunyuan3D 2.1
- Qwen Image
- Wan 2.1 / 2.2

