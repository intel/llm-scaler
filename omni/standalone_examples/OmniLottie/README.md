# GLM-TTS with Intel XPU Support

This project demonstrates how to run [OmniLottie](https://github.com/OpenVGLab/OmniLottie) on Intel GPUs (XPU) using Docker.

## 1. Build the Docker Image

```bash
bash build.sh
```

## 2. Prepare Models

You need to download the pretrained models before running the container.

Download the [OmniLottie](https://huggingface.co/OmniLottie/OmniLottie) model.

## 3. Run the Container

Replace `/path/to/...` with the actual paths on your host machine.

```bash
# Set your model paths
export MODEL_PATH=/path/to/OmniLottie
export CONTAINER_NAME=omnilottie
export DOCKER_IMAGE=llm-scaler-omni:omnilottie

docker run -itd \
    --privileged \
    --net=host \
    --device /dev/dri \
    -e no_proxy=localhost,127.0.0.1 \
    --name=${CONTAINER_NAME} \
    -v ${MODEL_PATH}:/llm/model/OmniLottie \
    --shm-size="16g" \
    --entrypoint=/bin/bash \
    ${DOCKER_IMAGE}

docker exec -it ${CONTAINER_NAME} /bin/bash
```

## 4. Inference Examples

To run standard OmniLottie inference:

```bash
cd /llm/OmniLottie

# Modify `checkpoint_path` in `app.py` to `/llm/model/OmniLottie`

python app.py
```
