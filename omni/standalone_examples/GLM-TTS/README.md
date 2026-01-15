# GLM-TTS with Intel XPU Support

This project demonstrates how to run [GLM-TTS](https://github.com/zai-org/GLM-TTS) on Intel GPUs (XPU) using Docker.

## 1. Build the Docker Image

```bash
bash build.sh
```

## 2. Prepare Models

You need to download the pretrained models before running the container.

Download the [GLM-TTS](https://huggingface.co/zai-org/GLM-TTS) model.

## 3. Run the Container

Run the container by mounting the downloaded model to the expected path.

- **GLM-TTS Model**: Mount to `/audio/GLM-TTS/ckpt`

Replace `/path/to/...` with the actual paths on your host machine.

```bash
# Set your model paths
export GLM_TTS_MODEL_PATH=/path/to/GLM-TTS
export CONTAINER_NAME=glm-tts
export DOCKER_IMAGE=llm-scaler-omni:glm-tts

docker run -itd \
    --privileged \
    --net=host \
    --device /dev/dri \
    -e no_proxy=localhost,127.0.0.1 \
    --name=${CONTAINER_NAME} \
    -v ${GLM_TTS_MODEL_PATH}:/audio/GLM-TTS/ckpt \
    --shm-size="16g" \
    --entrypoint=/bin/bash \
    ${DOCKER_IMAGE}

docker exec -it ${CONTAINER_NAME} /bin/bash
```

## 4. Inference Examples

To run standard GLM-TTS inference:

```bash
cd /audio/GLM-TTS
python3 glmtts_inference.py \
    --data=example_zh \
    --exp_name=_test \
    --use_cache
```
