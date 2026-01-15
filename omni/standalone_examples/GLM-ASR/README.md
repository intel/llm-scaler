# GLM-ASR with Intel XPU Support

This project demonstrates how to run [GLM-ASR](https://github.com/zai-org/GLM-ASR) on Intel GPUs (XPU) using Docker.

## 1. Build the Docker Image

```bash
bash build.sh
```

## 2. Prepare Models

Download the [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) model.


## 3. Run the Container

Run the container by mounting the downloaded model to the expected path.

- **GLM-ASR Model**: Mount to `/audio/GLM-ASR/GLM-ASR-Nano-2512`

Replace `/path/to/...` with the actual paths on your host machine.

```bash
# Set your model paths
export GLM_ASR_MODEL_PATH=/path/to/GLM-ASR-Nano-2512
export CONTAINER_NAME=glm-asr
export DOCKER_IMAGE=llm-scaler-omni:glm-asr

docker run -itd \
    --privileged \
    --net=host \
    --device /dev/dri \
    -e no_proxy=localhost,127.0.0.1 \
    --name=${CONTAINER_NAME} \
    -v ${GLM_ASR_MODEL_PATH}:/audio/GLM-ASR/GLM-ASR-Nano-2512 \
    --shm-size="16g" \
    --entrypoint=/bin/bash \
    ${DOCKER_IMAGE}

docker exec -it ${CONTAINER_NAME} /bin/bash
```

## 4. Inference Examples

To run standard GLM-ASR inference:

```bash
cd /audio/GLM-ASR
python inference.py --checkpoint_dir ./GLM-ASR-Nano-2512 --audio examples/example_zh.wav
```
