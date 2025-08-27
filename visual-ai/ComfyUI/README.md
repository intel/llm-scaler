# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-visualai:latest-comfyui
export CONTAINER_NAME=comfyui
export MODEL_DIR=<your_model_dir>
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        -v $MODEL_DIR:/llm/models/ \
        --shm-size="16g" \
        --entrypoint=/bin/bash \
        $DOCKER_IMAGE

docker exec -it wan-2.2 bash

MODEL_PATH=/llm/models/comfyui/comfyui_models/
rm -rf /llm/ComfyUI/models
ln -s $MODEL_PATH /llm/ComfyUI/models
echo "Symbolic link created from $MODEL_PATH to /llm/ComfyUI/models"
```

Start ComfyUI:
```bash
cd /llm/ComfyUI
python3 main.py
```
