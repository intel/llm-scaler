# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-omni:qwen-image
export CONTAINER_NAME=qwen-image
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

docker exec -it qwen-image /bin/bash
```

Download models before start example.
Run qwen-image demo on Single GPU:
```bash
python3 qwen_image_example.py
```

Run qwen-image demo on two GPUs:
```bash
export qwen_image_enable_two_card=1
python3 qwen_image_example.py
```