# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-visualai:latest
export CONTAINER_NAME=hunyuan3d-2.1
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        --entrypoint=/bin/bash \
        $DOCKER_IMAGE 
```

Run Hunyuan 3D 2.1 demo:
```bash
docker exec -it hunyuan3d-2.1 bash
# At /llm/Hunyuan3D-2.1 path

# Configure proxy to download model files
export http_proxy=<your_http_proxy>
export https_proxy=<your_https_proxy>
export no_proxy=localhost,127.0.0.1

# Run shape + paint demo
python3 demo.py
# (Optional) Run shape only demo
python3 demo_shape.py
# (Optional) Run paint only demo
python3 demo_texture.py
```