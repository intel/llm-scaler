# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-visualai:latest-wan2.1
export CONTAINER_NAME=wan-2.1
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

Run Wan 2.1 demo:
```bash

```

Known issues:
