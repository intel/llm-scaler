# HY-WorldPlay on Intel Arc GPU

This project enables [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) (Hunyuan Video World Simulator) to run on Intel Arc GPUs with XPU support.

## Sample Output on 4*B60
![sample output](./assets/sample_output.gif)

## Docker Setup

### Build Docker Image

```bash
export HTTP_PROXY=<your_http_proxy>
export HTTPS_PROXY=<your_https_proxy>

bash build.sh
```

### Run Docker Container

```bash
export DOCKER_IMAGE=llm-scaler-omni:hy-worldplay
export CONTAINER_NAME=hy-worldplay
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        -e http_proxy=<your_http_proxy> \
        -e https_proxy=<your_https_proxy> \
        --name=$CONTAINER_NAME \
        --shm-size="64g" \
        --entrypoint=/bin/bash \
        $DOCKER_IMAGE 
```

### Run HY-WorldPlay

```bash
docker exec -it hy-worldplay bash
# Working directory: /llm/HY-WorldPlay

bash run.sh
```

## Features

- Full Intel Arc GPU (XPU) support via PyTorch XPU backend
- Multi-GPU sequence parallelism with XCCL
- Memory-efficient chunked attention for large video generation
- Optimized VAE decoding with tile parallelism

## Environment Variables

- `CCL_SYCL_ALLTOALL_ARC_LL=1` - Enable optimized AllToAll for Arc GPUs
- `CCL_SYCL_CCL_BARRIER=1` - Enable CCL barrier synchronization
