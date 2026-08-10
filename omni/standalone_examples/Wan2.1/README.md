# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-omni:latest-wan2.1
export CONTAINER_NAME=wan-2.1
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

docker exec -it wan-2.1 bash
```

Run Wan 2.1 demo on Single GPU:
```bash
python3.10 generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir /llm/models/Wan2.1-T2V-1.3B  --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --frame_num 33 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

```
