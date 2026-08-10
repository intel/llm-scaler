# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-omni:latest-wan2.2
export CONTAINER_NAME=wan-2.2
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
```

Run Wan 2.2 demo on Single B60 GPU:
```bash
python3 generate.py  --task ti2v-5B --size 1280*704 --ckpt_dir /llm/models/Wan2.2-TI2V-5B/  --offload_model True --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --convert_model_dtype  --frame_num 101 --sample_steps 50 
```

Run Wan 2.2 demo on 2 * B60 GPUs:
```bash
torchrun --nproc_per_node=2 generate.py --task ti2v-5B  --size 1280*704 --ckpt_dir /llm/models/Wan2.2-TI2V-5B/ --ulysses_size 2 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --offload_model True --t5_cpu --convert_model_dtype --frame_num 101 --sample_steps 50 
```
