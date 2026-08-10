# FunAudioLLM on Intel GPU

This project provides a Docker environment for running FunAudioLLM (CosyVoice and FunASR) on Intel GPUs.

## 1. Prepare Models

Make sure you have the following models ready on your host machine:
- `Fun-CosyVoice3-0.5B`
- `Fun-ASR-Nano-2512`

## 2. Build Docker Image

Use the provided script to build the Docker image.

```bash
bash build.sh
```

## 3. Run Docker Container

Run the container by mounting the downloaded models to the expected paths.

- **CosyVoice Model**: Mount to `/audio/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`
- **FunASR Model**: Mount to `/audio/FunASR/pretrained_models/Fun-ASR-Nano-2512`

Replace `/path/to/...` with the actual paths on your host machine.

```bash
# Set your model paths
export HOST_COSYVOICE_MODEL=/path/to/Fun-CosyVoice3-0.5B
export HOST_FUNASR_MODEL=/path/to/Fun-ASR-Nano-2512
export DOCKER_IMAGE=llm-scaler-omni:fun-audio-llm
export CONTAINER_NAME=fun-audio-llm

sudo docker run -itd \
    --privileged \
    --net=host \
    --device=/dev/dri \
    -e no_proxy=localhost,127.0.0.1 \
    --name=$CONTAINER_NAME \
    -v $HOST_COSYVOICE_MODEL:/audio/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    -v $HOST_FUNASR_MODEL:/audio/FunASR/pretrained_models/Fun-ASR-Nano-2512 \
    --shm-size="16g" \
    --entrypoint=/bin/bash \
    $DOCKER_IMAGE

docker exec -it fun-audio-llm /bin/bash
```

## 4. Run Inference (CosyVoice3)

```bash
cd /audio/CosyVoice
python cosyvoice3_example.py
```

## 5. Run Inference (FunASR)

```bash
cd /audio/FunASR
python Fun_ASR_Nano_2512_example.py
```
