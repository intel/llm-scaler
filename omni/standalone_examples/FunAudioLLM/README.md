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

```python
# Make sure to run this script from /audio/CosyVoice
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

def cosyvoice3_example():
    # Using the mounted model path
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    
    # zero_shot usage
    # Ensure ./asset/zero_shot_prompt.wav exists (it is part of the repo)
    for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print(f"Saved zero_shot_{i}.wav")

if __name__ == '__main__':
    cosyvoice3_example()
```

## 5. Run Inference (FunASR)

```python
# Make sure to run this script from /audio/FunASR
from funasr import AutoModel

# Path mapped in the docker run command
model_dir = "pretrained_models/Fun-ASR-Nano-2512"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="xpu",  # Use 'xpu' for Intel GPU
)

# Replace with your audio file path
wav_path = "/path/to/your/audio.wav"

res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
text = res[0]["text"]
print(text)
```


