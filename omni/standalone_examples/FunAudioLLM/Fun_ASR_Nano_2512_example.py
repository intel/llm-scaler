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
