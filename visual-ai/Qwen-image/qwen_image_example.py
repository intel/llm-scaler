from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import time
# https://huggingface.co/Qwen/Qwen-Image
model_path="/llm/models/models/Qwen-Image"
# https://huggingface.co/SahilCarterr/Qwen-Image-Distill-Full
# use it to get better performance
model_path1="/llm/models/models/Qwen-Image-Distill-Full"


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="xpu",
    model_configs=[
        ModelConfig(
                    model_id=model_path1, origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    #model_id=model_path, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    offload_device="cpu",
                    skip_download=True,),
        ModelConfig(model_id=model_path,
                    offload_device="cpu",
                    skip_download=True, origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id=model_path,
                    offload_device="cpu",
                    skip_download=True, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id=model_path, skip_download=True, origin_file_pattern="tokenizer/"),
)
#print(f"pipe: {pipe}")
pipe.enable_vram_management(vram_buffer=1)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
st0 = time.perf_counter()
image = pipe(prompt,
             seed=0,
             cfg_scale=1,
             num_inference_steps=15)
st1 = time.perf_counter()
print(f"total cost time is {st1-st0}s")

image.save("image.jpg")
