export model="/llm/models/Z-Image-Turbo/"

SERVER_ARGS=(
  --model-path $model
  --vae-cpu-offload
  --pin-cpu-memory
  --num-gpus 1
  --ulysses-degree=1
  --tp-size=1
  --port 30010
  --attention-backend torch_sdpa
)

sglang serve "${SERVER_ARGS[@]}" 2>&1 | tee sglang.log
