export model="/llm/models/Z-Image-Turbo/"

SERVER_ARGS=(
  --model-path $model
  --vae-cpu-offload
  --pin-cpu-memory
  --num-gpus 2
  --tp-size=2
  --port 30010
  --attention-backend torch_sdpa
)

sglang serve "${SERVER_ARGS[@]}" 2>&1 | tee sglang.log
