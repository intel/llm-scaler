export http_proxy=<your_http_proxy>
export https_proxy=<your_https_proxy>
export no_proxy=localhost,127.0.0.1

export model="/llm/models/Z-Image-Turbo/"

SERVER_ARGS=(
  --model-path $model
  --vae-cpu-offload
  --pin-cpu-memory
  --num-gpus 1
  --ulysses-degree=1
  --ring-degree=1
  --port 30010
)

sglang serve "${SERVER_ARGS[@]}" 2>&1 | tee sglang.log