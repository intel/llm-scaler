DOCKER_BUILDKIT=1 docker build \
  --build-arg http_proxy=http://proxy.iil.intel.com:911 \
  --build-arg https_proxy=http://proxy.iil.intel.com:911 \
  -t amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:b8-new \
  -f ./docker/Dockerfile \
  .

# sudo docker buildx build \
#   --build-arg http_proxy=http://proxy.iil.intel.com:911 \
#   --build-arg https_proxy=http://proxy.iil.intel.com:911 \
#   -t amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:b7.5 \
#   -f ./docker/Dockerfile \
#   --load .
