set -x

export HTTP_PROXY=<your_http_proxy>
export HTTPS_PROXY=<your_https_proxy>

TAG="0.1.0-b7-dev"

DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile . -t intel/llm-scaler-omni:$TAG --build-arg https_proxy=$HTTPS_PROXY --build-arg http_proxy=$HTTP_PROXY
