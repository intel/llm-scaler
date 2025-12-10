set -x

export HTTP_PROXY=<your_http_proxy>
export HTTPS_PROXY=<your_https_proxy>

docker build -f ./docker/Dockerfile . -t intel/llm-scaler-omni:0.1.0-b4 --build-arg https_proxy=$HTTPS_PROXY --build-arg http_proxy=$HTTP_PROXY
