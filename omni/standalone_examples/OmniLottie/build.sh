export HTTP_PROXY=<your_http_proxy>
export HTTPS_PROXY=<your_http_proxy>

docker build -f ./docker/Dockerfile . -t llm-scaler-omni:omnilottie --build-arg https_proxy=$HTTPS_PROXY --build-arg http_proxy=$HTTP_PROXY