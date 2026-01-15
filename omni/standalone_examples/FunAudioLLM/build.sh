export HTTP_PROXY=http://proxy.iil.intel.com:911
export HTTPS_PROXY=http://proxy.iil.intel.com:911

docker build -f ./docker/Dockerfile . -t llm-scaler-omni:fun-audio-llm --build-arg https_proxy=$HTTPS_PROXY --build-arg http_proxy=$HTTP_PROXY