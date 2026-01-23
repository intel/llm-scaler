set -x

export HTTP_PROXY=<your_http_proxy>
export HTTPS_PROXY=<your_https_proxy>
export INSTALL_AUDIO_NODES=<true_or_false>
export INSTALL_HY_MOTION1_NODES=<true_or_false>
export INSTALL_HUNYUAN3D_NODES=<true_or_false>

docker build -f ./docker/Dockerfile . -t intel/llm-scaler-omni:0.1.0-b5 --build-arg https_proxy=$HTTPS_PROXY --build-arg http_proxy=$HTTP_PROXY --build-arg INSTALL_AUDIO_NODES=$INSTALL_AUDIO_NODES --build-arg INSTALL_HY_MOTION1_NODES=$INSTALL_HY_MOTION1_NODES --build-arg INSTALL_HUNYUAN3D_NODES=$INSTALL_HUNYUAN3D_NODES