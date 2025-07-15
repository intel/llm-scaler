set -x
docker build -f ./docker/Dockerfile . -t intelanalytics/hunyuan3d-2.1:0715 --build-arg https_proxy=http://proxy.iil.intel.com:911 --build-arg http_proxy=http://proxy.iil.intel.com:911