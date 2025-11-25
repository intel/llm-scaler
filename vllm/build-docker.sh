sudo docker build \
   --build-arg http_proxy=http://child-prc.intel.com:913/ \
   --build-arg https_proxy=http://child-prc.intel.com:913/ \
   -t amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:temp-b6-platform \
   -f ./docker/Dockerfile \
   .
 
