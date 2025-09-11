Running Miner-U on intel BMG cards

## 1. Start docker and download models

```bash
docker pull intel/llm-scaler-vllm:0.10.0-b3
export DOCKER_IMAGE=intel/llm-scaler-vllm:0.10.0-b3
name="test-mineru"
docker rm -f $name
docker run -itd \
        --net=host \
        --device=/dev/dri \
        --privileged \
        --entrypoint /bin/bash \
        --name=$name \
        -v /home/intel/LLM:/llm/models/ \
        -e http_proxy=your_http_proxy \
        -e https_proxy=your_http_proxy \
        -e HF_ENDPOINT=https://hf-mirror.com \
        -e no_proxy="127.0.0.1,localhost" \
        --shm-size="16g" \
        $DOCKER_IMAGE

docker exec -it test-mineru /bin/bash
mineru-models-download
# choose huggingface
# choose pipeline
```


## 2. start miner-u and use

offline running:
```bash
mineru -p /llm/MinerU/demo/pdfs/small_ocr.pdf -o ./ --source local --device xpu
```

start api-serving to use:
```bash
export MINERU_DEVICE_MODE="xpu"
mineru-api --host 0.0.0.0 --port 8008

time curl -X POST "http://127.0.0.1:8008/file_parse" \
  -F "files=@/llm/MinerU/demo/pdfs/small_ocr.pdf" \
  -F "output_dir=./output" \
  -F "lang_list=ch"
```

start gradio web-ui to use:
```bash
export MINERU_DEVICE_MODE="xpu"
mineru-gradio --server-name 0.0.0.0 --server-port 7860
```

Refer to [here](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#_2) for more details.
