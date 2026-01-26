# OCR Model Support

## Table of Contents

1. [dots.ocr Support](#1-dotsocr-support)
2. [MinerU Support](#2-mineru-26-support)
3. [Paddler-OCR Support](#3-paddler-ocr-support)
4. [DeepSeek-OCR Support](#4-deepseek-ocr-support)

## 1. dots.ocr Support

To launch `dots.ocr`, follow the instructions in [Launching the Serving Service](../README.md#14-launching-the-serving-service), specifying the dots.ocr model, setting the model path to `/llm/models/dots.ocr`, the served-model-name to `model`, and the port to 8000.

Once the service is running, you can use the method provided in the `dots.ocr` repository to launch Gradio for testing.

---

### Clone the repository

```bash
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
```

### Install dependencies
```bash
pip install -e . --no-deps
pip install gradio gradio_image_annotation PyMuPDF qwen_vl_utils
```

### Launch Gradio for testing

```bash
python demo/demo_gradio.py 9000
```

You can refer to the dots.ocr [guide](https://github.com/rednote-hilab/dots.ocr#quick-start) for more details.

---

## 2. MinerU 2.6 Support

This guide shows how to launch the MinerU 2.6 model using the vLLM inference backend.

### Start the MinerU Service

Set up the environment variables and launch the vLLM API server:
```bash
export MODEL_NAME="/llm/models/MinerU2.5-2509-1.2B/"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1

python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --dtype float16 \
  --enforce-eager \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-util 0.85 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens=32768 \
  --max-model-len=32768 \
  --block-size 64 \
  --max-num-seqs 256 \
  --served-model-name MinerU \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor
```

> **💡 Notes**
>
> - `--logits-processors mineru_vl_utils:MinerULogitsProcessor` enables MinerU’s custom post-processing logic.

### Run the demo
1. To verify mineru

```bash
#mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://127.0.0.1:8000
mineru -p /llm/MinerU/demo/pdfs/small_ocr.pdf -o ./ -b vlm-http-client -u http://127.0.0.1:8000
```

2. Using by gradio

```bash
# refer to http://your_ip:8002/?view=api for gradio's api guide
mineru-gradio --server-name 0.0.0.0 --server-port 8002
```

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:8002/")
result = client.predict(
    file_path=handle_file('/llm/MinerU/demo/pdfs/small_ocr.pdf'),
    end_pages=500,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    language="ch",
    backend="vlm-http-client",
    url="http://localhost:8000",
    api_name="/to_markdown"
)
print(result)
```

You can refer to the MinerU [usage guide](https://opendatalab.github.io/MinerU/usage/) for more details.

---

## 3. Paddler-OCR Support

### Start vLLM Service
```
ZE_AFFINITY_MASK=6 \
vllm serve --model /llm/models/LLM2/PaddleOCR-VL \
    --served-model-name PaddleOCR-VL-0.9B \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --enforce-eager 
```

### Install Paddle Dependencies
```
pip install "paddleocr[doc-parser]" paddlepaddle
paddlex --install serving
```

### Deploy Paddle Service

- generate configuration
```
# include PP-DocLayoutV2 and PaddleOCR-VL-0.9B by default
paddlex --get_pipeline_config PaddleOCR-VL
```

- replace native backend to vllm server
```
SubModules:
    LayoutDetection:
        module_name: layout_detection
        model_name: PP-DocLayoutV2
        ...
    VLRecognition:
        ...
        # replaced part
        genai_config:
            backend: vllm-server
            server_url: http://127.0.0.1:8000/v1
```

- start paddlex service
```
paddlex --serve --pipeline ./PaddleOCR-VL.yaml
```

### Run the demo

<details>


<summary>1.Call vLLM Service</summary>
Need to use the specified format to use paddleocr.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": TASKS["ocr"]
            }
        ]
    }
]

response = client.chat.completions.create(
    model="PaddleOCR-VL-0.9B",
    messages=messages,
    temperature=0.0,
    max_tokens=128,
)
print(f"Generated text: {response.choices[0].message.content}")
```

</details>

<details open>
<summary>2.Call Paddle Service</summary>

```python
import requests
request_data = {
    "file": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
    "fileType": 1,
    "useLayoutDetection": True, # default value is True, used for layout_det_res
}

response = requests.post(
    url="http://localhost:8080/layout-parsing",
    json=request_data,
    timeout=3600
)

# print result from PaddleOCR-VL-0.9B
print(response.json()['result']['layoutParsingResults'][0]['markdown']['text'])

# print result from PP-DocLayoutV2
print(response.json()['result']['layoutParsingResults'][0]['prunedResult']['layout_det_res'])
```

</details>

<details >
<summary>3.Offline Paddle Service</summary>

Use PP-DocLayoutV2 model offline. Refer to [this](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2).

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict("./layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```
</details>

You can refer to the Paddler-OCR [guide](https://github.com/PaddlePaddle/PaddleOCR/blob/437943ff0d462a2e3abbc4a409074ebdbd2deafd/docs/version3.x/pipeline_usage/PaddleOCR-VL.md#43-%E5%AE%A2%E6%88%B7%E7%AB%AF%E8%B0%83%E7%94%A8%E6%96%B9%E5%BC%8F) for more details.

---

## 4. DeepSeek-OCR Support

You can refer to the DeepSeek-OCR [guide](https://github.com/deepseek-ai/DeepSeek-OCR?tab=readme-ov-file#vllm-inference) for details.

---

