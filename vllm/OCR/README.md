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

Need to use the specified format to use paddleocr.
```bash
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8002/v1",
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
    model="PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
    max_tokens=128,
)
print(f"Generated text: {response.choices[0].message.content}")
```

You can refer to the Paddler-OCR [guide](https://www.paddleocr.ai/main/) for more details.

---

## 4. DeepSeek-OCR Support

You can refer to the DeepSeek-OCR [guide](https://github.com/deepseek-ai/DeepSeek-OCR?tab=readme-ov-file#vllm-inference) for details.

---

