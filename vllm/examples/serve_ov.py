#!/usr/bin/env python3
"""
OpenVINO GenAI server with OpenAI-compatible API for OpenClaw.

Designed for Intel iGPU (Meteor Lake / Lunar Lake) with 16-32 GB shared memory.
Uses ContinuousBatchingPipeline with chunked prefill and INT4/INT8 KV cache
to support up to 32K token input context.

Usage:
    # Basic (auto-downloads Qwen3-4B INT4):
    python serve_ov.py

    # Custom model path:
    python serve_ov.py --model /path/to/openvino/model

    # CPU fallback for very long contexts:
    python serve_ov.py --device CPU --cache-size 6

    # Minimal memory mode:
    python serve_ov.py --kv-precision u4 --cache-size 2 --chunk-size 256

OpenClaw config (~/.openclaw/openclaw.json):
    {"baseUrl": "http://127.0.0.1:8000/v1", "apiKey": "no-key", "api": "openai-completions"}

Requires: pip install openvino-genai fastapi uvicorn
"""

import argparse
import json
import time
import uuid
from typing import Generator

import openvino_genai as ov_genai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="OpenVINO GenAI Server")

# Global pipeline (initialized at startup)
pipe = None
model_name = ""


# --- Request/Response models ---

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = Field(default=512, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False


# --- Endpoints ---

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # Build prompt from messages using chat template
    prompt = build_chat_prompt(request.messages)

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = request.max_tokens
    gen_config.temperature = max(request.temperature, 0.01)
    gen_config.top_p = request.top_p

    if request.stream:
        return StreamingResponse(
            stream_chat_response(prompt, gen_config, request.model or model_name),
            media_type="text/event-stream",
        )

    start = time.time()
    result = pipe.generate(prompt, gen_config)
    duration = time.time() - start

    text = result if isinstance(result, str) else str(result)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


@app.post("/v1/completions")
def completions(request: CompletionRequest):
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = request.max_tokens
    gen_config.temperature = max(request.temperature, 0.01)
    gen_config.top_p = request.top_p

    if request.stream:
        return StreamingResponse(
            stream_completion_response(
                request.prompt, gen_config, request.model or model_name
            ),
            media_type="text/event-stream",
        )

    result = pipe.generate(request.prompt, gen_config)
    text = result if isinstance(result, str) else str(result)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model or model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
    }


# --- Streaming helpers ---

def stream_chat_response(
    prompt: str, gen_config: ov_genai.GenerationConfig, model: str
) -> Generator[str, None, None]:
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    tokens = []

    def streamer(token):
        tokens.append(token)
        return False

    gen_config.set_streamer(streamer)
    pipe.generate(prompt, gen_config)

    for token in tokens:
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def stream_completion_response(
    prompt: str, gen_config: ov_genai.GenerationConfig, model: str
) -> Generator[str, None, None]:
    resp_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    tokens = []

    def streamer(token):
        tokens.append(token)
        return False

    gen_config.set_streamer(streamer)
    pipe.generate(prompt, gen_config)

    for token in tokens:
        chunk = {
            "id": resp_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": token, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


# --- Chat template ---

def build_chat_prompt(messages: list[Message]) -> str:
    """Build a simple chat prompt. Models with chat_template will override this."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# --- Pipeline initialization ---

def create_pipeline(args) -> ov_genai.LLMPipeline:
    """Create OpenVINO GenAI pipeline with 32K-ready config."""
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.cache_size = args.cache_size
    scheduler_config.max_num_batched_tokens = args.chunk_size
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.enable_prefix_caching = False  # avoid bug openvino.genai#2406

    pipeline_kwargs = {
        "scheduler_config": scheduler_config,
    }

    # KV cache precision (u8 default on iGPU, u4 for max memory savings)
    if args.kv_precision:
        pipeline_kwargs["KV_CACHE_PRECISION"] = args.kv_precision

    print(f"Loading model from: {args.model}")
    print(f"Device: {args.device}")
    print(f"KV cache: {args.kv_precision or 'default (u8 on iGPU)'}")
    print(f"Cache size: {args.cache_size} GB")
    print(f"Chunk size: {args.chunk_size} tokens")

    pipeline = ov_genai.LLMPipeline(args.model, args.device, **pipeline_kwargs)

    print("Model loaded successfully.")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="OpenVINO GenAI API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="OpenVINO/Qwen3-4B-int4-ov",
        help="Model path or HuggingFace model ID (default: OpenVINO/Qwen3-4B-int4-ov)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["GPU", "CPU", "NPU"],
        help="Inference device (default: GPU)",
    )
    parser.add_argument(
        "--kv-precision",
        type=str,
        default="u8",
        choices=["u4", "u8", "f16", "bf16"],
        help="KV cache precision (default: u8, use u4 for 32K context on 16GB)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=3,
        help="KV cache size in GB (default: 3)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Prefill chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    args = parser.parse_args()

    global pipe, model_name
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    pipe = create_pipeline(args)

    import uvicorn

    print(f"\nServing at http://{args.host}:{args.port}/v1")
    print("Endpoints: /v1/models, /v1/chat/completions, /v1/completions")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
