# SPDX-License-Identifier: Apache-2.0
"""Serve MiniCPM-V 4.7 with compiler/graph acceleration and no draft model."""

import argparse
import json
import os
import shlex
import subprocess
import sys


def command(arguments):
    compilation = {
        "mode": 3,
        "backend": "inductor",
        "splitting_ops": [],
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "pass_config": {
            "fuse_xpu_moe_shared": True,
            "fuse_xpu_qkv_norm_rope": True,
            "fuse_xpu_fp8_gemm_pair": True,
            "fuse_xpu_norm_fp8_gemm": True,
            "xpu_gdn_output_alloc": True,
            "xpu_inplace_all_reduce": True,
        },
        "cudagraph_capture_sizes": [1],
        "max_cudagraph_capture_size": 1,
    }
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        arguments.model,
        "--served-model-name",
        arguments.served_model_name,
        "--trust-remote-code",
        "--model-impl",
        "vllm",
        "--tensor-parallel-size",
        str(arguments.tensor_parallel_size),
        "--dtype",
        arguments.dtype,
        "--quantization",
        "fp8",
        "--max-model-len",
        str(arguments.max_model_len),
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "1",
        "--kv-cache-memory-bytes",
        str(arguments.kv_cache_memory_bytes),
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}),
        "--optimization-level",
        "2",
        "--compilation-config",
        json.dumps(compilation),
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="MiniCPM-V-4.7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--max-model-len", type=int, default=33280)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--custom-all-reduce", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.custom_all_reduce and arguments.tensor_parallel_size != 2:
        parser.error("--custom-all-reduce requires --tensor-parallel-size=2")
    os.environ["MINICPMV47_MAIN_DECODE"] = "1"
    os.environ["VLLM_XPU_USE_CUSTOM_ALLREDUCE"] = (
        "1" if arguments.custom_all_reduce else "0"
    )
    invocation = command(arguments)
    print(shlex.join(invocation), flush=True)
    if not arguments.dry_run:
        raise SystemExit(subprocess.call(invocation))
