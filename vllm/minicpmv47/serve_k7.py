# SPDX-License-Identifier: Apache-2.0
"""Start MiniCPM-V 4.7 with a BF16 DSpark draft and seven speculative tokens."""

import argparse
import json
import shlex
import subprocess
import sys


def command(arguments):
    compilation = {
        "mode": 3,
        "backend": "inductor",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "pass_config": {
            "fuse_xpu_moe_shared": True,
            "fuse_xpu_qkv_norm_rope": True,
            "fuse_xpu_fp8_gemm_pair": True,
            "fuse_xpu_norm_fp8_gemm": True,
            "xpu_gdn_output_alloc": True,
            "xpu_inplace_all_reduce": True,
        },
        "cudagraph_capture_sizes": [1, 8],
        "max_cudagraph_capture_size": 8,
    }
    speculation = {
        "method": "dspark",
        "model": arguments.draft_model,
        "num_speculative_tokens": 7,
        "draft_tensor_parallel_size": arguments.tensor_parallel_size,
        "enable_adaptive_verification": False,
    }
    result = [
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
        "bfloat16",
        "--quantization",
        "fp8",
        "--max-model-len",
        str(arguments.max_model_len),
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(arguments.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}),
        "--optimization-level",
        "2",
        "--compilation-config",
        json.dumps(compilation),
        "--speculative-config",
        json.dumps(speculation),
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--served-model-name", default="MiniCPM-V-4.7")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    invocation = command(arguments)
    print(shlex.join(invocation), flush=True)
    if not arguments.dry_run:
        raise SystemExit(subprocess.call(invocation))
