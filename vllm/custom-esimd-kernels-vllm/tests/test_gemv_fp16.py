"""Numerical regression for esimd_gemv_fp16 at the gemma4 router shape."""
import sys

import custom_esimd_kernels_vllm  # noqa: F401  (registers ops)
import torch


def run(M, N, K, tol=5e-2):
    torch.manual_seed(0)
    dev = torch.device("xpu")
    x = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.1
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
    out = torch.empty(M, N, dtype=torch.float16, device=dev)
    torch.ops.custom_esimd_kernels_vllm.esimd_gemv_fp16(x, w, out)
    ref = torch.nn.functional.linear(x, w)
    diff = (out.float() - ref.float()).abs().max().item()
    ok = diff < tol
    print(
        f"  [{'PASS' if ok else 'FAIL'}] M={M} N={N} K={K}  "
        f"max_abs={diff:.3e}"
    )
    return ok


def rejects(label, x, w, out):
    try:
        torch.ops.custom_esimd_kernels_vllm.esimd_gemv_fp16(x, w, out)
    except RuntimeError:
        print(f"  [PASS] rejects {label}")
        return True
    print(f"  [FAIL] accepts invalid {label}")
    return False


if __name__ == "__main__":
    cases = [
        (1, 128, 2816),  # gemma4-26B router (TP=2)
        (1, 128, 1408),  # gemma4-26B router (TP=4 hypothetical)
        (2, 32, 2048),   # Qwen3.6 GDN in_proj_ba
        (4, 32, 2048),
        (8, 32, 2048),
        (2, 256, 2048),  # Qwen3.6 MoE router
        (4, 256, 2048),
        (8, 256, 2048),
    ]
    torch.manual_seed(0)
    dev = torch.device("xpu")
    M, N, K = 2, 32, 2048
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    w = torch.randn(N, K, dtype=torch.float16, device=dev)
    out = torch.empty(M, N, dtype=torch.float16, device=dev)
    invalid_cases = [
        rejects("bf16 input", x.to(torch.bfloat16), w, out),
        rejects(
            "non-contiguous input",
            torch.randn(K, M, dtype=torch.float16, device=dev).t(),
            w,
            out,
        ),
        rejects(
            "wrong output shape",
            x,
            w,
            torch.empty(M, N - 1, dtype=torch.float16, device=dev),
        ),
    ]
    sys.exit(0 if all(run(*c) for c in cases) and all(invalid_cases) else 1)
