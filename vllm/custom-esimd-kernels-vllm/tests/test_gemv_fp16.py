"""Numerical regression for the FP16 GEMV dispatch and small-K safety."""
import os
import sys

import torch
import torch.nn.functional as F

# Build-only regression runs can load an explicitly fingerprinted DSO without
# importing the production package (which may contain a different artifact).
_gemv_dso = os.environ.get("GEMV_FP16_DSO")
if _gemv_dso:
    torch.ops.load_library(_gemv_dso)
else:
    import custom_esimd_kernels_vllm  # noqa: F401  (registers ops)


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


def run_hc_down(N, tol=5e-2):
    torch.manual_seed(37 + N)
    dev = torch.device("xpu")
    K = 10240
    x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.02
    out = torch.empty(1, N, dtype=torch.float16, device=dev)
    pointer = out.data_ptr()
    result = torch.ops.custom_esimd_kernels_vllm.esimd_hc_down_fp16_out(
        x, w, out
    )
    linear = F.linear(x, w)
    ref = linear.clone()
    ref[:, :320] = F.silu(linear[:, :320] / 4)
    max_abs = (out.float() - ref.float()).abs().max().item()
    tail_abs = (
        (out[:, 320:].float() - linear[:, 320:].float()).abs().max().item()
        if N == 336
        else 0.0
    )
    ok = (
        result is None
        and out.data_ptr() == pointer
        and max_abs < tol
        and tail_abs < tol
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] HC down N={N} K={K}  "
        f"max_abs={max_abs:.3e} tail_abs={tail_abs:.3e}"
    )
    return ok


def rejects_hc(label, x, w, out):
    try:
        torch.ops.custom_esimd_kernels_vllm.esimd_hc_down_fp16_out(x, w, out)
    except RuntimeError:
        print(f"  [PASS] HC down rejects {label}")
        return True
    print(f"  [FAIL] HC down accepts invalid {label}")
    return False


if __name__ == "__main__":
    cases = [
        # Small and non-aligned K values exercise every scalar-safe tail width.
        (1, 3, 1),
        (1, 4, 2),
        (2, 5, 6),
        (1, 6, 7),
        (1, 4, 8),
        (2, 5, 16),
        (3, 7, 24),
        (1, 4, 33),
        (1, 8, 65),
        (1, 128, 2052),
        # Production-shaped and split-dispatch regressions.
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
        rejects(
            "zero K dimension",
            torch.empty(1, 0, dtype=torch.float16, device=dev),
            torch.empty(3, 0, dtype=torch.float16, device=dev),
            torch.empty(1, 3, dtype=torch.float16, device=dev),
        ),
    ]

    hc_n = 336
    hc_k = 10240
    hc_x = torch.randn(1, hc_k, dtype=torch.float16, device=dev)
    hc_w = torch.randn(hc_n, hc_k, dtype=torch.float16, device=dev)
    hc_out = torch.empty(1, hc_n, dtype=torch.float16, device=dev)
    odd_x = torch.randn(1, hc_k + 1, dtype=torch.float16, device=dev)[:, 1:]
    odd_w = torch.randn(
        hc_n * hc_k + 1, dtype=torch.float16, device=dev
    )[1:].view(hc_n, hc_k)
    odd_out = torch.empty(
        hc_n + 1, dtype=torch.float16, device=dev
    )[1:].view(1, hc_n)
    hc_invalid_cases = [
        rejects_hc("bf16 input", hc_x.to(torch.bfloat16), hc_w, hc_out),
        rejects_hc(
            "M=2",
            torch.randn(2, hc_k, dtype=torch.float16, device=dev),
            hc_w,
            torch.empty(2, hc_n, dtype=torch.float16, device=dev),
        ),
        rejects_hc(
            "N=321",
            hc_x,
            torch.randn(321, hc_k, dtype=torch.float16, device=dev),
            torch.empty(1, 321, dtype=torch.float16, device=dev),
        ),
        rejects_hc(
            "K=10239",
            torch.randn(1, hc_k - 1, dtype=torch.float16, device=dev),
            torch.randn(hc_n, hc_k - 1, dtype=torch.float16, device=dev),
            hc_out,
        ),
        rejects_hc(
            "non-contiguous input",
            torch.randn(1, hc_k * 2, dtype=torch.float16, device=dev)[:, ::2],
            hc_w,
            hc_out,
        ),
        rejects_hc(
            "non-contiguous weight",
            hc_x,
            torch.randn(hc_k, hc_n, dtype=torch.float16, device=dev).t(),
            hc_out,
        ),
        rejects_hc(
            "non-contiguous output",
            hc_x,
            hc_w,
            torch.empty(1, hc_n * 2, dtype=torch.float16, device=dev)[:, ::2],
        ),
        rejects_hc("odd input offset", odd_x, hc_w, hc_out),
        rejects_hc("odd weight offset", hc_x, odd_w, hc_out),
        rejects_hc("odd output offset", hc_x, hc_w, odd_out),
        rejects_hc("input alias", hc_x, hc_w, hc_x[:, :hc_n]),
        rejects_hc(
            "weight alias",
            hc_x,
            hc_w,
            hc_w.view(-1)[:hc_n].view(1, hc_n),
        ),
    ]
    ok = (
        all(run(*case) for case in cases)
        and all(invalid_cases)
        and run_hc_down(320)
        and run_hc_down(336)
        and all(hc_invalid_cases)
    )
    sys.exit(0 if ok else 1)
