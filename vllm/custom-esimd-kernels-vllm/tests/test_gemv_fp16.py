"""Numerical regression for esimd_gemv_fp16 at the gemma4 router shape."""
import sys, torch
import custom_esimd_kernels_vllm  # noqa: F401  (registers ops)


def run(N, K, tol=5e-2):
    torch.manual_seed(0)
    dev = torch.device("xpu")
    x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.1
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
    out = torch.empty(1, N, dtype=torch.float16, device=dev)
    torch.ops.custom_esimd_kernels_vllm.esimd_gemv_fp16(x, w, out)
    ref = torch.nn.functional.linear(x, w)
    diff = (out.float() - ref.float()).abs().max().item()
    ok = diff < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] N={N} K={K}  max_abs={diff:.3e}")
    return ok


if __name__ == "__main__":
    cases = [
        (128, 2816),    # gemma4-26B router (TP=2)
        (128, 1408),    # gemma4-26B router (TP=4 hypothetical)
        (256, 2048),    # generic
    ]
    sys.exit(0 if all(run(*c) for c in cases) else 1)
