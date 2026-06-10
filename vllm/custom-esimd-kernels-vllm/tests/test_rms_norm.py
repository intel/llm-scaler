"""Numerical regression for esimd_rms_norm."""
import sys, torch
import custom_esimd_kernels_vllm  # noqa


def py_ref(x, w, eps):
    var = (x.float() * x.float()).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * w.float()).to(torch.float16)


def run(K, tol=1e-2):
    torch.manual_seed(0)
    dev = torch.device("xpu")
    eps = 1e-6
    x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.5
    w = (torch.randn(K, dtype=torch.float16, device=dev) * 0.1 + 1.0)
    out = torch.empty(1, K, dtype=torch.float16, device=dev)
    torch.ops.custom_esimd_kernels_vllm.esimd_rms_norm(x, out, w, eps)
    ref = py_ref(x, w, eps)
    diff = (out.float() - ref.float()).abs().max().item()
    ok = diff < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] K={K}  max_abs={diff:.3e}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all(run(K) for K in [2816, 2048, 1408, 4096]) else 1)
