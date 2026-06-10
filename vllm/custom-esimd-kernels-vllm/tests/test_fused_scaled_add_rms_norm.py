"""Numerical regression for esimd_fused_scaled_add_rms_norm.

Output: hidden_states = rmsnorm((hs+r)*scalar) * weight
Side effect: residual is overwritten in-place with (hs+r)*scalar.
"""
import sys, torch
import custom_esimd_kernels_vllm  # noqa: F401  (registers ops)


def py_ref(hs, residual, weight, eps, scalar):
    res = (hs.float() + residual.float()) * scalar
    var = (res * res).mean(dim=-1, keepdim=True)
    out = (res * torch.rsqrt(var + eps) * weight.float()).to(torch.float16)
    return out, res.to(torch.float16)


def run(K, scalar, tol=1e-2):
    torch.manual_seed(0)
    dev = torch.device("xpu")
    eps = 1e-6
    hs = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.5
    res = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.3
    w = (torch.randn(K, dtype=torch.float16, device=dev) * 0.1 + 1.0)

    out_ref, res_ref = py_ref(hs.clone(), res.clone(), w, eps, scalar)
    hs2, res2 = hs.clone(), res.clone()
    torch.ops.custom_esimd_kernels_vllm.esimd_fused_scaled_add_rms_norm(
        hs2, res2, w, eps, scalar)

    do = (hs2.float() - out_ref.float()).abs().max().item()
    dr = (res2.float() - res_ref.float()).abs().max().item()
    ok = do < tol and dr < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] K={K} scalar={scalar}  "
          f"out_diff={do:.2e}  res_diff={dr:.2e}")
    return ok


if __name__ == "__main__":
    cases = [
        (2816, 0.07, 1e-2),    # gemma4 hidden + small scalar
        (2816, 0.82, 1e-2),    # gemma4 hidden + larger scalar
        (2816, 1.0, 1e-2),     # match plain fused_add_rms_norm
    ]
    ok = all(run(*c) for c in cases)
    sys.exit(0 if ok else 1)
