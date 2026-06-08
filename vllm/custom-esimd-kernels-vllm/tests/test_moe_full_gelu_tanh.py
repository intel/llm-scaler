"""Numerical correctness for moe_forward_full_gelu_tanh with vllm FusedMoE
weight layout: w13 [E, 2*inter, hidden], w2 [E, hidden, inter] fp8_e4m3fn.

Reference uses fp16 dequant + matmul; tolerates fp8 quant noise (~1e-2).
"""
import math, sys, torch
import custom_esimd_kernels_vllm  # noqa: F401  (registers torch.ops.moe_ops)


def gelu_tanh(t):
    return 0.5 * t * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (t + 0.044715 * t ** 3)))


def ref_moe(x, logits, w13_fp8, w2_fp8, gu_s, dn_s, top_k, inter):
    probs = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.float16)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(top_k):
            eid = indices[t, k].item()
            w13 = w13_fp8[eid].to(torch.float16) * float(gu_s[eid])
            w2 = w2_fp8[eid].to(torch.float16) * float(dn_s[eid])
            gu = w13 @ x[t]
            gate, up = gu.split(inter)
            mid = gelu_tanh(gate.float()).to(torch.float16) * up
            out[t] += (w2 @ mid) * weights[t, k]
    return out


def run(n_experts, top_k, hidden, inter, tokens, seed, tol):
    torch.manual_seed(seed)
    dev = torch.device("xpu")
    x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev) * 0.05
    logits = torch.randn(tokens, n_experts, dtype=torch.float16, device=dev)
    sc = 0.3
    w13 = (torch.randn(n_experts, 2 * inter, hidden, dtype=torch.float32, device=dev) * 0.05
           ).clamp(-sc, sc).div(sc).to(torch.float8_e4m3fn)
    w2 = (torch.randn(n_experts, hidden, inter, dtype=torch.float32, device=dev) * 0.05
          ).clamp(-sc, sc).div(sc).to(torch.float8_e4m3fn)
    gu_s = torch.full((n_experts,), sc, dtype=torch.float32, device=dev)
    dn_s = torch.full((n_experts,), sc, dtype=torch.float32, device=dev)

    r = ref_moe(x, logits, w13, w2, gu_s, dn_s, top_k, inter)
    o = torch.ops.moe_ops.moe_forward_full_gelu_tanh(
        x, logits, w13, gu_s, w2, dn_s, top_k, n_experts)
    diff = (o.float() - r.float()).abs().max().item()
    label = f"E={n_experts} K={top_k} H={hidden} I={inter} T={tokens}"
    ok = diff < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  max_abs_diff={diff:.2e} (tol={tol:.0e})")
    return ok


if __name__ == "__main__":
    # gemma4 real shape (E=128, top_k=8, hidden=2816, inter=704).
    # Note: kernel reuses internal buffers across calls keyed by n_tokens only,
    # so a single shape per test process is the safe pattern.
    ok = run(128, 8, 2816, 704, 2, 42, 2e-2)
    sys.exit(0 if ok else 1)
