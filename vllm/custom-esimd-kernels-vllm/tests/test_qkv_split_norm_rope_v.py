"""Numerical regression for esimd_qkv_split_norm_rope_v.

Kernel internally applies (weight + 1.0) RMSNorm on Q/K/V plus RoPE on
Q/K. To use it for a model whose RMSNorm is `w * x / rms(x)` (e.g. gemma4
sliding attention), the caller passes (w - 1.0) for Q/K and a zero
buffer for V (gemma4 V-Norm has_weight=False, effective weight=ones).

Compares against a pure-PyTorch reference at gemma-4-26B-A4B-it
sliding-attention shape (head_dim=256, GQA=2).
"""
import math, sys, torch
import custom_esimd_kernels_vllm  # noqa: F401  (registers ops)


def gemma_rms_norm(x, weight, eps):
    var = (x * x).mean(dim=-1, keepdim=True)
    return x * weight * torch.rsqrt(var + eps)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def py_ref(qkv, w_q, w_k, w_v, positions, cos_sin_cache, num_heads, num_kv_heads,
           head_dim, eps):
    n_tok = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = qkv[:, :q_size].reshape(n_tok, num_heads, head_dim)
    k = qkv[:, q_size:q_size + kv_size].reshape(n_tok, num_kv_heads, head_dim)
    v = qkv[:, q_size + kv_size:].reshape(n_tok, num_kv_heads, head_dim)
    q = gemma_rms_norm(q.float(), w_q.float(), eps).half()
    k = gemma_rms_norm(k.float(), w_k.float(), eps).half()
    v = gemma_rms_norm(v.float(), w_v.float(), eps).half()
    pos = positions.to(torch.long)
    half = head_dim // 2
    cos = cos_sin_cache[pos][:, :half]
    sin = cos_sin_cache[pos][:, half:]
    q = apply_rope(q.float(), cos.float()[:, None, :], sin.float()[:, None, :]).half()
    k = apply_rope(k.float(), cos.float()[:, None, :], sin.float()[:, None, :]).half()
    return (q.reshape(n_tok, q_size).contiguous(),
            k.reshape(n_tok, kv_size).contiguous(),
            v.reshape(n_tok, kv_size).contiguous())


def main():
    torch.manual_seed(0)
    dev = torch.device("xpu")
    NUM_HEADS, NUM_KV_HEADS = 8, 4    # gemma4 sliding layer at TP=2
    HEAD_DIM = 256
    EPS = 1e-6
    N_TOK = 4
    HIDDEN = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    MAX_POS = 1024

    qkv = torch.randn(N_TOK, HIDDEN, dtype=torch.float16, device=dev) * 0.1
    wq_g = torch.randn(HEAD_DIM, dtype=torch.float16, device=dev) * 0.05 + 1.0
    wk_g = torch.randn(HEAD_DIM, dtype=torch.float16, device=dev) * 0.05 + 1.0
    wv_g = torch.ones(HEAD_DIM, dtype=torch.float16, device=dev)

    pos_idx = torch.arange(MAX_POS, dtype=torch.float32, device=dev)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32,
                                              device=dev) / HEAD_DIM))
    freqs = pos_idx[:, None] * inv_freq[None, :]
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float16)
    positions = torch.tensor([0, 7, 100, 500], dtype=torch.int32, device=dev)

    q_ref, k_ref, v_ref = py_ref(qkv, wq_g, wk_g, wv_g, positions, cos_sin,
                                 NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, EPS)

    wq_kernel = (wq_g - 1.0).contiguous()
    wk_kernel = (wk_g - 1.0).contiguous()
    wv_kernel = torch.zeros(HEAD_DIM, dtype=torch.float16, device=dev)

    q_out = torch.empty(N_TOK, NUM_HEADS * HEAD_DIM, dtype=torch.float16, device=dev)
    g_out = torch.empty(N_TOK, NUM_HEADS * HEAD_DIM, dtype=torch.float16, device=dev)
    k_out = torch.empty(N_TOK, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float16, device=dev)
    v_out = torch.empty(N_TOK, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float16, device=dev)
    torch.ops.custom_esimd_kernels_vllm.esimd_qkv_split_norm_rope_v(
        qkv, q_out, g_out, k_out, v_out,
        wq_kernel, wk_kernel, wv_kernel, positions,
        NUM_HEADS, NUM_KV_HEADS, False, HEAD_DIM, cos_sin,
    )

    ok = True
    for name, a, b in (("Q", q_out, q_ref), ("K", k_out, k_ref), ("V", v_out, v_ref)):
        d = (a.float() - b.float()).abs().max().item()
        good = d < 1e-2
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}  max_abs_diff={d:.4e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
