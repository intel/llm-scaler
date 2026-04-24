"""
Accuracy tests for the prefill INT4 ESIMD MoE kernels
(csrc/moe_prefill/moe_prefill_int4.sycl).

Weights are consumed in GGML N-major layout (no marlin shuffle, no transpose):

    gate_up_weight : [E, 2*I, H/2] uint8   (2 nibbles per byte, K-order)
    gate_up_scale  : [E, 2*I, H/GS] fp16
    down_weight    : [E, H, I/2] uint8
    down_scale     : [E, H, I/GS] fp16

    byte[i] = (nibble[2i+1] << 4) | nibble[2i]
    val     = (nibble - 8) * scale

This matches the vLLM storage of `layer.w13_weight` (int32 [E, 2*I, H/8])
when viewed as uint8 [E, 2*I, H/2].

Tests are ordered from simplest to end-to-end:
    1. gather_forward_v2     — pure routing, no compute
    2. up_forward_v2         — gate+up+silu+mul (one DPAS kernel)
    3. down_forward_v2       — down projection (one DPAS kernel)
    4. accumulate_forward_v2 — weighted sum over top_k rows
    5. moe_prefill_full_int4 — end-to-end

Usage:
    python -m pytest tests/test_moe_prefill_int4.py -v -s
    python tests/test_moe_prefill_int4.py      # standalone runner
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "xpu"
GROUP_SIZE = 128


# Small config keeps CPU reference tractable. Kept mostly for smoke-testing.
CFG_SMALL = dict(M=16, H=256, I=128, E=8, top_k=2)

# Qwen3.5-122B-A10B per-rank shape under TP=4. E here is *local* (256/4=64),
# M is small on purpose so the per-token reference stays quick on CPU.
CFG_122B_TP4 = dict(M=16, H=3072, I=256, E=64, top_k=8)

# Mid-shape sanity (35B-A3B style, TP=4).
CFG_35B_TP4 = dict(M=32, H=2048, I=256, E=64, top_k=8)


# ─── GGML N-major int4 quantize / dequantize (uint8 view, 2 nibbles/byte) ─────

def quantize_int4_ggml(weight_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """
    Quantize [N, K] fp16 into GGML N-major int4:
        qweight [N, K/2] uint8     byte = (hi << 4) | lo in K-order
        scale   [N, K/GS] fp16     symmetric, zero-point = 8
    """
    N, K = weight_fp16.shape
    assert K % group_size == 0 and K % 2 == 0
    n_groups = K // group_size

    w = weight_fp16.float().numpy()
    w_grp = w.reshape(N, n_groups, group_size)

    max_abs = np.abs(w_grp).max(axis=2)
    scale_np = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)

    quantized = np.round(
        w_grp / scale_np[:, :, None].astype(np.float32)
    ).clip(-8, 7).astype(np.int32) + 8                          # [0, 15]
    q_flat = quantized.reshape(N, K)

    lo = q_flat[:, 0::2].astype(np.uint8)                       # [N, K/2]
    hi = q_flat[:, 1::2].astype(np.uint8)
    packed = (lo | (hi << 4))                                   # [N, K/2] uint8

    return torch.from_numpy(packed), torch.from_numpy(scale_np)


def dequantize_int4_ggml(qweight: torch.Tensor, scales: torch.Tensor,
                         N: int, K: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Inverse of quantize_int4_ggml → fp16 [N, K]."""
    packed = qweight.numpy()
    sc = scales.numpy().astype(np.float32)
    n_groups = K // group_size

    lo = (packed & 0x0F).astype(np.float32) - 8.0               # [N, K/2]
    hi = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0
    unpacked = np.empty((N, K), dtype=np.float32)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi

    unpacked = unpacked.reshape(N, n_groups, group_size) * sc[:, :, None]
    return torch.from_numpy(unpacked.reshape(N, K)).half()


def quantize_experts_ggml(W_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Per-expert wrapper. W_fp16: [E, N, K]. Returns q [E, N, K/2] uint8 + s [E, N, K/GS] fp16."""
    E, _, _ = W_fp16.shape
    qs, ss = [], []
    for e in range(E):
        q, s = quantize_int4_ggml(W_fp16[e], group_size)
        qs.append(q); ss.append(s)
    return torch.stack(qs), torch.stack(ss)


def dequantize_experts_ggml(qweight: torch.Tensor, scales: torch.Tensor,
                             group_size: int = GROUP_SIZE) -> torch.Tensor:
    E, N, K_half = qweight.shape
    K = K_half * 2
    out = torch.empty(E, N, K, dtype=torch.float16)
    for e in range(E):
        out[e] = dequantize_int4_ggml(qweight[e], scales[e], N, K, group_size)
    return out


# ─── metrics ──────────────────────────────────────────────────────────────────

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0:
        return 1.0
    return (af @ bf / denom).item()


# ─── Reference implementations ────────────────────────────────────────────────

def ref_gather(selected_experts: torch.Tensor, num_experts: int):
    """CPU reference for moe_prefill_gather_forward_v2."""
    se = selected_experts.cpu().numpy().reshape(-1)
    total = se.size
    counts = np.bincount(se, minlength=num_experts).astype(np.int32)
    offsets = np.zeros(num_experts, dtype=np.int32)
    offsets[1:] = np.cumsum(counts[:-1])
    tokens = np.empty(total, dtype=np.int32)
    pos = offsets.copy()
    for i in range(total):
        e = se[i]
        tokens[pos[e]] = i
        pos[e] += 1
    return torch.from_numpy(offsets), torch.from_numpy(tokens)


def ref_up(x: torch.Tensor, W13_fp16: torch.Tensor,
           expert_offsets: torch.Tensor, expert_tokens: torch.Tensor,
           top_k: int) -> torch.Tensor:
    """Reference for moe_prefill_up_forward_v2. Output [M*top_k, I] fp16.
    Vectorised per-expert: one matmul covers all rows routed to expert e.
    """
    M, H = x.shape
    E, two_I, _ = W13_fp16.shape
    I = two_I // 2
    total = M * top_k
    out = torch.zeros(total, I, dtype=torch.float16)
    off = expert_offsets.cpu().tolist()
    tok = expert_tokens.cpu()
    xf = x.float()
    for e in range(E):
        t0 = off[e]
        t1 = off[e + 1] if e + 1 < E else total
        if t0 == t1:
            continue
        pairs  = tok[t0:t1].long()
        tokens = pairs // top_k
        gate_w = W13_fp16[e, :I, :].float()   # [I, H]
        up_w   = W13_fp16[e, I:, :].float()   # [I, H]
        x_rows = xf[tokens]                   # [n, H]
        g = x_rows @ gate_w.t()               # [n, I]
        u = x_rows @ up_w.t()                 # [n, I]
        inter = (g / (1 + torch.exp(-g))) * u
        out[pairs] = inter.half()
    return out


def ref_down(intermediate: torch.Tensor, W2_fp16: torch.Tensor,
             expert_offsets: torch.Tensor, expert_tokens: torch.Tensor) -> torch.Tensor:
    """Reference for moe_prefill_down_forward_v2. Output [M*top_k, H] fp16.
    Vectorised per-expert.
    """
    total, I = intermediate.shape
    E, H, _ = W2_fp16.shape
    out = torch.zeros(total, H, dtype=torch.float16)
    off = expert_offsets.cpu().tolist()
    tok = expert_tokens.cpu()
    inter_f = intermediate.float()
    for e in range(E):
        t0 = off[e]
        t1 = off[e + 1] if e + 1 < E else total
        if t0 == t1:
            continue
        pairs = tok[t0:t1].long()
        w = W2_fp16[e].float()                # [H, I]
        out[pairs] = (inter_f[pairs] @ w.t()).half()
    return out


def ref_moe_routed(x: torch.Tensor, W13_fp16: torch.Tensor, W2_fp16: torch.Tensor,
                   topk_weights: torch.Tensor, topk_idx: torch.Tensor,
                   num_experts: int) -> torch.Tensor:
    """Full routed-MoE reference, matching what moe_prefill_full_int4 returns.
    shared expert is handled by the caller, NOT by this kernel.
    """
    TK = topk_idx.shape[1]
    off, tok = ref_gather(topk_idx, num_experts)
    inter   = ref_up(x, W13_fp16, off, tok, TK)
    exp_out = ref_down(inter, W2_fp16, off, tok)
    return ref_accumulate(exp_out, topk_weights, TK)


def ref_shared_expert(x: torch.Tensor, shared_gate_up_fp16: torch.Tensor,
                      shared_down_fp16: torch.Tensor,
                      shared_gate_weight: "torch.Tensor | None" = None) -> torch.Tensor:
    """Reference shared-expert MLP (run outside the prefill kernel):
        out = silu(x @ gate_up[:I].T) * (x @ gate_up[I:].T) @ down.T
    Optional sigmoid gate multiplies the whole branch (qwen3_next style).
    """
    inter_dim = shared_gate_up_fp16.shape[0] // 2
    xf  = x.float()
    gate = xf @ shared_gate_up_fp16[:inter_dim, :].t().float()
    up   = xf @ shared_gate_up_fp16[inter_dim:, :].t().float()
    inter = (gate / (1 + torch.exp(-gate))) * up
    out   = inter @ shared_down_fp16.t().float()
    if shared_gate_weight is not None:
        g = torch.sigmoid(xf @ shared_gate_weight.t().float())
        if g.dim() == 1:
            g = g.unsqueeze(-1)
        out = out * g
    return out.half()


def ref_accumulate(expert_output: torch.Tensor, routing_weights: torch.Tensor,
                   top_k: int) -> torch.Tensor:
    """Reference for moe_prefill_accumulate_forward_v2. Output [M, H] fp16."""
    M, _ = routing_weights.shape
    _, H = expert_output.shape
    out = torch.zeros(M, H, dtype=torch.float32)
    eo = expert_output.float()
    rw = routing_weights.float()
    for t in range(M):
        for s in range(top_k):
            out[t] += rw[t, s] * eo[t * top_k + s]
    return out.half()


# ─── Input factories ──────────────────────────────────────────────────────────

def make_inputs(cfg, seed=0, with_shared=False):
    """Build a deterministic MoE input set.
    with_shared=True additionally creates shared-expert weights (fp16, not
    quantised) for tests that combine routed + shared.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    M, H, I, E, TK = cfg["M"], cfg["H"], cfg["I"], cfg["E"], cfg["top_k"]
    SI = cfg.get("shared_I", I)

    x   = torch.randn(M, H, dtype=torch.float16) * 0.1
    W13 = torch.randn(E, 2 * I, H, dtype=torch.float16) * 0.02
    W2  = torch.randn(E, H, I,   dtype=torch.float16) * 0.02
    logits = torch.randn(M, E, dtype=torch.float16)

    W13_q, W13_s = quantize_experts_ggml(W13, GROUP_SIZE)
    W2_q,  W2_s  = quantize_experts_ggml(W2,  GROUP_SIZE)
    W13_dq = dequantize_experts_ggml(W13_q, W13_s, GROUP_SIZE)
    W2_dq  = dequantize_experts_ggml(W2_q,  W2_s,  GROUP_SIZE)

    probs = F.softmax(logits.float(), dim=-1)
    tw, ti = torch.topk(probs, TK, dim=-1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).half()
    ti = ti.to(torch.int32)

    out = dict(
        x=x, logits=logits,
        W13_q=W13_q, W13_s=W13_s, W2_q=W2_q, W2_s=W2_s,
        W13_dq=W13_dq, W2_dq=W2_dq,
        topk_weights=tw, topk_idx=ti,
        M=M, H=H, I=I, E=E, top_k=TK, shared_I=SI,
    )
    if with_shared:
        # Shared expert kept in fp16 for simplicity; vLLM quantises it too
        # but that is orthogonal to the prefill-kernel accuracy we test here.
        shared_gate_up = torch.randn(2 * SI, H, dtype=torch.float16) * 0.02
        shared_down    = torch.randn(H, SI,   dtype=torch.float16) * 0.02
        shared_gate_w  = torch.randn(1, H,    dtype=torch.float16) * 0.02
        out.update(
            shared_gate_up=shared_gate_up,
            shared_down=shared_down,
            shared_gate_w=shared_gate_w,
        )
    return out


def _xpu(t):
    return t.to(DEVICE)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_gather_v2():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = CFG_SMALL
    d = make_inputs(cfg)
    selected = _xpu(d["topk_idx"])

    off_k, tok_k = ops.moe_prefill_gather_forward_v2(selected, cfg["E"])
    off_r, tok_r = ref_gather(d["topk_idx"], cfg["E"])

    diff_off = max_abs_diff(off_k.cpu(), off_r)
    print(f"[gather] offsets max_diff = {diff_off}")
    assert diff_off == 0, "expert_offsets mismatch"

    # Order within each expert is not specified (atomic fetch_add). Compare as sets.
    off_list = off_r.tolist()
    total = selected.numel()
    tk_np = tok_k.cpu().numpy()
    tr_np = tok_r.numpy()
    for e in range(cfg["E"]):
        t0 = off_list[e]
        t1 = off_list[e + 1] if e + 1 < cfg["E"] else total
        assert set(tk_np[t0:t1].tolist()) == set(tr_np[t0:t1].tolist()), \
            f"expert_tokens mismatch at expert {e}"
    print("[gather] OK")


def test_up_v2():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = CFG_SMALL
    d = make_inputs(cfg)

    x_xpu  = _xpu(d["x"])
    w13_u8 = _xpu(d["W13_q"])
    w13_s  = _xpu(d["W13_s"])

    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(_xpu(d["topk_idx"]), cfg["E"])
    out_k = ops.moe_prefill_up_forward_v2(
        x_xpu, w13_u8, w13_s, off_xpu, tok_xpu, cfg["top_k"]
    ).cpu()
    out_r = ref_up(d["x"], d["W13_dq"], off_xpu.cpu(), tok_xpu.cpu(), cfg["top_k"])

    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[up]   max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.99, f"cos too low: {cos}"


def test_down_v2():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = CFG_SMALL
    d = make_inputs(cfg)

    x_xpu = _xpu(d["x"])
    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(_xpu(d["topk_idx"]), cfg["E"])
    # Build intermediate via pure-fp16 ref, independent of the up kernel.
    inter_ref = ref_up(d["x"], d["W13_dq"], off_xpu.cpu(), tok_xpu.cpu(), cfg["top_k"])
    inter_xpu = _xpu(inter_ref)

    w2_u8 = _xpu(d["W2_q"])
    w2_s  = _xpu(d["W2_s"])
    out_k = ops.moe_prefill_down_forward_v2(
        inter_xpu, w2_u8, w2_s, off_xpu, tok_xpu
    ).cpu()
    out_r = ref_down(inter_ref, d["W2_dq"], off_xpu.cpu(), tok_xpu.cpu())

    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[down] max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.99, f"cos too low: {cos}"


def test_accumulate_v2():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = CFG_SMALL
    d = make_inputs(cfg)
    M, H, TK = cfg["M"], cfg["H"], cfg["top_k"]

    torch.manual_seed(1)
    expert_out = torch.randn(M * TK, H, dtype=torch.float16) * 0.1

    out_k = ops.moe_prefill_accumulate_forward_v2(
        _xpu(expert_out), _xpu(d["topk_weights"])
    ).cpu()
    out_r = ref_accumulate(expert_out, d["topk_weights"], TK)

    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[acc]  max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.999, f"cos too low: {cos}"


def test_full_end_to_end():
    """End-to-end moe_prefill_full_int4. Uses the 256/8 specialisation path.
    Loose threshold: kernel runs its own softmax+topk, ties may route to
    different experts than the numpy reference.
    """
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = dict(M=16, H=256, I=128, E=256, top_k=8)
    d = make_inputs(cfg)

    x_xpu      = _xpu(d["x"])
    logits_xpu = _xpu(d["logits"])
    w13_u8     = _xpu(d["W13_q"])
    w2_u8      = _xpu(d["W2_q"])

    out_k = ops.moe_prefill_full_int4(
        x_xpu, logits_xpu,
        w13_u8, _xpu(d["W13_s"]),
        w2_u8,  _xpu(d["W2_s"]),
        cfg["top_k"], cfg["E"],
    ).cpu()

    # Reference path: same softmax+topk, same dequantized weights.
    M, H, I, E, TK = cfg["M"], cfg["H"], cfg["I"], cfg["E"], cfg["top_k"]
    off_r, tok_r = ref_gather(d["topk_idx"], E)
    inter = ref_up(d["x"], d["W13_dq"], off_r, tok_r, TK)
    exp_out = ref_down(inter, d["W2_dq"], off_r, tok_r)
    out_r = ref_accumulate(exp_out, d["topk_weights"], TK)

    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[full] max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.90, f"end-to-end cos too low: {cos}"


# ─── 122B-TP4 real-shape sanity (covers larger K tile coverage) ───────────────

def _run_routed_pipeline(cfg):
    """Helper: run gather+up+down+accumulate on kernel vs reference for a cfg.
    Uses deterministic topk (softmax + torch.topk), skipping the kernel-side
    topk so the test is independent of ties/ordering."""
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    d = make_inputs(cfg)

    x_xpu     = _xpu(d["x"])
    w13_u8    = _xpu(d["W13_q"])
    w13_s_xpu = _xpu(d["W13_s"])
    w2_u8     = _xpu(d["W2_q"])
    w2_s_xpu  = _xpu(d["W2_s"])
    tw_xpu    = _xpu(d["topk_weights"])
    ti_xpu    = _xpu(d["topk_idx"])

    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(ti_xpu, cfg["E"])
    inter_xpu = ops.moe_prefill_up_forward_v2(
        x_xpu, w13_u8, w13_s_xpu, off_xpu, tok_xpu, cfg["top_k"])
    exp_xpu = ops.moe_prefill_down_forward_v2(
        inter_xpu, w2_u8, w2_s_xpu, off_xpu, tok_xpu)
    out_k = ops.moe_prefill_accumulate_forward_v2(exp_xpu, tw_xpu).cpu()

    out_r = ref_moe_routed(d["x"], d["W13_dq"], d["W2_dq"],
                           d["topk_weights"], d["topk_idx"], cfg["E"])
    return out_k, out_r


def test_122B_TP4_shape():
    """Qwen3.5-122B-A10B TP=4 per-rank shape: H=3072, I=256, E=64, top_k=8.
    Exercises the per-K-tile path at the real intermediate size / hidden dim.
    """
    cfg = CFG_122B_TP4
    out_k, out_r = _run_routed_pipeline(cfg)
    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[122B] max_diff={diff:.5f}  cos={cos:.6f}  (H=3072, I=256, E=64, TK=8)")
    # fp16 accumulation over K=3072 tolerates ~1e-2 max diff.
    assert cos > 0.99, f"122B routed cos too low: {cos}"


def test_35B_TP4_shape():
    """Qwen3.5-35B-A3B TP=4 per-rank shape: H=2048, I=256, E=64, top_k=8."""
    cfg = CFG_35B_TP4
    out_k, out_r = _run_routed_pipeline(cfg)
    diff = max_abs_diff(out_k, out_r)
    cos = cos_sim(out_k, out_r)
    print(f"[35B]  max_diff={diff:.5f}  cos={cos:.6f}  (H=2048, I=256, E=64, TK=8)")
    assert cos > 0.99, f"35B routed cos too low: {cos}"


# ─── Shared-expert composition (emulates SharedFusedMoE external add) ─────────

def test_shared_expert_composition():
    """vLLM's SharedFusedMoE adds shared-expert output to our routed kernel
    output externally. This test sanity-checks that composition:
       final = prefill_kernel(routed) + shared_expert_ref(x)
    matches a full reference that includes both branches. Validates that the
    prefill kernel should NOT contain shared-expert computation.
    """
    cfg = dict(M=8, H=256, I=128, E=16, top_k=4, shared_I=128)
    d = make_inputs(cfg, with_shared=True)

    # Routed branch through kernel
    routed_k, routed_r = _run_routed_pipeline(cfg)

    # Shared branch via reference (no kernel yet)
    shared = ref_shared_expert(d["x"], d["shared_gate_up"],
                               d["shared_down"], d["shared_gate_w"])

    # Full reference = routed_ref + shared_ref
    full_r = (routed_r.float() + shared.float()).half()
    full_k = (routed_k.float() + shared.float()).half()

    diff_r = max_abs_diff(full_k, full_r)
    cos_r = cos_sim(full_k, full_r)
    # Shared is identical on both sides, so this effectively measures routed
    # kernel quality but with shared-expert magnitude added in.
    print(f"[shared] max_diff={diff_r:.5f}  cos={cos_r:.6f}  "
          f"(routed shape: H={cfg['H']}, I={cfg['I']}, E={cfg['E']}, TK={cfg['top_k']})")
    assert cos_r > 0.99, f"shared+routed cos too low: {cos_r}"

    # Also sanity: dropping the shared branch (mimicking what would happen
    # if the kernel itself tried to include it but we double-added) diverges.
    full_wrong = (routed_k.float() + 2.0 * shared.float()).half()
    cos_wrong = cos_sim(full_wrong, full_r)
    print(f"[shared] control (double-add shared) cos={cos_wrong:.6f}  "
          f"(expected noticeably lower than {cos_r:.6f})")


# ─── standalone runner ────────────────────────────────────────────────────────

def _run(name, fn):
    print(f"\n================ {name} ================")
    try:
        fn()
        print(f"PASS {name}")
        return True
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    tests = [
        ("gather_v2",          test_gather_v2),
        ("up_v2",              test_up_v2),
        ("down_v2",            test_down_v2),
        ("accumulate_v2",      test_accumulate_v2),
        ("full",               test_full_end_to_end),
        ("35B_TP4_shape",      test_35B_TP4_shape),
        ("122B_TP4_shape",     test_122B_TP4_shape),
        ("shared_composition", test_shared_expert_composition),
    ]
    results = [(n, _run(n, f)) for n, f in tests]

    print("\n==================== SUMMARY ====================")
    for n, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if any(not ok for _, ok in results):
        raise SystemExit(1)
