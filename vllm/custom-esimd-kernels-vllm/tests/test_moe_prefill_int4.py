"""
Accuracy tests for the prefill INT4 ESIMD MoE kernels
(csrc/moe_prefill/moe_prefill_int4.sycl).

Weights are consumed in IPEX K-major + marlin-shuffled layout, matching the
format that llm-scaler's decode-oriented moe_int4_ops already uses. This
unified layout is required so that prefill and decode can share the same
layer.w13_weight / layer.w2_weight tensors (no per-kernel repacking).

    gate_up_qweight : [E, H/8, 2*I] int32   marlin-shuffled nibbles
    gate_up_scale   : [E, H/GS, 2*I] fp16
    down_qweight    : [E, I/8, H] int32
    down_scale      : [E, I/GS, H] fp16

    marlin pack shift sequence  = [0, 4, 1, 5, 2, 6, 3, 7]
    kernel-side unshuffle slots = [0, 2, 4, 6, 1, 3, 5, 7]
    val = ((nibble - 8) * scale) as fp16

Tests are ordered from simplest to end-to-end:
    1. gather_forward_v2     — pure routing, no compute
    2. up_forward_v2         — gate+up+silu+mul (one DPAS kernel)
    3. down_forward_v2       — down projection (one DPAS kernel)
    4. accumulate_forward_v2 — weighted sum over top_k rows
    5. moe_prefill_full_int4 — end-to-end
    6. 35B-TP4 real-shape sanity
    7. 122B-TP4 real-shape sanity
    8. shared-expert composition (shared expert lives outside the kernel)

Usage:
    python -m pytest tests/test_moe_prefill_int4.py -v -s
    python tests/test_moe_prefill_int4.py      # standalone runner
"""
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "xpu"
GROUP_SIZE = 128
PACK_FACTOR = 8
# IPEX marlin packing: new nibble slot i comes from old nibble at shuffled_idx[i]
SHUFFLED_IDX = np.array([0, 4, 1, 5, 2, 6, 3, 7])


CFG_SMALL       = dict(M=16, H=256,  I=128, E=8,   top_k=2)
CFG_35B_TP4     = dict(M=32, H=2048, I=256, E=64,  top_k=8)
CFG_122B_TP4    = dict(M=16, H=3072, I=256, E=64,  top_k=8)


# ─── IPEX K-major + marlin int4 quantize / dequantize ────────────────────────

def quantize_int4(weight_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Quantize [N, K] fp16 into K-last int4:
        qweight [N, K/8] int32   nibbles in K-order [0..7]
        scale   [N, K/GS] fp16   symmetric, zero-point = 8
    This is the pre-shuffle/pre-transpose storage (matches what the llm-scaler
    test helpers produce).
    """
    N, K = weight_fp16.shape
    assert K % group_size == 0 and K % PACK_FACTOR == 0
    n_groups = K // group_size

    w = weight_fp16.float().numpy()
    w_grp = w.reshape(N, n_groups, group_size)

    max_abs = np.abs(w_grp).max(axis=2)
    scale_np = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)

    quantized = np.round(
        w_grp / scale_np[:, :, None].astype(np.float32)
    ).clip(-8, 7).astype(np.int32) + 8

    q_flat = quantized.reshape(N, K)
    q_packed = q_flat.reshape(N, K // PACK_FACTOR, PACK_FACTOR).astype(np.uint32)
    packed = np.zeros((N, K // PACK_FACTOR), dtype=np.uint32)
    for b in range(PACK_FACTOR):
        packed |= (q_packed[:, :, b] & 0xF) << (b * 4)

    return (torch.from_numpy(packed.view(np.int32)),      # [N, K/8] int32 (K-order)
            torch.from_numpy(scale_np))                   # [N, K/GS] fp16


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor,
                    N: int, K: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Inverse of quantize_int4 (K-order, pre-shuffle) -> fp16 [N, K]."""
    qw = qweight.numpy().view(np.uint32)
    sc = scales.numpy().astype(np.float32)
    n_groups = K // group_size

    unpacked = np.zeros((N, K), dtype=np.float32)
    for b in range(PACK_FACTOR):
        nibbles = ((qw >> (b * 4)) & 0xF).astype(np.float32) - 8.0
        unpacked[:, b::PACK_FACTOR] = nibbles

    unpacked = unpacked.reshape(N, n_groups, group_size) * sc[:, :, None]
    return torch.from_numpy(unpacked.reshape(N, K)).half()


def marlin_shuffle_weight(qweight_np: np.ndarray) -> np.ndarray:
    """IPEX marlin shuffle: within each int32, reorder nibbles so that slot i
    stores the original nibble at SHUFFLED_IDX[i]. Vectorised over the whole
    tensor (no per-expert Python loop).
    """
    result = np.zeros_like(qweight_np)
    for new_pos in range(PACK_FACTOR):
        old_pos = int(SHUFFLED_IDX[new_pos])
        nibbles = (qweight_np >> np.uint32(old_pos * 4)) & np.uint32(0xF)
        result |= nibbles << np.uint32(new_pos * 4)
    return result


def to_ipex_kmajor(qweight_nk: torch.Tensor, scales_nk: torch.Tensor):
    """Transform per-expert weights from [E, N, K/8] / [E, N, K/GS] (pre-shuffle)
    to the IPEX K-major + marlin-shuffled form expected by the kernels:
        qweight -> [E, K/8, N] int32 (marlin shuffled)
        scales  -> [E, K/GS, N] fp16 (transposed)
    """
    # transpose
    qw_t = qweight_nk.permute(0, 2, 1).contiguous()
    sc_t = scales_nk.permute(0, 2, 1).contiguous()
    # marlin shuffle (vectorised; entire expert tensor at once)
    qw_np = qw_t.numpy().view(np.uint32)
    qw_np = marlin_shuffle_weight(qw_np)
    qw_t  = torch.from_numpy(qw_np.view(np.int32))
    return qw_t, sc_t


def quantize_experts_ipex(W_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Per-expert wrapper: W_fp16 [E, N, K] fp16
    returns (qweight_ipex [E, K/8, N] int32, scales_ipex [E, K/GS, N] fp16)
    AND (W_dq [E, N, K] fp16) for reference paths.
    """
    E, N, K = W_fp16.shape
    qs, ss, dqs = [], [], []
    for e in range(E):
        q, s = quantize_int4(W_fp16[e], group_size)
        qs.append(q); ss.append(s)
        dqs.append(dequantize_int4(q, s, N, K, group_size))
    qweight_nk = torch.stack(qs)          # [E, N, K/8]
    scales_nk  = torch.stack(ss)          # [E, N, K/GS]
    qweight_ipex, scales_ipex = to_ipex_kmajor(qweight_nk, scales_nk)
    W_dq = torch.stack(dqs)               # [E, N, K] fp16
    return qweight_ipex, scales_ipex, W_dq


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
        gate_w = W13_fp16[e, :I, :].float()
        up_w   = W13_fp16[e, I:, :].float()
        x_rows = xf[tokens]
        g = x_rows @ gate_w.t()
        u = x_rows @ up_w.t()
        inter = (g / (1 + torch.exp(-g))) * u
        out[pairs] = inter.half()
    return out


def ref_down(intermediate: torch.Tensor, W2_fp16: torch.Tensor,
             expert_offsets: torch.Tensor, expert_tokens: torch.Tensor) -> torch.Tensor:
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
        w = W2_fp16[e].float()
        out[pairs] = (inter_f[pairs] @ w.t()).half()
    return out


def ref_accumulate(expert_output: torch.Tensor, routing_weights: torch.Tensor,
                   top_k: int) -> torch.Tensor:
    M, _ = routing_weights.shape
    _, H = expert_output.shape
    out = torch.zeros(M, H, dtype=torch.float32)
    eo = expert_output.float()
    rw = routing_weights.float()
    for t in range(M):
        for s in range(top_k):
            out[t] += rw[t, s] * eo[t * top_k + s]
    return out.half()


def ref_moe_routed(x: torch.Tensor, W13_fp16: torch.Tensor, W2_fp16: torch.Tensor,
                   topk_weights: torch.Tensor, topk_idx: torch.Tensor,
                   num_experts: int) -> torch.Tensor:
    TK = topk_idx.shape[1]
    off, tok = ref_gather(topk_idx, num_experts)
    inter   = ref_up(x, W13_fp16, off, tok, TK)
    exp_out = ref_down(inter, W2_fp16, off, tok)
    return ref_accumulate(exp_out, topk_weights, TK)


def ref_shared_expert(x: torch.Tensor, shared_gate_up_fp16: torch.Tensor,
                      shared_down_fp16: torch.Tensor,
                      shared_gate_weight=None) -> torch.Tensor:
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


# ─── Input factories ──────────────────────────────────────────────────────────

def make_inputs(cfg, seed=0, with_shared=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    M, H, I, E, TK = cfg["M"], cfg["H"], cfg["I"], cfg["E"], cfg["top_k"]
    SI = cfg.get("shared_I", I)

    x   = torch.randn(M, H, dtype=torch.float16) * 0.1
    W13 = torch.randn(E, 2 * I, H, dtype=torch.float16) * 0.02
    W2  = torch.randn(E, H, I,   dtype=torch.float16) * 0.02
    logits = torch.randn(M, E, dtype=torch.float16)

    # IPEX K-major + marlin shuffled for the kernel, plus dequantised fp16 for refs.
    W13_q, W13_s, W13_dq = quantize_experts_ipex(W13, GROUP_SIZE)
    W2_q,  W2_s,  W2_dq  = quantize_experts_ipex(W2,  GROUP_SIZE)

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

    assert max_abs_diff(off_k.cpu(), off_r) == 0, "expert_offsets mismatch"

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

    x_xpu = _xpu(d["x"])
    w13_q = _xpu(d["W13_q"])
    w13_s = _xpu(d["W13_s"])
    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(_xpu(d["topk_idx"]), cfg["E"])
    out_k = ops.moe_prefill_up_forward_v2(
        x_xpu, w13_q, w13_s, off_xpu, tok_xpu, cfg["top_k"]
    ).cpu()

    out_r = ref_up(d["x"], d["W13_dq"], off_xpu.cpu(), tok_xpu.cpu(), cfg["top_k"])
    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
    print(f"[up]   max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.99, f"cos too low: {cos}"


def test_down_v2():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = CFG_SMALL
    d = make_inputs(cfg)

    x_xpu = _xpu(d["x"])
    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(_xpu(d["topk_idx"]), cfg["E"])
    inter_ref = ref_up(d["x"], d["W13_dq"], off_xpu.cpu(), tok_xpu.cpu(), cfg["top_k"])
    inter_xpu = _xpu(inter_ref)

    out_k = ops.moe_prefill_down_forward_v2(
        inter_xpu, _xpu(d["W2_q"]), _xpu(d["W2_s"]), off_xpu, tok_xpu
    ).cpu()
    out_r = ref_down(inter_ref, d["W2_dq"], off_xpu.cpu(), tok_xpu.cpu())

    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
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

    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
    print(f"[acc]  max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.999, f"cos too low: {cos}"


def test_full_end_to_end():
    """End-to-end: kernel-side softmax+topk may break ties differently from
    torch.topk, so use a loose cos threshold.
    """
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    cfg = dict(M=16, H=256, I=128, E=256, top_k=8)
    d = make_inputs(cfg)

    out_k = ops.moe_prefill_full_int4(
        _xpu(d["x"]), _xpu(d["logits"]),
        _xpu(d["W13_q"]), _xpu(d["W13_s"]),
        _xpu(d["W2_q"]),  _xpu(d["W2_s"]),
        cfg["top_k"], cfg["E"],
    ).cpu()

    out_r = ref_moe_routed(d["x"], d["W13_dq"], d["W2_dq"],
                           d["topk_weights"], d["topk_idx"], cfg["E"])
    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
    print(f"[full] max_diff={diff:.5f}  cos={cos:.6f}")
    assert cos > 0.90, f"end-to-end cos too low: {cos}"


# ─── real-shape sanity ───────────────────────────────────────────────────────

def _run_routed_pipeline(cfg):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    d = make_inputs(cfg)
    off_xpu, tok_xpu = ops.moe_prefill_gather_forward_v2(_xpu(d["topk_idx"]), cfg["E"])
    inter_xpu = ops.moe_prefill_up_forward_v2(
        _xpu(d["x"]), _xpu(d["W13_q"]), _xpu(d["W13_s"]),
        off_xpu, tok_xpu, cfg["top_k"])
    exp_xpu = ops.moe_prefill_down_forward_v2(
        inter_xpu, _xpu(d["W2_q"]), _xpu(d["W2_s"]), off_xpu, tok_xpu)
    out_k = ops.moe_prefill_accumulate_forward_v2(exp_xpu, _xpu(d["topk_weights"])).cpu()
    out_r = ref_moe_routed(d["x"], d["W13_dq"], d["W2_dq"],
                           d["topk_weights"], d["topk_idx"], cfg["E"])
    return out_k, out_r


def test_35B_TP4_shape():
    cfg = CFG_35B_TP4
    out_k, out_r = _run_routed_pipeline(cfg)
    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
    print(f"[35B]  max_diff={diff:.5f}  cos={cos:.6f}  (H=2048, I=256, E=64, TK=8)")
    assert cos > 0.99, f"35B routed cos too low: {cos}"


def test_122B_TP4_shape():
    cfg = CFG_122B_TP4
    out_k, out_r = _run_routed_pipeline(cfg)
    diff = max_abs_diff(out_k, out_r); cos = cos_sim(out_k, out_r)
    print(f"[122B] max_diff={diff:.5f}  cos={cos:.6f}  (H=3072, I=256, E=64, TK=8)")
    assert cos > 0.99, f"122B routed cos too low: {cos}"


def test_shared_expert_composition():
    """Shared expert lives outside the prefill kernel — verify the composition
    final = prefill_kernel(routed) + shared_expert_ref(x) is what a full
    reference computes.
    """
    cfg = dict(M=8, H=256, I=128, E=16, top_k=4, shared_I=128)
    d = make_inputs(cfg, with_shared=True)
    routed_k, routed_r = _run_routed_pipeline(cfg)
    shared = ref_shared_expert(d["x"], d["shared_gate_up"],
                               d["shared_down"], d["shared_gate_w"])

    full_r = (routed_r.float() + shared.float()).half()
    full_k = (routed_k.float() + shared.float()).half()

    diff = max_abs_diff(full_k, full_r); cos = cos_sim(full_k, full_r)
    print(f"[shared] max_diff={diff:.5f}  cos={cos:.6f}  "
          f"(H={cfg['H']}, I={cfg['I']}, E={cfg['E']}, TK={cfg['top_k']})")
    assert cos > 0.99, f"shared+routed cos too low: {cos}"

    full_wrong = (routed_k.float() + 2.0 * shared.float()).half()
    cos_wrong = cos_sim(full_wrong, full_r)
    print(f"[shared] control (double-add shared) cos={cos_wrong:.6f}  "
          f"(expected noticeably lower than {cos:.6f})")


# ─── standalone runner ────────────────────────────────────────────────────────

def _run(name, fn):
    print(f"\n================ {name} ================")
    try:
        fn()
        print(f"PASS {name}")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
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
