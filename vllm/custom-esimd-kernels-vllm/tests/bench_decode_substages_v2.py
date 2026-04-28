"""Decode sub-stage benchmark v2: measure each component of the new decode path.

New path = ESIMD router + ESIMD topk + CUTLASS xpu_fused_moe + shared_expert(torch)
We measure each piece to find what's worth replacing with ESIMD.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/bench_decode_substages_v2.py"
"""
import gc, time, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp
from custom_esimd_kernels_vllm import moe_int4_ops, moe_int4_prefill_ops as prefill_ops
from custom_esimd_kernels_vllm.ops import moe_forward_full_int4

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP, ITERS = 20, 100
H, I, E, TK, GS = 3072, 1024, 256, 8, 128
TWO_I = 2 * I
BATCHES = [1, 4, 8, 16, 32]


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def r(a, b):
    return f"{a/b:.3f}x" if b > 0 and a == a and b == b else "N/A"


# ═══ Weights ═══
# ESIMD old path: K-major marlin
W13_q_old = torch.randint(-2**30, 2**30, (E, H//8, TWO_I), dtype=torch.int32, device=DEVICE)
W13_s_old = (torch.rand(E, H//GS, TWO_I, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_q_old = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=DEVICE)
W2_s_old = (torch.rand(E, I//GS, H, device=DEVICE)*0.04+0.002).to(DTYPE)

# CUTLASS new path: N-major uint8
W13_u8 = torch.randint(0, 0xFF, (E, TWO_I, H//2), dtype=torch.uint8, device=DEVICE)
W13_s_new = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_u8 = torch.randint(0, 0xFF, (E, H, I//2), dtype=torch.uint8, device=DEVICE)
W2_s_new = (torch.rand(E, H, I//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W13_new = torch.empty_like(W13_u8)
W2_new = torch.empty_like(W2_u8)
for i in range(E):
    W13_new[i] = implement_zp(W13_u8[i])
    W2_new[i] = implement_zp(W2_u8[i])
W13_new = W13_new.contiguous(); W2_new = W2_new.contiguous()
W13_new.xpu_fused_moe = True
del W13_u8, W2_u8; torch.xpu.empty_cache()

# Router: GGML
gate_q = torch.randint(0, 1<<30, (E, H//8), dtype=torch.int32, device=DEVICE)
gate_s = (torch.rand(E, H//GS, device=DEVICE)*0.02).to(DTYPE)
gate_u8 = gate_q.view(torch.uint8)

# Shared expert FP16
sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
sd_w = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
sg_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)

print(f"Decode sub-stage benchmark v2: Qwen3.5-122B-A10B TP=4")
print(f"H={H} I={I} E={E} TK={TK} warmup={WARMUP} iters={ITERS}\n")

# ════════════════════════════════════════════════════
print("## Old path (ESIMD fused) vs New path (breakdown)")
print()
print("| batch | OLD total(μs) | router(μs) | topk(μs) | fused_moe(μs) | shared(μs) | NEW total(μs) | NEW/OLD |")
print("|------:|--------------:|-----------:|---------:|--------------:|-----------:|--------------:|--------:|")

for batch in BATCHES:
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1
    logits_fp16 = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    # OLD: moe_forward_full_int4 (all fused)
    us_old = bench(lambda: moe_forward_full_int4(
        x, logits_fp16, W13_q_old, W13_s_old, sgu_w, sgus_empty,
        W2_q_old, W2_s_old, sd_w, sds_empty, sg_w, TK, 1, E, False))

    # NEW breakdown:
    # 1. Router (ESIMD)
    us_router = bench(lambda: moe_int4_ops.moe_router_forward_int4(
        x, gate_u8, gate_s, True))

    # 2. TopK (ESIMD)
    us_topk = bench(lambda: prefill_ops.moe_topk_softmax(logits_fp16, TK, E))

    # 3. xpu_fused_moe (CUTLASS routed)
    tw = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits_fp16, True)
    us_fused = bench(lambda: xpu_fused_moe(
        hidden_states=x, w13=W13_new, w13_scales=W13_s_new, w13_bias=None,
        w2=W2_new, w2_scales=W2_s_new, w2_bias=None,
        topk_weights=tw, topk_ids=ti, n_experts_per_token=TK,
        activation="silu", num_experts=E, is_int4=True))

    # 4. Shared expert (FP16 torch)
    def _shared():
        gate_up = x @ sgu_w.t()
        gate = gate_up[:, :I]; up = gate_up[:, I:]
        act = torch.nn.functional.silu(gate) * up
        down = act @ sd_w.t()
        gv = torch.sigmoid(x @ sg_w.t())
        return down * gv
    us_shared = bench(_shared)

    us_new = us_router + us_topk + us_fused + us_shared

    print(f"| {batch:5d} | {us_old:13.0f} | {us_router:10.0f} | {us_topk:8.0f} | "
          f"{us_fused:13.0f} | {us_shared:10.0f} | {us_new:13.0f} | {r(us_new, us_old)} |")

    del x, logits_fp16, tw, ti, tei
    torch.xpu.empty_cache()

print()
print("Note: NEW total is sum of parts (no kernel launch overlap).")
print("OLD = ESIMD moe_forward_full_int4 (router logits passed in, topk+routed+shared all fused)")
print("Done.")
