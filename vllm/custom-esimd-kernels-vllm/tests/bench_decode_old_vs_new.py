"""Decode E2E: old ESIMD fused vs new (ESIMD router + ESIMD topk + CUTLASS routed + torch shared)

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/bench_decode_old_vs_new.py"
"""
import gc, time, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp
from custom_esimd_kernels_vllm import moe_int4_ops, moe_int4_prefill_ops
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

def shared_expert_forward(x, sgu_w, sd_w, sg_w):
    gate_up = x @ sgu_w.t()
    act_out = torch.empty(x.size(0), gate_up.size(1) // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act_out, gate_up)
    down_out = act_out @ sd_w.t()
    gate = torch.sigmoid(x @ sg_w.t())
    return down_out * gate

# ═══ Weights ═══
# Old: K-major marlin
W13_q_old = torch.randint(-2**30, 2**30, (E, H//8, TWO_I), dtype=torch.int32, device=DEVICE)
W13_s_old = (torch.rand(E, H//GS, TWO_I, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_q_old = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=DEVICE)
W2_s_old = (torch.rand(E, I//GS, H, device=DEVICE)*0.04+0.002).to(DTYPE)

# New: CUTLASS N-major
W13_new = implement_zp(torch.randint(0, 0xFF, (E, TWO_I, H//2), dtype=torch.uint8, device=DEVICE)).contiguous()
W13_s_new = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_new = implement_zp(torch.randint(0, 0xFF, (E, H, I//2), dtype=torch.uint8, device=DEVICE)).contiguous()
W2_s_new = (torch.rand(E, H, I//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W13_new.xpu_fused_moe = True

# Router: GGML
gate_q_u8 = torch.randint(0, 1<<30, (E, H//8), dtype=torch.int32, device=DEVICE).view(torch.uint8)
gate_s = (torch.rand(E, H//GS, device=DEVICE)*0.02).to(DTYPE)

# Shared expert FP16
sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
sd_w = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
sg_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)

print(f"Decode E2E: Qwen3.5-122B-A10B TP=4  H={H} I={I} E={E} TK={TK} GS={GS}")
print(f"Old = ESIMD moe_forward_full_int4 (logits passed in, topk+routed+shared fused)")
print(f"New = ESIMD router + ESIMD topk + CUTLASS xpu_fused_moe + torch shared_expert")
print(f"warmup={WARMUP}  iters={ITERS}")
print()
print(f"| batch | Old(μs) | New(μs) | New/Old | Δ(μs) |")
print(f"|------:|--------:|--------:|--------:|------:|")

for batch in BATCHES:
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1
    logits_fp16 = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    # OLD: logits already computed externally, fused kernel does topk+routed+shared
    old_us = bench(lambda: moe_forward_full_int4(
        x, logits_fp16,
        W13_q_old, W13_s_old, sgu_w, sgus_empty,
        W2_q_old, W2_s_old, sd_w, sds_empty, sg_w,
        TK, 1, E, False))

    # NEW: ESIMD router → ESIMD topk → CUTLASS routed → torch shared
    def _new_fn():
        logits = moe_int4_ops.moe_router_forward_int4(x, gate_q_u8, gate_s, True)
        tw, ti = moe_int4_prefill_ops.moe_topk_softmax(logits, TK, E)
        routed = xpu_fused_moe(
            hidden_states=x,
            w13=W13_new, w13_scales=W13_s_new, w13_bias=None,
            w2=W2_new, w2_scales=W2_s_new, w2_bias=None,
            topk_weights=tw.float(), topk_ids=ti,
            n_experts_per_token=TK, activation="silu",
            num_experts=E, is_int4=True)
        shared = shared_expert_forward(x, sgu_w, sd_w, sg_w)
        return routed + shared

    new_us = bench(_new_fn)

    ratio = f"{new_us / old_us:.3f}x"
    delta = new_us - old_us
    print(f"| {batch:5d} | {old_us:7.0f} | {new_us:7.0f} | {ratio:>7s} | {delta:+5.0f} |")

    del x, logits_fp16
    torch.xpu.empty_cache()
    gc.collect()

print()
print("Note: Old path receives pre-computed logits; New path computes router internally.")
print("Done.")
