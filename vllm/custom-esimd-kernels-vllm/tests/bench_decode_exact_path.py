"""Benchmark exact decode path from qwen3_next.py: old vs new, per-stage breakdown.

Simulates the exact code path for 122B TP=4 decode with realistic batch sizes.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=7 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/bench_decode_exact_path.py"
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
WARMUP, ITERS = 30, 200
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

# ═══ OLD path weights (K-major marlin) ═══
W13_q_old = torch.randint(-2**30, 2**30, (E, H//8, TWO_I), dtype=torch.int32, device=DEVICE)
W13_s_old = (torch.rand(E, H//GS, TWO_I, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_q_old = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=DEVICE)
W2_s_old = (torch.rand(E, I//GS, H, device=DEVICE)*0.04+0.002).to(DTYPE)

# ═══ NEW path weights (CUTLASS N-major) ═══
W13_new = implement_zp(torch.randint(0, 0xFF, (E, TWO_I, H//2), dtype=torch.uint8, device=DEVICE)).contiguous()
W13_s_new = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W2_new = implement_zp(torch.randint(0, 0xFF, (E, H, I//2), dtype=torch.uint8, device=DEVICE)).contiguous()
W2_s_new = (torch.rand(E, H, I//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
W13_new.xpu_fused_moe = True

# Router GGML
gate_q_u8 = torch.randint(0, 1<<30, (E, H//8), dtype=torch.int32, device=DEVICE).view(torch.uint8)
gate_s = (torch.rand(E, H//GS, device=DEVICE)*0.02).to(DTYPE)

# Shared expert FP16
sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
sd_w = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01
sg_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01
sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)

def shared_expert_torch(x):
    gate_up = x @ sgu_w.t()
    act_out = torch.empty(x.size(0), gate_up.size(1)//2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act_out, gate_up)
    down = act_out @ sd_w.t()
    gv = torch.sigmoid(x @ sg_w.t())
    return down * gv

print(f"Decode path benchmark: 122B-TP4  H={H} I={I} E={E} TK={TK}")
print(f"warmup={WARMUP}  iters={ITERS}")

# ════════ Per-stage breakdown for NEW path ════════
print(f"\n## Per-stage: OLD fused vs NEW breakdown")
print(f"| batch | OLD fused(μs) | NEW router | topk | fused_moe | shared | NEW sum(μs) | New/Old |")
print(f"|------:|--------------:|-----------:|-----:|----------:|-------:|------------:|--------:|")

for batch in BATCHES:
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1
    logits_fp16 = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    # OLD fused
    us_old = bench(lambda: moe_forward_full_int4(
        x, logits_fp16, W13_q_old, W13_s_old, sgu_w, sgus_empty,
        W2_q_old, W2_s_old, sd_w, sds_empty, sg_w, TK, 1, E, False))

    # NEW per-stage
    us_r = bench(lambda: moe_int4_ops.moe_router_forward_int4(x, gate_q_u8, gate_s, True))

    logits = moe_int4_ops.moe_router_forward_int4(x, gate_q_u8, gate_s, True)
    us_t = bench(lambda: moe_int4_prefill_ops.moe_topk_softmax(logits, TK, E))

    tw, ti = moe_int4_prefill_ops.moe_topk_softmax(logits, TK, E)
    us_f = bench(lambda: xpu_fused_moe(
        hidden_states=x, w13=W13_new, w13_scales=W13_s_new, w13_bias=None,
        w2=W2_new, w2_scales=W2_s_new, w2_bias=None,
        topk_weights=tw.float(), topk_ids=ti,
        n_experts_per_token=TK, activation="silu", num_experts=E, is_int4=True))

    us_s = bench(lambda: shared_expert_torch(x))

    us_sum = us_r + us_t + us_f + us_s
    ratio = f"{us_sum/us_old:.3f}x"
    print(f"| {batch:5d} | {us_old:13.0f} | {us_r:10.0f} | {us_t:4.0f} | {us_f:9.0f} | {us_s:6.0f} | {us_sum:11.0f} | {ratio:>7s} |")

    del x, logits_fp16, logits, tw, ti; torch.xpu.empty_cache(); gc.collect()

print("\nDone.")
