"""Decode sub-stage benchmark: compare ESIMD vs IPEX vs XPU-K for router, topk, shared_expert.

These are the three ops we're considering replacing in the decode path.
Small batch sizes (1, 4, 8, 16, 32) — actual decode scenario.

Usage: docker exec wj-test-new-0422 bash -c "ZE_AFFINITY_MASK=0 python /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests/bench_decode_substages.py"
"""
import gc, time, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import implement_zp

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


def ratio(a, b):
    if a != a or b != b or b <= 0:
        return "N/A"
    return f"{a / b:.3f}x"


# ═══ Weights ═══

# ESIMD router: GGML N-major [E, K/8] int32, scale [E, K/GS] fp16
gate_q_ggml = torch.randint(0, 1 << 30, (E, H // 8), dtype=torch.int32, device=DEVICE)
gate_s_ggml = (torch.rand(E, H // GS, device=DEVICE) * 0.02).to(DTYPE)
gate_q_u8 = gate_q_ggml.view(torch.uint8)  # [E, H/2] for GGML layout

# IPEX WoQ router: needs ipex.llm.quantization setup — we simulate with self.gate(x)
# Actually for IPEX we just use a standard FP16 matmul as proxy (self.gate uses ipex qlinear)
gate_fp16 = torch.randn(E, H, dtype=DTYPE, device=DEVICE) * 0.01

# Shared expert weights
# ESIMD: IPEX-format (OneDNN repacked) — we use K-major for simplicity
# For benchmark we just need the kernel call latency
shared_gu_ipex = torch.randint(0, 1 << 30, (TWO_I, H // 8), dtype=torch.int32, device=DEVICE)
shared_gu_s_ipex = (torch.rand(H // GS, TWO_I, device=DEVICE) * 0.02).to(DTYPE)
shared_d_ipex = torch.randint(0, 1 << 30, (H, I // 8), dtype=torch.int32, device=DEVICE)
shared_d_s_ipex = (torch.rand(I // GS, H, device=DEVICE) * 0.02).to(DTYPE)
shared_gate_w = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01

# FP16 shared expert weights (for IPEX/torch path)
shared_gu_fp16 = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01
shared_d_fp16 = torch.randn(H, I, dtype=DTYPE, device=DEVICE) * 0.01

from custom_esimd_kernels_vllm import moe_int4_ops

print(f"Decode sub-stage benchmark: Qwen3.5-122B-A10B TP=4")
print(f"H={H} I={I} E={E} TK={TK} GS={GS} warmup={WARMUP} iters={ITERS}")

# ═══════════════════════════════════════════════════
# 1. ROUTER: x @ gate_weight.T → logits [M, E]
# ═══════════════════════════════════════════════════
print(f"\n## 1. Router (x → logits)")
print(f"| batch | ESIMD(μs) | FP16 matmul(μs) | IPEX topk_softmax includes router? | fp16/esimd |")
print(f"|------:|----------:|----------------:|---:|---:|")

for batch in BATCHES:
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1

    # ESIMD moe_router_forward_int4 (GGML layout)
    us_esimd = bench(lambda: moe_int4_ops.moe_router_forward_int4(
        x, gate_q_u8, gate_s_ggml, True))

    # FP16 matmul (simulates self.gate(x) = x @ weight.T)
    us_fp16 = bench(lambda: x @ gate_fp16.t())

    print(f"| {batch:5d} | {us_esimd:9.1f} | {us_fp16:15.1f} | - | {ratio(us_fp16, us_esimd)} |")
    del x
    torch.xpu.empty_cache()

# ═══════════════════════════════════════════════════
# 2. TOPK: logits → topk_weights, topk_ids
# ═══════════════════════════════════════════════════
print(f"\n## 2. TopK + Softmax")
print(f"| batch | ESIMD(μs) | IPEX(μs) | XPU-K(μs) | ipex/esimd | xpuk/esimd |")
print(f"|------:|----------:|---------:|----------:|---:|---:|")

from custom_esimd_kernels_vllm import moe_int4_prefill_ops as prefill_ops

for batch in BATCHES:
    logits = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    # ESIMD
    us_e = bench(lambda: prefill_ops.moe_topk_softmax(logits, TK, E))

    # IPEX
    tw_i = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti_i = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei_i = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    us_i = bench(lambda: torch.ops.torch_ipex.topk_softmax(tw_i, ti_i, tei_i, logits, True))

    # XPU-K
    tw_x = torch.empty(batch, TK, dtype=torch.float32, device=DEVICE)
    ti_x = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    tei_x = torch.empty(batch, TK, dtype=torch.int32, device=DEVICE)
    us_x = bench(lambda: torch.ops._moe_C.topk_softmax(tw_x, ti_x, tei_x, logits, True, None))

    print(f"| {batch:5d} | {us_e:9.1f} | {us_i:8.1f} | {us_x:9.1f} | {ratio(us_i, us_e)} | {ratio(us_x, us_e)} |")
    del logits
    torch.xpu.empty_cache()

# ═══════════════════════════════════════════════════
# 3. SHARED EXPERT: x → gate_up → silu_mul → down → gate → output
# ═══════════════════════════════════════════════════
print(f"\n## 3. Shared Expert (full: gate_up + silu_mul + down + sigmoid_gate)")
print(f"| batch | ESIMD_fused(μs) | FP16_torch(μs) | torch/esimd |")
print(f"|------:|----------------:|---------------:|---:|")

# We can't call shared expert sub-kernels separately from moe_forward_full_int4
# but we can measure moe_forward_full_int4 with only shared experts (num_routed=0 won't work)
# So measure the Python torch path as baseline

for batch in BATCHES:
    x = torch.randn(batch, H, dtype=DTYPE, device=DEVICE) * 0.1

    # FP16 torch path (what self.shared_expert(x) does)
    def _shared_torch():
        gate_up = x @ shared_gu_fp16.t()
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        act = torch.nn.functional.silu(gate) * up
        down = act @ shared_d_fp16.t()
        gate_val = torch.sigmoid(x @ shared_gate_w.t())
        return down * gate_val

    us_torch = bench(_shared_torch)

    # ESIMD: measure full moe_forward_full_int4 minus routed part
    # Run full pipeline with dummy routed weights, then subtract
    # Actually, let's measure the full thing and the routed-only separately
    # to isolate shared expert cost.

    # Full: routed + shared
    W13_q = torch.randint(-2**30, 2**30, (E, H // 8, TWO_I), dtype=torch.int32, device=DEVICE)
    W13_s = (torch.rand(E, H // GS, TWO_I, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    W2_q = torch.randint(-2**30, 2**30, (E, I // 8, H), dtype=torch.int32, device=DEVICE)
    W2_s = (torch.rand(E, I // GS, H, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
    sgus_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)
    sds_empty = torch.empty(0, dtype=DTYPE, device=DEVICE)

    logits = torch.randn(batch, E, dtype=DTYPE, device=DEVICE)

    # Full (routed + shared)
    us_full = bench(lambda: moe_int4_ops.moe_forward_full_int4(
        x, logits, W13_q, W13_s,
        shared_gu_fp16, sgus_empty,
        W2_q, W2_s,
        shared_d_fp16, sds_empty,
        shared_gate_w, TK, 1, E, False))

    # Routed only (no shared: pass dummy 0-dim shared weights, num_shared=0)
    # moe_forward_full_int4 with num_shared=0 skips shared expert
    try:
        us_routed = bench(lambda: moe_int4_ops.moe_forward_full_int4(
            x, logits, W13_q, W13_s,
            torch.zeros(1, 1, dtype=DTYPE, device=DEVICE), sgus_empty,
            W2_q, W2_s,
            torch.zeros(1, 1, dtype=DTYPE, device=DEVICE), sds_empty,
            None, TK, 0, E, False))
        us_esimd_shared = us_full - us_routed
        esimd_str = f"{us_esimd_shared:15.1f}"
    except Exception:
        us_esimd_shared = float('nan')
        esimd_str = "err"

    print(f"| {batch:5d} | {esimd_str} | {us_torch:14.1f} | {ratio(us_torch, us_esimd_shared) if us_esimd_shared == us_esimd_shared else 'N/A'} |")

    del x, logits, W13_q, W13_s, W2_q, W2_s
    torch.xpu.empty_cache()
    gc.collect()

# ═══════════════════════════════════════════════════
# 4. IPEX WoQ Linear (self.gate) vs ESIMD router — latency comparison
# ═══════════════════════════════════════════════════
print(f"\n## 4. IPEX WoQ Linear vs ESIMD INT4 GEMV (router-sized)")
print(f"Router: [batch, {H}] x [{E}, {H}].T → [batch, {E}]")
print(f"IPEX WoQ simulation: FP16 matmul (actual WoQ would be faster due to INT4 compression)")
print(f"ESIMD: moe_router_forward_int4 with GGML layout")
print()

print("Done.")
