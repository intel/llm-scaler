"""122B-TP4 batched bench: GEMM / Prefill E2E / Decode E2E with multiple batch sizes.
Weights constructed directly on XPU for fast startup.

Usage: ZE_AFFINITY_MASK=2 python tests/bench_122b_batch.py
"""
import gc, sys, time, pathlib, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe, implement_zp

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP, ITERS = 5, 20
H, I, E, TK, GS = 3072, 1024, 256, 8, 128
TWO_I = 2 * I
OUTPUT_MD = "/llm/models/test/bench_122b_moe_int4_batch.md"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6

def tflops(flops, us):
    return flops / (us * 1e-6) / 1e12 if us > 0 else 0.0

def safe_ratio(a, b):
    if a != a or b != b or b <= 0: return "N/A"
    return f"{a/b:.3f}x"

# ═══ Weights ═══
def make_ipex_weights():
    W13_q = torch.randint(-2**30, 2**30, (E, H//8, TWO_I), dtype=torch.int32, device=DEVICE)
    W13_s = (torch.rand(E, H//GS, TWO_I, device=DEVICE)*0.04+0.002).to(DTYPE)
    W2_q  = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=DEVICE)
    W2_s  = (torch.rand(E, I//GS, H, device=DEVICE)*0.04+0.002).to(DTYPE)
    return W13_q, W13_s, W2_q, W2_s

def make_cutlass_weights():
    W13_u8 = torch.randint(0, 0xFF, (E, TWO_I, H//2), dtype=torch.uint8, device=DEVICE)
    W13_s  = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
    W2_u8  = torch.randint(0, 0xFF, (E, H, I//2), dtype=torch.uint8, device=DEVICE)
    W2_s   = (torch.rand(E, H, I//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
    W13_s4 = implement_zp(W13_u8.view(-1, H//2)).view(E, TWO_I, H//2)
    W2_s4  = implement_zp(W2_u8.view(-1, I//2)).view(E, H, I//2)
    W13_s4.xpu_fused_moe = True
    return W13_s4, W13_s, W2_s4, W2_s

def make_ipex_fusion():
    from intel_extension_for_pytorch.llm.modules import GatedMLPMOE
    W13_n = torch.randint(0, 1<<30, (E, TWO_I, H//8), dtype=torch.int32, device=DEVICE)
    W2_n  = torch.randint(0, 1<<30, (E, H, I//8), dtype=torch.int32, device=DEVICE)
    W13s  = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.02).to(DTYPE)
    W2s   = (torch.rand(E, H, I//GS, device=DEVICE)*0.02).to(DTYPE)
    return GatedMLPMOE(W13_n, W2_n, w1_scale_inv=W13s, w2_scale_inv=W2s, is_int4=True)

def make_routing(M):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    torch.manual_seed(1+M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    efto = torch.zeros(E+1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64); efto[E] = M*TK
    return dict(logits=logits, tw=tw, ti=ti, off=off, tok=tok, p2p=p2p, rows=rows, efto=efto)

# ═══ 1. GEMM batched ═══
def run_gemm_batched(lines):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    # M values: prefill-like (tokens entering one MoE layer)
    # Single card memory limits total activation to ~2GB -> max total ~350k
    GEMM_MS = [4096, 8192, 16384, 32768]

    lines.append("## 1. GEMM-only (W13 + W2)\n")
    lines.append("| M | total | IPEX W13(μs) | IPEX W2(μs) | IPEX Sum | CUT W13(μs) | CUT W2(μs) | CUT Sum | CUT/IPEX |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    # IPEX phase
    print("  IPEX GEMM...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    ipex_r = {}
    for M in GEMM_MS:
        total = M * TK
        rt = make_routing(M)
        xp = torch.randn(total, H, dtype=DTYPE, device=DEVICE)*0.1
        o1 = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
        w13 = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(o1, xp, W13_q, W13_s, None, rt["rows"], None, GS))
        del xp, o1; torch.xpu.empty_cache()
        ip = torch.randn(total, I, dtype=DTYPE, device=DEVICE)*0.1
        o2 = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
        w2 = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(o2, ip, W2_q, W2_s, None, rt["rows"], None, GS))
        ipex_r[M] = (w13, w2)
        del ip, o2, rt; torch.xpu.empty_cache()
    del W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # CUTLASS phase
    print("  CUTLASS GEMM...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()
    cut_r = {}
    for M in GEMM_MS:
        total = M * TK
        rt = make_routing(M)
        xp = torch.randn(total, H, dtype=DTYPE, device=DEVICE)*0.1
        o1 = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
        w13 = bench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=xp, ptr_B=cW13, ptr_scales=cW13s, ptr_bias=None, ptr_D=o1,
            expert_first_token_offset=rt["efto"], N=TWO_I, K=H, num_experts=E, is_B_int4=True, is_B_mxfp4=False))
        del xp, o1; torch.xpu.empty_cache()
        ip = torch.randn(total, I, dtype=DTYPE, device=DEVICE)*0.1
        o2 = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
        w2 = bench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=ip, ptr_B=cW2, ptr_scales=cW2s, ptr_bias=None, ptr_D=o2,
            expert_first_token_offset=rt["efto"], N=H, K=I, num_experts=E, is_B_int4=True, is_B_mxfp4=False))
        cut_r[M] = (w13, w2)
        del ip, o2, rt; torch.xpu.empty_cache()
    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    for M in GEMM_MS:
        total = M * TK
        iw13, iw2 = ipex_r[M]; isum = iw13+iw2
        cw13, cw2 = cut_r[M]; csum = cw13+cw2
        lines.append(f"| {M} | {total} | {iw13:.0f} | {iw2:.0f} | {isum:.0f} | {cw13:.0f} | {cw2:.0f} | {csum:.0f} | {safe_ratio(csum,isum)} |")
    lines.append("")
    print("  GEMM done")

# ═══ 2. Prefill E2E batched ═══
def run_prefill_batched(lines):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    # (batch, seq_len) -> M = batch * seq_len, capped for single-card memory
    PREFILL_CASES = [
        (1, 32768),
        (4, 8192),
        (8, 4096),
        (12, 2048),
    ]

    lines.append("## 2. Prefill E2E (one MoE layer)\n")
    lines.append("| batch | seq | M | IPEX(μs) | IPEX+ESIMD(μs) | ie/ipex | CUTLASS(μs) | cut/ipex |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")

    # IPEX + IPEX+ESIMD
    print("  IPEX / IPEX+ESIMD prefill...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    fusion = make_ipex_fusion()
    ipex_r, ie_r = {}, {}

    for batch, seq in PREFILL_CASES:
        M = batch * seq; total = M * TK
        x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
        logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

        ipex_r[(batch,seq)] = bench(lambda: fusion(x, False, TK, logits, True,
            topk_group=None, num_expert_group=None,
            custom_routing_function=None, scoring_func="softmax", activation="silu"))

        def _ie():
            tw, ti = ops.moe_topk_softmax(logits, TK, E)
            off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
            xp = ops.moe_prefill_scatter_x_forward(x, tok, TK)
            gu = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
            torch.ops.torch_ipex.group_mm_int4_out_marlin(gu, xp, W13_q, W13_s, None, rows, None, GS)
            ip = ops.moe_prefill_silu_mul_forward(gu)
            dp = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
            torch.ops.torch_ipex.group_mm_int4_out_marlin(dp, ip, W2_q, W2_s, None, rows, None, GS)
            return ops.moe_prefill_accumulate_permuted_forward_v2(dp, tw, p2p)
        ie_r[(batch,seq)] = bench(_ie)
        del x, logits; torch.xpu.empty_cache()

    del fusion, W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # CUTLASS
    print("  CUTLASS prefill...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()
    cut_r = {}
    for batch, seq in PREFILL_CASES:
        M = batch * seq
        x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
        rt = make_routing(M)
        tw_f32 = rt["tw"].float()
        try:
            cut_r[(batch,seq)] = bench(lambda: xpu_fused_moe(
                x, cW13, cW13s, None, cW2, cW2s, None,
                tw_f32, rt["ti"].to(torch.int64), TK, "silu", E, is_int4=True))
        except Exception as e:
            cut_r[(batch,seq)] = float('nan'); print(f"    batch={batch}: {e}")
        del x, rt, tw_f32; torch.xpu.empty_cache()
    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    for batch, seq in PREFILL_CASES:
        M = batch * seq
        ui = ipex_r[(batch,seq)]; ue = ie_r[(batch,seq)]; uc = cut_r[(batch,seq)]
        lines.append(f"| {batch} | {seq} | {M} | {ui:.0f} | {ue:.0f} | {safe_ratio(ue,ui)} | "
                     f"{'nan' if uc!=uc else f'{uc:.0f}'} | {safe_ratio(uc,ui)} |")
    lines.append("")
    print("  Prefill done")

# ═══ Shared expert helper ═══
def shared_expert_forward(x, sgu_w, sd_w, sg_w):
    """FP16 shared expert: gate_up matmul → silu_and_mul → down matmul → sigmoid gate → scale.
    This is what vllm does in Python for IPEX/CUTLASS paths (no fused shared expert kernel).
    Returns the shared expert contribution to add to routed MoE output."""
    # gate_up: x @ sgu_w.T -> [M, 2*I_shared]
    gate_up = x @ sgu_w.t()
    # silu_and_mul
    act_out = torch.empty(x.size(0), gate_up.size(1) // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act_out, gate_up)
    # down: act_out @ sd_w.T -> [M, H]
    down_out = act_out @ sd_w.t()
    # shared expert gate: sigmoid(x @ sg_w.T) -> [M, 1]
    gate = torch.sigmoid(x @ sg_w.t())
    return down_out * gate


# ═══ 3. Decode E2E batched ═══
def run_decode_batched(lines):
    DECODE_BATCHES = [1, 4, 8, 16, 32]

    # Shared expert dims (122B TP=4): I_shared = I = 1024
    I_SHARED = I

    lines.append("## 3. Decode E2E (one MoE layer, routed MoE + shared expert)\n")
    lines.append("All three paths: topk → routed MoE → shared expert → add.\n")
    lines.append("| batch | total | IPEX(μs) | CUTLASS(μs) | cut/ipex | ESIMD_dec(μs) | esm/ipex |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    # Shared expert weights (FP16, same for IPEX and CUTLASS paths)
    sgu_w = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE) * 0.01  # [2*I_shared, H]
    sd_w  = torch.randn(H, I_SHARED, dtype=DTYPE, device=DEVICE) * 0.01  # [H, I_shared]
    sg_w  = torch.randn(1, H, dtype=DTYPE, device=DEVICE) * 0.01  # [1, H]

    # IPEX (GatedMLPMOE + shared expert in Python)
    print("  IPEX decode...")
    fusion = make_ipex_fusion()
    ipex_r = {}
    for batch in DECODE_BATCHES:
        M = batch
        x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
        logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
        def _ipex_fn():
            routed = fusion(x, False, TK, logits, True,
                topk_group=None, num_expert_group=None,
                custom_routing_function=None, scoring_func="softmax", activation="silu")
            shared = shared_expert_forward(x, sgu_w, sd_w, sg_w)
            return routed + shared
        ipex_r[batch] = bench(_ipex_fn)
        del x, logits; torch.xpu.empty_cache()
    del fusion; torch.xpu.empty_cache(); gc.collect()

    # CUTLASS (topk_softmax + xpu_fused_moe + shared expert in Python)
    print("  CUTLASS decode...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights()
    cut_r = {}
    for batch in DECODE_BATCHES:
        M = batch
        x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
        logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
        tw_buf = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
        ti_buf = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
        tei_buf = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
        try:
            def _cut_fn():
                torch.ops._moe_C.topk_softmax(tw_buf, ti_buf, tei_buf, logits, True, None)
                routed = xpu_fused_moe(
                    x, cW13, cW13s, None, cW2, cW2s, None,
                    tw_buf, ti_buf.to(torch.int64), TK, "silu", E, is_int4=True)
                shared = shared_expert_forward(x, sgu_w, sd_w, sg_w)
                return routed + shared
            cut_r[batch] = bench(_cut_fn)
        except Exception as e:
            cut_r[batch] = float('nan'); print(f"    CUTLASS batch={batch}: {e}")
        del x, logits, tw_buf, ti_buf, tei_buf; torch.xpu.empty_cache()
    del cW13, cW13s, cW2, cW2s; torch.xpu.empty_cache(); gc.collect()

    # ESIMD decode (moe_forward_full_int4: topk + routed + shared all fused)
    print("  ESIMD decode...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights()
    from custom_esimd_kernels_vllm.ops import moe_forward_full_int4
    # Pass the same shared expert weights (FP16)
    sgus = torch.empty(0, dtype=DTYPE, device=DEVICE)
    sds  = torch.empty(0, dtype=DTYPE, device=DEVICE)

    esimd_r = {}
    for batch in DECODE_BATCHES:
        M = batch
        x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
        logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
        try:
            esimd_r[batch] = bench(lambda: moe_forward_full_int4(
                x, logits, W13_q, W13_s, sgu_w, sgus, W2_q, W2_s, sd_w, sds, sg_w,
                TK, 1, E, False))
        except Exception as e:
            esimd_r[batch] = float('nan'); print(f"    ESIMD batch={batch}: {e}")
        del x, logits; torch.xpu.empty_cache()
    del W13_q, W13_s, W2_q, W2_s, sgus, sds
    torch.xpu.empty_cache(); gc.collect()

    del sgu_w, sd_w, sg_w; torch.xpu.empty_cache()

    for batch in DECODE_BATCHES:
        total = batch * TK
        ui = ipex_r[batch]; uc = cut_r[batch]; ue = esimd_r[batch]
        lines.append(f"| {batch} | {total} | {ui:.0f} | "
                     f"{'nan' if uc!=uc else f'{uc:.0f}'} | {safe_ratio(uc,ui)} | "
                     f"{'nan' if ue!=ue else f'{ue:.0f}'} | {safe_ratio(ue,ui)} |")
    lines.append("")
    print("  Decode done")


def main():
    lines = ["# MoE INT4 Kernel Batch Comparison — Qwen3.5-122B-A10B TP=4\n"]
    lines.append(f"H={H}, I={I}, E={E}, TK={TK}, GS={GS}, dtype=fp16, single card\n")

    t0 = time.time()
    run_gemm_batched(lines)
    run_prefill_batched(lines)
    run_decode_batched(lines)
    print(f"\nTotal: {time.time()-t0:.1f}s")

    md = "\n".join(lines)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)
    print(f"Report: {OUTPUT_MD}\n")
    print(md)

if __name__ == "__main__":
    main()
