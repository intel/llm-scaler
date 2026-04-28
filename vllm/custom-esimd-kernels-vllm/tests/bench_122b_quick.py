"""Quick 122B-TP4 three-table bench: GEMM / Prefill E2E / Decode E2E.
Weights constructed directly on XPU (random packed INT4) for fast startup.
Outputs markdown report.

Usage: ZE_AFFINITY_MASK=2 python tests/bench_122b_quick.py
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
OUTPUT_MD = "/llm/models/test/bench_122b_moe_int4_comparison.md"

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


# ═══ Fast weight factories (all on XPU, no CPU quantize) ═══

def make_ipex_weights_fast():
    """Random IPEX K-major marlin weights directly on XPU.
    [E, K/8, N] int32, [E, K/GS, N] fp16."""
    W13_q = torch.randint(-2**30, 2**30, (E, H//8, TWO_I), dtype=torch.int32, device=DEVICE)
    W13_s = (torch.rand(E, H//GS, TWO_I, device=DEVICE)*0.04+0.002).to(DTYPE)
    W2_q  = torch.randint(-2**30, 2**30, (E, I//8, H), dtype=torch.int32, device=DEVICE)
    W2_s  = (torch.rand(E, I//GS, H, device=DEVICE)*0.04+0.002).to(DTYPE)
    return W13_q, W13_s, W2_q, W2_s

def make_cutlass_weights_fast():
    """Random CUTLASS format weights directly on XPU.
    [E, N, K/2] uint8 (s4 packed), [E, N, K/GS] fp16."""
    W13_u8 = torch.randint(0, 0xFF, (E, TWO_I, H//2), dtype=torch.uint8, device=DEVICE)
    W13_s  = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
    W2_u8  = torch.randint(0, 0xFF, (E, H, I//2), dtype=torch.uint8, device=DEVICE)
    W2_s   = (torch.rand(E, H, I//GS, device=DEVICE)*0.04+0.002).to(DTYPE)
    # implement_zp per-expert (vectorized over whole tensor is fine for random data)
    W13_s4 = implement_zp(W13_u8.view(-1, H//2)).view(E, TWO_I, H//2)
    W2_s4  = implement_zp(W2_u8.view(-1, I//2)).view(E, H, I//2)
    W13_s4.xpu_fused_moe = True
    return W13_s4, W13_s, W2_s4, W2_s

def make_ipex_fusion_fast():
    """IPEX GatedMLPMOE with random N-major weights. Prepack runs on first forward."""
    from intel_extension_for_pytorch.llm.modules import GatedMLPMOE
    W13_n = torch.randint(0, 1<<30, (E, TWO_I, H//8), dtype=torch.int32, device=DEVICE)
    W2_n  = torch.randint(0, 1<<30, (E, H, I//8), dtype=torch.int32, device=DEVICE)
    W13s  = (torch.rand(E, TWO_I, H//GS, device=DEVICE)*0.02).to(DTYPE)
    W2s   = (torch.rand(E, H, I//GS, device=DEVICE)*0.02).to(DTYPE)
    return GatedMLPMOE(W13_n, W2_n, w1_scale_inv=W13s, w2_scale_inv=W2s, is_int4=True)


# ═══ Routing ═══
def make_routing(M):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    torch.manual_seed(1+M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    efto = torch.zeros(E+1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64); efto[E] = M*TK
    return dict(logits=logits, tw=tw, ti=ti, off=off, tok=tok, p2p=p2p, rows=rows, efto=efto)


# ═══ Section 1: GEMM-only ═══
def run_gemm(lines):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    M = 32768; total = M * TK
    lines.append("## 1. GEMM-only (M=32768, total_tokens=262144)\n")
    lines.append("Pure grouped-GEMM, no topk/scatter/silu/accumulate.\n")

    # IPEX
    print("  IPEX GEMM...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights_fast()
    rt = make_routing(M)
    x_perm = torch.randn(total, H, dtype=DTYPE, device=DEVICE)*0.1
    out = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
    ipex_w13 = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(
        out, x_perm, W13_q, W13_s, None, rt["rows"], None, GS))
    del x_perm, out; torch.xpu.empty_cache()
    inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE)*0.1
    out2 = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
    ipex_w2 = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(
        out2, inter, W2_q, W2_s, None, rt["rows"], None, GS))
    del inter, out2, W13_q, W13_s, W2_q, W2_s, rt; torch.xpu.empty_cache(); gc.collect()

    # CUTLASS
    print("  CUTLASS GEMM...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights_fast()
    rt = make_routing(M)
    x_perm = torch.randn(total, H, dtype=DTYPE, device=DEVICE)*0.1
    out = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
    cut_w13 = bench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=x_perm, ptr_B=cW13, ptr_scales=cW13s, ptr_bias=None, ptr_D=out,
        expert_first_token_offset=rt["efto"], N=TWO_I, K=H, num_experts=E,
        is_B_int4=True, is_B_mxfp4=False))
    del x_perm, out; torch.xpu.empty_cache()
    inter = torch.randn(total, I, dtype=DTYPE, device=DEVICE)*0.1
    out2 = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
    cut_w2 = bench(lambda: torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=inter, ptr_B=cW2, ptr_scales=cW2s, ptr_bias=None, ptr_D=out2,
        expert_first_token_offset=rt["efto"], N=H, K=I, num_experts=E,
        is_B_int4=True, is_B_mxfp4=False))
    del inter, out2, cW13, cW13s, cW2, cW2s, rt; torch.xpu.empty_cache(); gc.collect()

    fl_w13 = 2.0*total*TWO_I*H; fl_w2 = 2.0*total*H*I
    lines.append("| Stage | IPEX (μs) | IPEX TFLOPS | CUTLASS (μs) | CUT TFLOPS | CUT/IPEX |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| W13 | {ipex_w13:.0f} | {tflops(fl_w13,ipex_w13):.1f} | {cut_w13:.0f} | {tflops(fl_w13,cut_w13):.1f} | {cut_w13/ipex_w13:.3f}x |")
    lines.append(f"| W2 | {ipex_w2:.0f} | {tflops(fl_w2,ipex_w2):.1f} | {cut_w2:.0f} | {tflops(fl_w2,cut_w2):.1f} | {cut_w2/ipex_w2:.3f}x |")
    lines.append(f"| **Sum** | **{ipex_w13+ipex_w2:.0f}** | | **{cut_w13+cut_w2:.0f}** | | **{(cut_w13+cut_w2)/(ipex_w13+ipex_w2):.3f}x** |")
    lines.append("")
    print(f"  done: IPEX W13={ipex_w13:.0f} W2={ipex_w2:.0f}  CUTLASS W13={cut_w13:.0f} W2={cut_w2:.0f}")

# ═══ Section 2: Prefill E2E ═══
def run_prefill_e2e(lines):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    M = 32768; total = M * TK
    lines.append("## 2. Prefill E2E (M=32768, one MoE layer)\n")
    lines.append("topk → scatter → GEMM(W13) → silu_mul → GEMM(W2) → accumulate.\n")

    # IPEX + IPEX+ESIMD
    print("  IPEX / IPEX+ESIMD prefill...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights_fast()
    fusion = make_ipex_fusion_fast()
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

    us_ipex = bench(lambda: fusion(x, False, TK, logits, True,
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
    us_ie = bench(_ie)
    del x, logits, fusion, W13_q, W13_s, W2_q, W2_s; torch.xpu.empty_cache(); gc.collect()

    # CUTLASS
    print("  CUTLASS prefill...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights_fast()
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
    rt = make_routing(M)
    tw_f32 = rt["tw"].float()
    try:
        us_cut = bench(lambda: xpu_fused_moe(
            x, cW13, cW13s, None, cW2, cW2s, None,
            tw_f32, rt["ti"].to(torch.int64), TK, "silu", E, is_int4=True))
    except Exception as e:
        us_cut = float('nan'); print(f"    CUTLASS: {e}")
    del x, cW13, cW13s, cW2, cW2s, rt, tw_f32; torch.xpu.empty_cache(); gc.collect()

    lines.append("| Path | μs | vs IPEX |")
    lines.append("|---|---:|---:|")
    lines.append(f"| IPEX (GatedMLPMOE) | {us_ipex:.0f} | 1.000x |")
    lines.append(f"| IPEX GEMM + ESIMD non-GEMM | {us_ie:.0f} | {us_ie/us_ipex:.3f}x |")
    cr = f"{us_cut/us_ipex:.3f}x" if us_cut==us_cut else "N/A"
    lines.append(f"| CUTLASS (xpu_fused_moe) | {'nan' if us_cut!=us_cut else f'{us_cut:.0f}'} | {cr} |")
    lines.append("")
    print(f"  done: IPEX={us_ipex:.0f} IE={us_ie:.0f} CUT={us_cut:.0f}")

# ═══ Section 3: Decode E2E ═══
def run_decode_e2e(lines):
    M = 8
    lines.append("## 3. Decode E2E (batch=8, one MoE layer)\n")
    lines.append("Full MoE forward including shared expert.\n")

    # IPEX + ESIMD
    print("  IPEX / ESIMD decode...")
    W13_q, W13_s, W2_q, W2_s = make_ipex_weights_fast()
    fusion = make_ipex_fusion_fast()
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

    us_ipex = bench(lambda: fusion(x, False, TK, logits, True,
        topk_group=None, num_expert_group=None,
        custom_routing_function=None, scoring_func="softmax", activation="silu"))

    from custom_esimd_kernels_vllm.ops import moe_forward_full_int4
    sgu = torch.randn(TWO_I, H, dtype=DTYPE, device=DEVICE)*0.01
    sd  = torch.randn(H, I, dtype=DTYPE, device=DEVICE)*0.01
    sg  = torch.randn(1, H, dtype=DTYPE, device=DEVICE)*0.01
    sgus = torch.empty(0, dtype=DTYPE, device=DEVICE)
    sds  = torch.empty(0, dtype=DTYPE, device=DEVICE)
    try:
        us_esimd = bench(lambda: moe_forward_full_int4(
            x, logits, W13_q, W13_s, sgu, sgus, W2_q, W2_s, sd, sds, sg,
            TK, 1, E, False))
    except Exception as e:
        us_esimd = float('nan'); print(f"    ESIMD: {e}")
    del x, logits, fusion, sgu, sd, sg, sgus, sds, W13_q, W13_s, W2_q, W2_s
    torch.xpu.empty_cache(); gc.collect()

    # CUTLASS
    print("  CUTLASS decode...")
    cW13, cW13s, cW2, cW2s = make_cutlass_weights_fast()
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE)*0.1
    rt = make_routing(M)
    tw_f32 = rt["tw"].float()
    try:
        us_cut = bench(lambda: xpu_fused_moe(
            x, cW13, cW13s, None, cW2, cW2s, None,
            tw_f32, rt["ti"].to(torch.int64), TK, "silu", E, is_int4=True))
    except Exception as e:
        us_cut = float('nan'); print(f"    CUTLASS: {e}")
    del x, cW13, cW13s, cW2, cW2s, rt, tw_f32; torch.xpu.empty_cache(); gc.collect()

    lines.append("| Path | μs | vs IPEX |")
    lines.append("|---|---:|---:|")
    lines.append(f"| IPEX (GatedMLPMOE) | {us_ipex:.0f} | 1.000x |")
    er = f"{us_esimd/us_ipex:.3f}x" if us_esimd==us_esimd else "N/A"
    lines.append(f"| ESIMD decode (moe_forward_full_int4) | {'nan' if us_esimd!=us_esimd else f'{us_esimd:.0f}'} | {er} |")
    cr = f"{us_cut/us_ipex:.3f}x" if us_cut==us_cut else "N/A"
    lines.append(f"| CUTLASS (xpu_fused_moe) | {'nan' if us_cut!=us_cut else f'{us_cut:.0f}'} | {cr} |")
    lines.append("")
    print(f"  done: IPEX={us_ipex:.0f} ESIMD={us_esimd:.0f} CUT={us_cut:.0f}")


def main():
    lines = ["# MoE INT4 Kernel Comparison — Qwen3.5-122B-A10B TP=4\n"]
    lines.append(f"H={H}, I={I}, E={E}, TK={TK}, GS={GS}, dtype=fp16, single card\n")

    t0 = time.time()
    print("=== GEMM-only ===")
    run_gemm(lines)
    print("=== Prefill E2E ===")
    run_prefill_e2e(lines)
    print("=== Decode E2E ===")
    run_decode_e2e(lines)
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    md = "\n".join(lines)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)
    print(f"Report: {OUTPUT_MD}\n")
    print(md)

if __name__ == "__main__":
    main()
