"""122B-TP4 non-GEMM kernel comparison: ESIMD vs IPEX vs vllm-xpu-kernels.

Compares individual non-GEMM stages of the MoE prefill pipeline.
Two cases: 32k×batch1 (M=32768) and 2k×batch16 (M=32768, same total).

Stages measured:
  1. topk        : softmax + topk + renormalize
  2. scatter     : reorder hidden_states by expert assignment
     ESIMD: gather_forward_v2 + scatter_x_forward (2 kernels)
     IPEX:  moe_rows_counts + moe_scatter (fused)
     XPU-K: remap_hidden_states (fused: scatter + offset computation)
  3. silu_mul    : gate_up → silu(gate) * up
  4. gather/acc  : weighted accumulate from expert-sorted output back to token order

Config: 122B-TP4 H=3072, I=1024, E=256, TK=8, GS=128

Usage: ZE_AFFINITY_MASK=2 python tests/bench_122b_nongemm.py
"""
import gc, sys, time, pathlib, torch
import intel_extension_for_pytorch as ipex
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP, ITERS = 5, 20
H, I, E, TK, GS = 3072, 1024, 256, 8, 128
TWO_I = 2 * I
OUTPUT_MD = "/llm/models/test/bench_122b_nongemm_comparison.md"

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def run_case(M, label, lines):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    total = M * TK
    torch.manual_seed(42)
    x = torch.randn(M, H, dtype=DTYPE, device=DEVICE) * 0.1
    router_logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)

    # Pre-compute topk once for consistent routing across all paths
    tw_esimd, ti_esimd = ops.moe_topk_softmax(router_logits, TK, E)

    lines.append(f"### {label} (M={M}, total_tokens={total})\n")
    lines.append("| Stage | ESIMD (μs) | IPEX (μs) | ipex/esimd | XPU-K (μs) | xpuk/esimd |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    results = {}

    # ════════════ 1. TOPK ════════════
    # ESIMD
    us_topk_e = bench(lambda: ops.moe_topk_softmax(router_logits, TK, E))

    # IPEX
    tw_ipex = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
    ti_ipex = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    tei_ipex = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    us_topk_i = bench(lambda: torch.ops.torch_ipex.topk_softmax(
        tw_ipex, ti_ipex, tei_ipex, router_logits, True))

    # XPU-K
    tw_xk = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
    ti_xk = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    tei_xk = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    us_topk_x = bench(lambda: torch.ops._moe_C.topk_softmax(
        tw_xk, ti_xk, tei_xk, router_logits, True, None))

    results["topk"] = (us_topk_e, us_topk_i, us_topk_x)
    lines.append(f"| topk | {us_topk_e:.1f} | {us_topk_i:.1f} | {us_topk_i/us_topk_e:.3f}x | {us_topk_x:.1f} | {us_topk_x/us_topk_e:.3f}x |")

    # ════════════ 2. SCATTER / REMAP ════════════
    # ESIMD: gather_forward_v2 (offsets + tokens + rows) + scatter_x_forward
    def _scatter_esimd():
        off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti_esimd, E)
        return ops.moe_prefill_scatter_x_forward(x, tok, TK)
    us_scatter_e = bench(_scatter_esimd)

    # IPEX: moe_rows_counts + moe_scatter (fused pair)
    # ipex topk produces int32 topk_ids — use ti_ipex from topk bench
    torch.ops.torch_ipex.topk_softmax(tw_ipex, ti_ipex, tei_ipex, router_logits, True)
    def _scatter_ipex():
        rows_i, offsets_i = torch.ops.torch_ipex.moe_rows_counts(ti_ipex, 0, E)
        return torch.ops.torch_ipex.moe_scatter(
            x, rows_i, ti_ipex, offsets_i, 0, E, TK)
    us_scatter_i = bench(_scatter_ipex)

    # XPU-K: remap_hidden_states (fused scatter + offset computation)
    torch.ops._moe_C.topk_softmax(tw_xk, ti_xk, tei_xk, router_logits, True, None)
    ti_xk_i64 = ti_xk.to(torch.int64)
    remapped = torch.empty(total, H, dtype=DTYPE, device=DEVICE)
    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    u2p = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    def _scatter_xk():
        efto.zero_()
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=x, hidden_states_scales=None,
            remapped_hidden_states=remapped, remapped_hidden_states_scales=None,
            expert_map=None,
            expert_first_token_offset=efto,
            unpermuted_row_to_permuted_row=u2p,
            topk_ids=ti_xk_i64,
            total_experts_num=E, local_experts_num=E)
    us_scatter_x = bench(_scatter_xk)

    results["scatter"] = (us_scatter_e, us_scatter_i, us_scatter_x)
    lines.append(f"| scatter/remap | {us_scatter_e:.1f} | {us_scatter_i:.1f} | {us_scatter_i/us_scatter_e:.3f}x | {us_scatter_x:.1f} | {us_scatter_x/us_scatter_e:.3f}x |")

    # ════════════ 3. SILU_MUL ════════════
    gate_up = torch.randn(total, TWO_I, dtype=DTYPE, device=DEVICE) * 0.1

    # ESIMD
    us_silu_e = bench(lambda: ops.moe_prefill_silu_mul_forward(gate_up))

    # IPEX
    act_ipex = torch.empty(total, I, dtype=DTYPE, device=DEVICE)
    us_silu_i = bench(lambda: torch.ops.torch_ipex.silu_and_mul(gate_up, act_ipex))

    # XPU-K
    act_xk = torch.empty(total, I, dtype=DTYPE, device=DEVICE)
    us_silu_x = bench(lambda: torch.ops._C.silu_and_mul(act_xk, gate_up))

    results["silu_mul"] = (us_silu_e, us_silu_i, us_silu_x)
    lines.append(f"| silu_mul | {us_silu_e:.1f} | {us_silu_i:.1f} | {us_silu_i/us_silu_e:.3f}x | {us_silu_x:.1f} | {us_silu_x/us_silu_e:.3f}x |")

    del gate_up, act_ipex, act_xk; torch.xpu.empty_cache()

    # ════════════ 4. GATHER / ACCUMULATE ════════════
    down_output = torch.randn(total, H, dtype=DTYPE, device=DEVICE) * 0.1

    # ESIMD: accumulate_permuted_forward_v2 (fused inverse-permute + weighted sum)
    off_e, tok_e, p2p_e, rows_e = ops.moe_prefill_gather_forward_v2(ti_esimd, E)
    us_gather_e = bench(lambda: ops.moe_prefill_accumulate_permuted_forward_v2(
        down_output, tw_esimd, p2p_e))

    # IPEX: moe_gather (fused: renormalize + inverse-permute + weighted sum)
    torch.ops.torch_ipex.topk_softmax(tw_ipex, ti_ipex, tei_ipex, router_logits, True)
    rows_i, offsets_i = torch.ops.torch_ipex.moe_rows_counts(ti_ipex, 0, E)
    _reordered, mapped_slot = torch.ops.torch_ipex.moe_scatter(
        x, rows_i, ti_ipex, offsets_i, 0, E, TK)
    # moe_gather needs the reordered output and mapped_slot from scatter
    reordered_output = torch.randn(total, H, dtype=DTYPE, device=DEVICE) * 0.1
    us_gather_i = bench(lambda: torch.ops.torch_ipex.moe_gather(
        reordered_output, tw_ipex, mapped_slot, E, TK, True))

    # XPU-K: moe_gather
    torch.ops._moe_C.topk_softmax(tw_xk, ti_xk, tei_xk, router_logits, True, None)
    ti_xk_i64 = ti_xk.to(torch.int64)
    efto.zero_()
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=x, hidden_states_scales=None,
        remapped_hidden_states=remapped, remapped_hidden_states_scales=None,
        expert_map=None, expert_first_token_offset=efto,
        unpermuted_row_to_permuted_row=u2p,
        topk_ids=ti_xk_i64, total_experts_num=E, local_experts_num=E)
    gather_out_xk = torch.empty(M, H, dtype=DTYPE, device=DEVICE)
    gemm2_out_xk = torch.randn(total, H, dtype=DTYPE, device=DEVICE) * 0.1
    us_gather_x = bench(lambda: torch.ops._moe_C.moe_gather(
        gather_out_xk, gemm2_out_xk, tw_xk, u2p, efto, E))

    results["gather"] = (us_gather_e, us_gather_i, us_gather_x)
    lines.append(f"| gather/accumulate | {us_gather_e:.1f} | {us_gather_i:.1f} | {us_gather_i/us_gather_e:.3f}x | {us_gather_x:.1f} | {us_gather_x/us_gather_e:.3f}x |")

    # ════════════ TOTAL ════════════
    sum_e = sum(v[0] for v in results.values())
    sum_i = sum(v[1] for v in results.values())
    sum_x = sum(v[2] for v in results.values())
    lines.append(f"| **Total non-GEMM** | **{sum_e:.0f}** | **{sum_i:.0f}** | **{sum_i/sum_e:.3f}x** | **{sum_x:.0f}** | **{sum_x/sum_e:.3f}x** |")
    lines.append("")

    print(f"  {label}: ESIMD={sum_e:.0f} IPEX={sum_i:.0f} XPU-K={sum_x:.0f}")

    del x, router_logits, down_output, remapped, gather_out_xk, gemm2_out_xk
    del reordered_output
    torch.xpu.empty_cache(); gc.collect()


def main():
    lines = ["# Non-GEMM Kernel Comparison — Qwen3.5-122B-A10B TP=4\n"]
    lines.append(f"H={H}, I={I}, E={E}, TK={TK}, GS={GS}, dtype=fp16, single card\n")
    lines.append("Comparing individual non-GEMM stages across three implementations.\n")
    lines.append("- **ESIMD**: our phase-1 kernels (moe_int4_prefill_ops)\n")
    lines.append("- **IPEX**: intel_extension_for_pytorch ops\n")
    lines.append("- **XPU-K**: vllm-xpu-kernels ops (_moe_C / _C)\n")

    t0 = time.time()
    run_case(32768, "32k × batch1", lines)
    run_case(32768, "2k × batch16 (same M=32768)", lines)
    print(f"\nTotal: {time.time()-t0:.1f}s")

    md = "\n".join(lines)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)
    print(f"Report: {OUTPUT_MD}\n")
    print(md)

if __name__ == "__main__":
    main()
