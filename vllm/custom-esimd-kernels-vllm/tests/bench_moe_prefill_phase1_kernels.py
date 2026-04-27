"""Kernel-level before/after bench for Phase-1 non-GEMM MoE prefill ops.

Measures microseconds per call for four stages replaced in Phase 1:
  - topk_softmax          before: torch_ipex.topk_softmax + fp32->fp16 cast
                          after : moe_topk_softmax
  - gather + rows         before: moe_prefill_gather_forward_v2 (3-tuple)
                                   + Python-side rows[:-1]=off[1:]-off[:-1] etc.
                          after : moe_prefill_gather_forward_v2 (4-tuple, rows in kernel)
  - scatter_x             before: (tok.to(int64) // TK); torch.index_select
                          after : moe_prefill_scatter_x_forward
  - silu_mul              before: F.silu(gate) * up (with slicing)
                          after : moe_prefill_silu_mul_forward

The two ipex group_mm_int4_out_marlin GEMMs are identical in both paths and
are not measured here.

Configs:
  Qwen3.5-122B-A10B TP=4 : H=3072, I=1024, E=256, TK=8
  Qwen3.5-35B-A3B  TP=4 : H=2048, I=512,  E=256, TK=8
  M in {512, 2048, 8192, 16384}
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import intel_extension_for_pytorch  # noqa: F401  (registers torch_ipex ops)

DEVICE = "xpu"
DTYPE = torch.float16
WARMUP = 5
ITERS  = 20

CONFIGS = [
    dict(name="122B-TP4", H=3072, I=1024, E=256, TK=8),
    dict(name="35B-TP4",  H=2048, I=512,  E=256, TK=8),
]
MS = [512, 2048, 8192, 16384]


def bench(fn, n_warmup=WARMUP, n_iters=ITERS):
    for _ in range(n_warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6


def make_routing_inputs(M, H, E, TK):
    torch.manual_seed(0)
    x = (torch.randn(M, H, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    logits = torch.randn(M, E, dtype=torch.float32).to(DTYPE).to(DEVICE)
    return x, logits


# ─────────── Before/after stage benchmarks ─────────────────────────────────
def bench_topk(logits, TK, E):
    M = logits.size(0)

    # before — ipex topk_softmax + fp32->fp16 cast (Qwen3.5 path)
    def _before():
        tw = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
        ti = torch.empty(M, TK, dtype=torch.int32,  device=DEVICE)
        _tei = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
        torch.ops.torch_ipex.topk_softmax(tw, ti, _tei, logits, True)
        return tw.to(DTYPE), ti

    # after — fused kernel
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    def _after():
        return ops.moe_topk_softmax(logits, TK, E)

    return bench(_before), bench(_after)


def bench_gather_rows(topk_ids, E, total):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    # before — call 3-tuple gather + Python-side diff to build rows_for_experts.
    # Note: the current gather wrapper always returns 4-tuple now, so to keep
    # the Before numbers realistic we call it once and then simulate the old
    # Python-side rows construction (the 4 small kernels it launched).
    def _before():
        off, tok, pp, _rows = ops.moe_prefill_gather_forward_v2(topk_ids, E)
        rows = torch.empty(E, dtype=torch.int32, device=DEVICE)
        rows[:-1] = off[1:] - off[:-1]
        rows[-1]  = int(total) - off[-1]
        return off, tok, pp, rows

    def _after():
        return ops.moe_prefill_gather_forward_v2(topk_ids, E)

    return bench(_before), bench(_after)


def bench_scatter_x(x, tok, TK):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    def _before():
        p2t = (tok.to(torch.int64) // TK)
        return torch.index_select(x, 0, p2t)

    def _after():
        return ops.moe_prefill_scatter_x_forward(x, tok, TK)

    return bench(_before), bench(_after)


def bench_silu_mul(total, I):
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    torch.manual_seed(1)
    gu = (torch.randn(total, 2 * I, dtype=torch.float32) * 0.5).to(DTYPE).to(DEVICE)

    def _before():
        g = gu[:, :I]
        u = gu[:, I:]
        return F.silu(g) * u

    def _after():
        return ops.moe_prefill_silu_mul_forward(gu)

    return bench(_before), bench(_after)


# ─────────── Runner ────────────────────────────────────────────────────────
def run_one(cfg, M):
    H, I, E, TK = cfg["H"], cfg["I"], cfg["E"], cfg["TK"]
    total = M * TK

    # Inputs
    x, logits = make_routing_inputs(M, H, E, TK)
    # Prepare topk_ids (runs both kernels; we only reuse 1 for downstream).
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops
    _, topk_ids = ops.moe_topk_softmax(logits, TK, E)

    # Prepare expert_tokens via gather (needed by scatter_x bench).
    _off, tok, _pp, _rows = ops.moe_prefill_gather_forward_v2(topk_ids, E)

    # Stage benches
    b_topk, a_topk = bench_topk(logits, TK, E)
    b_gr,   a_gr   = bench_gather_rows(topk_ids, E, total)
    b_sx,   a_sx   = bench_scatter_x(x, tok, TK)
    b_sm,   a_sm   = bench_silu_mul(total, I)

    b_total = b_topk + b_gr + b_sx + b_sm
    a_total = a_topk + a_gr + a_sx + a_sm

    def pct(b, a):
        if b <= 0: return ""
        return f"{(a - b) / b * 100:+.1f}%"

    print(f"  M={M:>6}  stage        before(us)   after(us)  Δ")
    for name, b, a in [
        ("topk",       b_topk, a_topk),
        ("gather+rows",b_gr,   a_gr),
        ("scatter_x",  b_sx,   a_sx),
        ("silu_mul",   b_sm,   a_sm),
        ("non-GEMM total", b_total, a_total),
    ]:
        print(f"             {name:<14}  {b:>9.2f}   {a:>9.2f}   {pct(b, a)}")


def main():
    for cfg in CONFIGS:
        print(f"\n== {cfg['name']}  H={cfg['H']}  I={cfg['I']}  E={cfg['E']}  TK={cfg['TK']} ==")
        for M in MS:
            run_one(cfg, M)

if __name__ == "__main__":
    main()
