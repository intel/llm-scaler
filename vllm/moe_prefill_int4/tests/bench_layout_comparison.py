"""Compare N-major XeTLA vs IPEX K-major marlin GEMM performance.

This measures whether the K-major layout itself has a performance
difference vs N-major at the 2D block load level. If N-major and K-major
perform similarly, then adding K-major kernel support won't regress.
"""
import torch, time, gc
import intel_extension_for_pytorch
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import implement_zp

torch.ops.load_library(
    "/llm/models/test/llm-scaler/vllm/moe_prefill_int4/build/libmoe_prefill_gemm_int4.so")

DEVICE, DTYPE = "xpu", torch.float16
H, I, E, TK, GS = 3072, 1024, 256, 8, 128
TWO_I = 2 * I

def bench(fn, w=5, n=20):
    for _ in range(w): fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6

from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

# N-major weights
W_u8 = torch.randint(0, 0xFF, (E, TWO_I, H // 2), dtype=torch.uint8, device=DEVICE)
W_s_nm = (torch.rand(E, TWO_I, H // GS, device=DEVICE) * 0.04 + 0.002).to(DTYPE)
W_s4 = implement_zp(W_u8.view(-1, H // 2)).view(E, TWO_I, H // 2)
W_s4.xpu_fused_moe = True

# IPEX K-major marlin weights (random, for throughput test only)
W13_q_ipex = torch.randint(-2**30, 2**30, (E, H // 8, TWO_I), dtype=torch.int32, device=DEVICE)
W13_s_ipex = (torch.rand(E, H // GS, TWO_I, device=DEVICE) * 0.04 + 0.002).to(DTYPE)

print(f"122B-TP4: H={H}, I={I}, E={E}, TK={TK}")
print(f"{'M':>6} {'total':>8}  {'N-major(us)':>12} {'NM TF':>7}  {'IPEX-KM(us)':>12} {'IPEX TF':>8}  {'NM/IPEX':>8}")

for M in [4096, 8192, 16384, 32768]:
    total = M * TK
    torch.manual_seed(1 + M)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw, ti = ops.moe_topk_softmax(logits, TK, E)
    off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64)
    efto[E] = total

    xp = torch.randn(total, H, dtype=DTYPE, device=DEVICE) * 0.1
    o1 = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
    o2 = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)

    us_nm = bench(lambda: torch.ops.moe_prefill_gemm.grouped_gemm_int4(
        xp, W_s4, W_s_nm, None, o1, efto, TWO_I, H, E, True, False))

    us_ipex = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(
        o2, xp, W13_q_ipex, W13_s_ipex, None, rows, None, GS))

    flops = 2.0 * total * TWO_I * H
    tf_nm = flops / us_nm / 1e6
    tf_ipex = flops / us_ipex / 1e6
    ratio = us_nm / us_ipex

    print(f"{M:>6} {total:>8}  {us_nm:>12.0f} {tf_nm:>7.1f}  {us_ipex:>12.0f} {tf_ipex:>8.1f}  {ratio:>8.3f}x")

    del xp, o1, o2, logits, off, tok, rows, efto
    torch.xpu.empty_cache()
