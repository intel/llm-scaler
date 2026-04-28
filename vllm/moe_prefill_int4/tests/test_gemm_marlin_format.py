"""Test: grouped GEMM INT4 with K-major marlin weights converted to N-major.

Verifies that:
  1. marlin_to_nmajor conversion is correct
  2. Our grouped_gemm_int4 on converted weights matches IPEX group_mm_int4_out_marlin
"""
import sys, pathlib, torch, time
import intel_extension_for_pytorch
import vllm_xpu_kernels
from vllm_xpu_kernels import _C, _xpu_C, _moe_C
from vllm_xpu_kernels.fused_moe_interface import implement_zp

torch.ops.load_library(
    "/llm/models/test/llm-scaler/vllm/moe_prefill_int4/build/libmoe_prefill_gemm_int4.so")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent /
    "custom-esimd-kernels-vllm" / "tests"))
from test_moe_prefill_int4 import quantize_experts_ipex, GROUP_SIZE

DEVICE = "xpu"
DTYPE = torch.float16
H, I, E, TK, GS = 3072, 1024, 64, 8, 128   # E=64 for speed
TWO_I = 2 * I

_MARLIN_SHUFFLED_IDX = (0, 4, 1, 5, 2, 6, 3, 7)


def marlin_to_nmajor(qw_marlin, scales_marlin):
    """Convert IPEX K-major marlin weights to XeTLA N-major uint8 format.

    Args:
        qw_marlin:     [E, K/8, N] int32 marlin-shuffled
        scales_marlin: [E, K/GS, N] fp16

    Returns:
        qw_nmajor:     [E, N, K/2] uint8 (signed int4 via implement_zp)
        scales_nmajor: [E, N, K/GS] fp16
    """
    E_dim, K_packed, N_dim = qw_marlin.shape
    K = K_packed * 8

    # 1. Un-shuffle marlin nibbles → natural order (vectorized on XPU)
    # marlin stores: slot i contains original nibble at position SHUFFLED_IDX[i]
    # To unshuffle: natural position k is stored at slot UNSHUFFLE[k]
    #   where UNSHUFFLE is the inverse of SHUFFLED_IDX
    UNSHUFFLE = [0, 2, 4, 6, 1, 3, 5, 7]
    qw = qw_marlin.to(torch.int64) & 0xFFFFFFFF
    unshuffled = torch.zeros_like(qw)
    for k in range(8):
        slot = UNSHUFFLE[k]
        nibble = (qw >> (slot * 4)) & 0xF
        unshuffled |= nibble << (k * 4)
    unshuffled = (unshuffled & 0xFFFFFFFF).to(torch.int32)

    # 2. Transpose [E, K/8, N] → [E, N, K/8]
    transposed = unshuffled.permute(0, 2, 1).contiguous()

    # 3. Repack int32 (8 nibbles, unsigned 0-15) → uint8 (2 nibbles)
    # Each int32 has nibbles at positions 0,1,...,7 in natural order.
    # Pack pairs into uint8: byte = lo_nibble | (hi_nibble << 4)
    result = torch.zeros(E_dim, N_dim, K // 2, dtype=torch.uint8,
                         device=qw_marlin.device)
    for kp in range(K_packed):
        for b in range(0, 8, 2):
            k_pair = kp * 4 + b // 2
            lo = (transposed[:, :, kp] >> (b * 4)) & 0xF
            hi = (transposed[:, :, kp] >> ((b + 1) * 4)) & 0xF
            result[:, :, k_pair] = (lo | (hi << 4)).to(torch.uint8)

    # 4. implement_zp: unsigned u4 → signed s4 (subtract 8, repack)
    result_s4 = implement_zp(result.view(-1, K // 2)).view(E_dim, N_dim, K // 2)
    result_s4.xpu_fused_moe = True

    # 5. Scale transpose: [E, K/GS, N] → [E, N, K/GS]
    scales_nm = scales_marlin.permute(0, 2, 1).contiguous()

    return result_s4, scales_nm


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def test_accuracy():
    from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

    print("Building weights (CPU quantize)...")
    torch.manual_seed(0)
    W13 = (torch.randn(E, TWO_I, H, dtype=torch.float32) * 0.02).to(DTYPE)
    W13_q, W13_s, _ = quantize_experts_ipex(W13, GS)  # K-major marlin on CPU
    W13_q = W13_q.to(DEVICE)  # [E, K/8, N=2I] int32
    W13_s = W13_s.to(DEVICE)  # [E, K/GS, N=2I] fp16

    # Convert to N-major
    print("Converting marlin → N-major...")
    W13_nm, W13_s_nm = marlin_to_nmajor(W13_q, W13_s)

    M = 512
    total = M * TK
    torch.manual_seed(42)

    # Routing (use ipex topk since E=64 not supported by esimd topk_v2)
    logits = torch.randn(M, E, dtype=DTYPE, device=DEVICE)
    tw = torch.empty(M, TK, dtype=torch.float32, device=DEVICE)
    ti = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    tei = torch.empty(M, TK, dtype=torch.int32, device=DEVICE)
    torch.ops.torch_ipex.topk_softmax(tw, ti, tei, logits, True)
    tw = tw.to(DTYPE)
    off, tok, p2p, rows = ops.moe_prefill_gather_forward_v2(ti, E)
    efto = torch.zeros(E + 1, dtype=torch.int64, device=DEVICE)
    efto[:E] = off.to(torch.int64)
    efto[E] = total

    x_perm = ops.moe_prefill_scatter_x_forward(
        torch.randn(M, H, dtype=DTYPE, device=DEVICE) * 0.1, tok, TK)

    # IPEX marlin GEMM
    out_ipex = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
    torch.ops.torch_ipex.group_mm_int4_out_marlin(
        out_ipex, x_perm, W13_q, W13_s, None, rows, None, GS)

    # Our GEMM with converted weights
    out_ours = torch.empty(total, TWO_I, dtype=DTYPE, device=DEVICE)
    torch.ops.moe_prefill_gemm.grouped_gemm_int4(
        x_perm, W13_nm, W13_s_nm, None, out_ours, efto, TWO_I, H, E, True, False)

    diff = (out_ipex.float() - out_ours.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        out_ipex.float().flatten().unsqueeze(0),
        out_ours.float().flatten().unsqueeze(0)).item()

    print(f"Accuracy: max_diff={max_diff:.4e} mean_diff={mean_diff:.4e} cos_sim={cos:.6f}")
    assert max_diff < 0.1, f"max_diff too large: {max_diff}"
    assert cos > 0.99, f"cos_sim too low: {cos}"
    print("PASS")

    # Performance
    print("\nPerformance (M=512, W13):")
    us_ipex = bench(lambda: torch.ops.torch_ipex.group_mm_int4_out_marlin(
        out_ipex, x_perm, W13_q, W13_s, None, rows, None, GS))
    us_ours = bench(lambda: torch.ops.moe_prefill_gemm.grouped_gemm_int4(
        x_perm, W13_nm, W13_s_nm, None, out_ours, efto, TWO_I, H, E, True, False))
    print(f"  IPEX: {us_ipex:.0f}us  Ours: {us_ours:.0f}us  ratio: {us_ours/us_ipex:.3f}x")


if __name__ == "__main__":
    test_accuracy()
