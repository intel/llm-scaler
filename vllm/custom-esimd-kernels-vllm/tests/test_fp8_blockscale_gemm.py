"""Unit test: esimd_gemm_fp8_blockscale vs a dequant->fp32 matmul reference.

Run under the allowed cards, e.g.:
    ZE_AFFINITY_MASK=4,5 python3 tests/test_fp8_blockscale_gemm.py
"""
import torch
import custom_esimd_kernels_vllm as esimd

FP8_MAX = 448.0  # float8_e4m3fn max magnitude
BN = BK = 128


def quantize_weight_block(w_f32, bn=BN, bk=BK):
    """Quantize [N,K] fp32 -> (fp8_e4m3 weight, fp32 weight_scale_inv[Nb,Kb])."""
    N, K = w_f32.shape
    Nb, Kb = (N + bn - 1) // bn, (K + bk - 1) // bk
    wq = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w_f32.device)
    ws = torch.empty(Nb, Kb, dtype=torch.float32, device=w_f32.device)
    for i in range(Nb):
        for j in range(Kb):
            blk = w_f32[i * bn:(i + 1) * bn, j * bk:(j + 1) * bk]
            amax = blk.abs().max().clamp(min=1e-12)
            scale = (amax / FP8_MAX).to(torch.float32)
            ws[i, j] = scale
            wq[i * bn:(i + 1) * bn, j * bk:(j + 1) * bk] = (
                (blk / scale).to(torch.float8_e4m3fn)
            )
    return wq, ws


def dequant_weight(wq, ws, bn=BN, bk=BK):
    """[N,K] fp8 + [Nb,Kb] fp32 -> [N,K] fp32 dequantized weight."""
    N, K = wq.shape
    wf = wq.to(torch.float32)
    scale_full = ws.repeat_interleave(bn, 0)[:N].repeat_interleave(bk, 1)[:, :K]
    return wf * scale_full


def run_case(M, N, K, dev):
    torch.manual_seed(1234 + M * 131 + N + K)
    w = (torch.randn(N, K, device=dev) * 0.3)
    wq, ws = quantize_weight_block(w)
    w_deq = dequant_weight(wq, ws)

    a = (torch.randn(M, K, device=dev) * 0.5).to(torch.float16)
    ref = a.to(torch.float32) @ w_deq.t()  # [M, N]

    out = torch.empty(M, N, dtype=torch.float16, device=dev)
    esimd.esimd_gemm_fp8_blockscale(a, wq, ws, out, BN, BK)
    torch.xpu.synchronize()

    o = out.to(torch.float32)
    denom = ref.abs().mean().clamp(min=1e-6)
    rel = (o - ref).abs().mean() / denom
    max_rel = ((o - ref).abs() / ref.abs().clamp(min=1e-3)).max()
    cos = torch.nn.functional.cosine_similarity(
        o.flatten(), ref.flatten(), dim=0
    )
    ok = rel < 5e-2 and cos > 0.999
    print(f"M={M:<5} N={N:<6} K={K:<6} mean_rel={rel:.4e} "
          f"max_rel={max_rel:.3e} cos={cos:.6f}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    dev = "xpu"
    torch.xpu.synchronize()
    all_ok = True
    for K in (5120, 6144, 17408):
        for N in (1024, 5120, 6144):
            all_ok &= run_case(1, N, K, dev)      # decode
    # M>1 tiled path + remainder tile
    for M in (2, 7, 8, 9, 33, 64):
        all_ok &= run_case(M, 5120, 5120, dev)
    # N not a multiple of PPG (scalar tail store path)
    all_ok &= run_case(1, 1000, 5120, dev)
    all_ok &= run_case(4, 1000, 5120, dev)
    print("\nRESULT:", "ALL OK" if all_ok else "FAILURES PRESENT")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
