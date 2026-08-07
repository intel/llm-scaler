"""Microbenchmark esimd_gemm_fp8_blockscale across decode (M=1) and prefill M.

    ZE_AFFINITY_MASK=4,5 python3 tests/bench_fp8_blockscale_gemm.py
"""
import time
import torch
import custom_esimd_kernels_vllm as esimd

BN = BK = 128
DEV = "xpu"

# Representative Qwen3.6-27B (TP=2) per-rank linear shapes (N, K):
#   qkv_proj, o_proj, gate_up, down. TP=2 splits N (col-parallel) or K (row-parallel).
SHAPES = [
    ("qkv_proj", 4096, 5120),    # (6144+1024+1024)/... approx per-rank; use round
    ("o_proj",   5120, 3072),
    ("gate_up",  17408, 5120),
    ("down",     5120, 8704),
]
# keep K a multiple of 128
SHAPES = [(n, N - N % 128, K - K % 128) for (n, N, K) in SHAPES]


def make_weight(N, K):
    w = (torch.randn(N, K, device=DEV) * 0.2)
    Nb, Kb = (N + BN - 1) // BN, (K + BK - 1) // BK
    wq = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=DEV)
    ws = torch.empty(Nb, Kb, dtype=torch.float32, device=DEV)
    for i in range(Nb):
        blk = w[i * BN:(i + 1) * BN]
        for j in range(Kb):
            b = blk[:, j * BK:(j + 1) * BK]
            s = (b.abs().max() / 448).clamp(min=1e-12)
            ws[i, j] = s
            wq[i * BN:(i + 1) * BN, j * BK:(j + 1) * BK] = (b / s).to(torch.float8_e4m3fn)
    return wq, ws


def bench(M, N, K, wq, ws, iters=30, warmup=10):
    a = (torch.randn(M, K, device=DEV) * 0.5).to(torch.float16)
    out = torch.empty(M, N, dtype=torch.float16, device=DEV)
    for _ in range(warmup):
        esimd.esimd_gemm_fp8_blockscale(a, wq, ws, out, BN, BK)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        esimd.esimd_gemm_fp8_blockscale(a, wq, ws, out, BN, BK)
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # bytes: weight fp8 (N*K) dominant for decode; + activation + scale
    wbytes = N * K + (N // BN) * (K // BK) * 4 + M * K * 2 + M * N * 2
    flops = 2.0 * M * N * K
    return dt * 1e3, wbytes / dt / 1e9, flops / dt / 1e12


def main():
    for name, N, K in SHAPES:
        wq, ws = make_weight(N, K)
        print(f"\n== {name}  N={N} K={K} ==")
        for M in (1, 2, 4, 8, 64, 512, 2048, 4096, 8192):
            try:
                ms, gbps, tflops = bench(M, N, K, wq, ws)
                print(f"  M={M:<5} {ms:8.3f} ms   {gbps:7.1f} GB/s   {tflops:6.2f} TFLOP/s")
            except Exception as e:
                print(f"  M={M:<5} ERROR {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
