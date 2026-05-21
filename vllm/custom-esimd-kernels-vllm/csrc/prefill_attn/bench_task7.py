"""Task 7: Benchmark ESIMD causal kernel vs torch SDPA (xetla backend).

GO/NO-GO decision point.
Tests Qwen3.5-27B realistic shapes: 12 Q heads, 2 KV heads, HD=256.

Usage:
  ZE_AFFINITY_MASK=4 python3 bench_task7.py
"""
import ctypes, glob, os, sys, torch, time
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
so = glob.glob(os.path.join(script_dir, "flash_attn_causal*.so"))
if not so:
    print("Build first: USE_DOUBLE_GRF=1 KERNEL=flash_attn_causal TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace")
    sys.exit(1)
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_causal.forward
device = torch.device("xpu:0")

NUM_Q_HEADS = 12
NUM_KV_HEADS = 2


def bench_esimd(Q, K, V, qsp, N=10):
    for _ in range(3):
        fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


def bench_sdpa(Q, K, V, qsp, N=10):
    """Torch SDPA with causal. Needs BHSD format."""
    qh, kvh = Q.size(1), K.size(1)
    r = qh // kvh
    ql, kvl = Q.size(0), K.size(0)
    K_e = K.repeat_interleave(r, dim=1)
    V_e = V.repeat_interleave(r, dim=1)
    q = Q.permute(1, 0, 2).unsqueeze(0)   # [1, qh, ql, 256]
    k = K_e.permute(1, 0, 2).unsqueeze(0)  # [1, qh, kvl, 256]
    v = V_e.permute(1, 0, 2).unsqueeze(0)

    if qsp == 0 and ql == kvl:
        # Standard causal
        for _ in range(3):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.xpu.synchronize()
    else:
        # Chunked prefill: manual mask
        mask = torch.zeros(ql, kvl, device=device, dtype=torch.float16)
        for i in range(ql):
            valid = min(qsp + i + 1, kvl)
            mask[i, valid:] = float('-inf')
        mask = mask.unsqueeze(0).unsqueeze(0)
        for _ in range(3):
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.xpu.synchronize()
    return (time.perf_counter() - t0) / N


print(f"Device: {torch.xpu.get_device_name(0)}")
print(f"Config: {NUM_Q_HEADS}Q / {NUM_KV_HEADS}KV heads, HD=256, causal")
print()
print(f"{'Scenario':<35s} {'ESIMD':>8s} {'SDPA':>8s} {'Ratio':>8s}")
print("-" * 65)

# Scenario 1: Standard causal (first chunk, q_len == kv_len)
scenarios = [
    # (q_len, kv_len, q_start_pos, description)
    (64, 64, 0, "First chunk Q=64"),
    (128, 128, 0, "First chunk Q=128"),
    (256, 256, 0, "First chunk Q=256"),
    (512, 512, 0, "First chunk Q=512"),
    # Chunked prefill: later iterations
    (128, 1024, 896, "Chunk Q=128 KV=1024"),
    (256, 2048, 1792, "Chunk Q=256 KV=2048"),
    (256, 4096, 3840, "Chunk Q=256 KV=4096"),
    (512, 4096, 3584, "Chunk Q=512 KV=4096"),
    # Larger (simulating 8K/32K scenarios per-layer)
    (256, 8192, 7936, "Chunk Q=256 KV=8K"),
]

for ql, kvl, qsp, desc in scenarios:
    Q = torch.randn(ql, NUM_Q_HEADS, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, NUM_KV_HEADS, 256, device=device, dtype=torch.float16)
    N = 5 if kvl >= 4096 else 10

    t_esimd = bench_esimd(Q, K, V, qsp, N)
    t_sdpa = bench_sdpa(Q, K, V, qsp, N)
    ratio = t_esimd / t_sdpa

    print(f"  {desc:<33s} {t_esimd*1000:6.2f}ms {t_sdpa*1000:6.2f}ms {ratio:6.1f}x")

print()
print("=" * 65)
print("NOTES:")
print("  ESIMD/SDPA ratio > 1 means ESIMD is slower")
print("  Target: ratio < 1 (ESIMD faster) or ratio < 2 (acceptable)")
print("  Current kernel: no K cooperative load, no K prefetch")
print("  Remaining optimization headroom: ~5-10x")
print("=" * 65)
