"""Profile per-tile cost to understand where time is spent."""
import ctypes, glob, os, torch, time

so = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn_opt*.so"))
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_opt.forward
device = torch.device("xpu:0")

Q = torch.randn(32, 12, 256, device=device, dtype=torch.float16)

print(f"{'KV':>6} {'Tiles':>6} {'Total ms':>10} {'Per-tile us':>12}")
for kvl in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    K = torch.randn(kvl, 2, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, 2, 256, device=device, dtype=torch.float16)
    qsp = max(0, kvl - 32)
    for _ in range(3):
        fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    t = (time.perf_counter() - t0) / N
    tiles = (kvl + 31) // 32
    per_tile_us = t / tiles * 1e6
    print(f"{kvl:6d} {tiles:6d} {t*1000:10.3f} {per_tile_us:12.1f}")

# Theoretical analysis
print("\n=== Theoretical per-tile breakdown (Q=32, 12 heads, KV_TILE=32) ===")
print("Per tile operations (per WG = 1 head, 8 threads):")
print("  K load: 32 rows × 256 × 2B = 16KB from global → SLM (cooperative, 1/8 per thread)")
print("  QK DPAS: 16 K_steps × 2 N_groups = 32 DPAS calls")
print("  VNNI pack: 2 N_groups × 16 K_steps × (16 element swizzle) = 32 × 16 scalar ops")
print("  Softmax: 4 Q_rows × 32 KV_positions × (exp + compare + add) = 128 exp calls")
print("  PV DPAS: 2 PV_K_steps × 16 N_groups = 32 DPAS calls")
print("  V load (next tile, overlapped): 32 rows × 256 × 2B = 16KB")
print("  Barriers: 3 per tile (K_load, V_done, V_next)")
print()
print("Likely bottlenecks:")
print("  1. VNNI pack: 32 iterations of scalar element-by-element transpose")
print("  2. Softmax: 128 scalar exp() calls per tile")
print("  3. SLM read bandwidth (K + V reads per tile)")
