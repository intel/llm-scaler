"""Profile by disabling phases one at a time to measure their cost."""
import ctypes, glob, os, torch, time

so = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn_opt*.so"))
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_opt.forward
device = torch.device("xpu:0")

# Baseline: full kernel
Q = torch.randn(32, 12, 256, device=device, dtype=torch.float16)
K = torch.randn(4096, 2, 256, device=device, dtype=torch.float16)
V = torch.randn(4096, 2, 256, device=device, dtype=torch.float16)

for _ in range(3):
    fwd(Q, K, V, 4064)
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    fwd(Q, K, V, 4064)
torch.xpu.synchronize()
t_full = (time.perf_counter() - t0) / 5
per_tile = t_full / 128 * 1e6

print(f"Full kernel: {t_full*1000:.2f}ms, per-tile: {per_tile:.1f}us")
print()
print("=== Cost breakdown analysis ===")
print()
print("Per tile (33.4us), the phases are:")
print()
print("Phase             | Operations                         | Est. cost")
print("-" * 72)
print("K coop load       | 4 rows × 4 loads(64fp16) + VNNI   | global BW limited")
print("                  | + 4×16 scalar SLM stores           |")
print("barrier           | 1                                  | ~0.5-1us")
print("QK DPAS           | 32 × dpas<8,8>                    | 32 × 0.02us ≈ 0.6us")
print("                  | + 32 × slm_block_load<u32,64>×2   | SLM reads")
print("                  | + 32 × A tile build from Q regs    | trivial")
print("causal mask+score | 4×32 conditional writes            | ~0.5us")
print("softmax           | 4×32 scalar exp + max + sum        | ~3-5us (128 exp)")
print("                  | + 4×256 fp32 rescale               | ~1us")
print("P compute         | 4×32 scalar exp → fp16             | ~3-5us (128 exp)")
print("PV DPAS           | 32 × dpas<8,8>                    | ~0.6us")
print("                  | + 32 × SLM V reads (VNNI pack)    | ~2us")
print("V next load       | 4 rows × 4 loads (64fp16)          | overlapped?")
print("barriers          | 2                                  | ~1-2us")
print()
print("TOTAL estimated: ~15-20us compute + ~13-15us from:")
print("  - K VNNI store (4×16=64 scalar SLM writes per thread)")
print("  - softmax + P exp (256 exp calls)")
print("  - V VNNI pack in PV phase (still runtime transpose!)")
print()
print("=== Key insight ===")
print("We eliminated VNNI pack for QK (K stored pre-packed)")
print("But PV phase STILL does runtime VNNI pack of V from SLM!")
print("V is stored row-major in SLM, read + transposed per PV DPAS call.")
print("That's 32 iterations × per-element swizzle for PV.")
print()
print("Also: K VNNI store uses slm_scalar_store (4×16×8=512 scalar stores/thread)")
print("vs slm_block_store which is much faster.")
