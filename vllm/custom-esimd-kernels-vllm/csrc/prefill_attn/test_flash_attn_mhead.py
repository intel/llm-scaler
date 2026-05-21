"""Task 5: Multi-Head + GQA correctness + performance.

Usage:
  USE_DOUBLE_GRF=1 KERNEL=flash_attn_mhead TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  ZE_AFFINITY_MASK=4 python3 test_flash_attn_mhead.py
"""
import ctypes, glob, os, sys, torch, time
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
so = glob.glob(os.path.join(script_dir, "flash_attn_mhead*.so"))
if not so:
    print("Build first: USE_DOUBLE_GRF=1 KERNEL=flash_attn_mhead TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace")
    sys.exit(1)
ctypes.CDLL(so[0])
fwd = torch.ops.flash_attn_mhead.forward
device = torch.device("xpu:0")


def reference_sdpa(Q, K, V):
    """Q: [q_len, qh, 256], K: [kv_len, kvh, 256], V: [kv_len, kvh, 256]"""
    qh = Q.size(1)
    kvh = K.size(1)
    gqa_ratio = qh // kvh
    # Expand KV heads to match Q heads
    K_exp = K.repeat_interleave(gqa_ratio, dim=1)  # [kv_len, qh, 256]
    V_exp = V.repeat_interleave(gqa_ratio, dim=1)
    # SDPA: [B, H, S, D]
    q = Q.permute(1, 0, 2).unsqueeze(0).float()      # [1, qh, q_len, 256]
    k = K_exp.permute(1, 0, 2).unsqueeze(0).float()   # [1, qh, kv_len, 256]
    v = V_exp.permute(1, 0, 2).unsqueeze(0).float()   # [1, qh, kv_len, 256]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.squeeze(0).permute(1, 0, 2).half()  # [q_len, qh, 256]


print(f"Device: {torch.xpu.get_device_name(0)}")

# === Test 1: MHA (equal heads) ===
print("\n=== Test 1: MHA (qh=kvh) ===")
pass1 = True
cases = [(32, 64, 4, 4), (64, 128, 8, 8), (128, 256, 12, 12)]
for ql, kvl, qh, kvh in cases:
    torch.manual_seed(ql + kvl + qh)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V)
    expected = reference_sdpa(Q, K, V)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass1 = False
    print(f"  Q={ql:3d} KV={kvl:3d} H={qh}/{kvh}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass1 else 'FAILED'}")

# === Test 2: GQA (Qwen3.5 config: 12Q/2KV) ===
print("\n=== Test 2: GQA (12Q / 2KV, ratio=6) ===")
pass2 = True
cases = [(32, 64, 12, 2), (64, 128, 12, 2), (128, 256, 12, 2), (256, 1024, 12, 2)]
for ql, kvl, qh, kvh in cases:
    torch.manual_seed(ql + kvl)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V)
    expected = reference_sdpa(Q, K, V)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass2 = False
    print(f"  Q={ql:3d} KV={kvl:4d} H={qh}/{kvh}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass2 else 'FAILED'}")

# === Test 3: Other GQA ratios ===
print("\n=== Test 3: Other GQA ratios ===")
pass3 = True
cases = [(64, 128, 8, 2), (64, 128, 8, 4), (64, 128, 6, 3)]
for ql, kvl, qh, kvh in cases:
    torch.manual_seed(ql + kvl + qh + kvh)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V)
    expected = reference_sdpa(Q, K, V)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass3 = False
    print(f"  Q={ql:3d} KV={kvl:3d} H={qh}/{kvh} (ratio={qh//kvh}): rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass3 else 'FAILED'}")

# === Performance ===
print("\n=== Performance (Qwen3.5 config: 12Q/2KV) ===")
bench = [(32, 256, 12, 2), (128, 1024, 12, 2), (256, 2048, 12, 2), (512, 4096, 12, 2)]
for ql, kvl, qh, kvh in bench:
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    N = 5 if kvl >= 4096 else 10
    for _ in range(3): fwd(Q, K, V)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fwd(Q, K, V)
    torch.xpu.synchronize()
    t = (time.perf_counter() - t0) / N
    wg_q = (ql + 31) // 32
    total_wg = wg_q * qh
    print(f"  Q={ql:3d} KV={kvl:4d} H={qh}/{kvh} ({total_wg:4d} WGs): {t*1000:.2f}ms")

# Summary
overall = pass1 and pass2 and pass3
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
