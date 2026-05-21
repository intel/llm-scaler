"""Task 6: Causal mask correctness tests.

Usage:
  USE_DOUBLE_GRF=1 KERNEL=flash_attn_causal TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  ZE_AFFINITY_MASK=4 python3 test_flash_attn_causal.py
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


def reference_causal(Q, K, V, q_start_pos=0):
    """Reference causal attention with q_start_pos offset."""
    qh, kvh = Q.size(1), K.size(1)
    ql, kvl = Q.size(0), K.size(0)
    r = qh // kvh
    K_e = K.repeat_interleave(r, dim=1)
    V_e = V.repeat_interleave(r, dim=1)
    # SDPA [B, H, S, D]
    q = Q.permute(1, 0, 2).unsqueeze(0).float()
    k = K_e.permute(1, 0, 2).unsqueeze(0).float()
    v = V_e.permute(1, 0, 2).unsqueeze(0).float()

    if q_start_pos == 0 and ql == kvl:
        # Simple causal: Q and KV same length, standard lower-triangular
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        # Manual causal mask for chunked prefill
        # Q[i] at global position q_start_pos + i can see KV[0..q_start_pos+i]
        mask = torch.zeros(ql, kvl, device=device, dtype=torch.float32)
        for i in range(ql):
            q_global = q_start_pos + i
            # Can attend to KV positions 0..q_global (inclusive)
            valid_kv = min(q_global + 1, kvl)
            mask[i, valid_kv:] = float('-inf')
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, ql, kvl]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    return out.squeeze(0).permute(1, 0, 2).half()


print(f"Device: {torch.xpu.get_device_name(0)}")

# === Test 1: Standard causal (q_start_pos=0, q_len == kv_len) ===
print("\n=== Test 1: Standard Causal (q_len == kv_len) ===")
pass1 = True
cases = [(32, 32, 12, 2), (64, 64, 12, 2), (128, 128, 12, 2), (256, 256, 12, 2)]
for ql, kvl, qh, kvh in cases:
    torch.manual_seed(ql + kvl + qh)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V, 0)
    expected = reference_causal(Q, K, V, 0)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass1 = False
    print(f"  Q={ql:3d} KV={kvl:3d} H={qh}/{kvh}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass1 else 'FAILED'}")

# === Test 2: Chunked prefill (q_start_pos > 0, kv_len > q_len) ===
print("\n=== Test 2: Chunked Prefill (q_start_pos > 0) ===")
pass2 = True
cases = [
    (64, 128, 12, 2, 64),     # 2nd chunk: Q covers [64:128], KV covers [0:128]
    (64, 256, 12, 2, 192),    # last chunk: Q covers [192:256], KV covers [0:256]
    (128, 512, 12, 2, 384),   # Q covers [384:512], KV covers [0:512]
    (32, 1024, 12, 2, 992),   # Q covers [992:1024], KV covers [0:1024]
]
for ql, kvl, qh, kvh, qsp in cases:
    torch.manual_seed(ql + kvl + qsp)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V, qsp)
    expected = reference_causal(Q, K, V, qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass2 = False
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d} H={qh}/{kvh}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass2 else 'FAILED'}")

# === Test 3: First chunk (lots of causal masking, 50% skipped) ===
print("\n=== Test 3: First Chunk (heavy masking) ===")
pass3 = True
cases = [(64, 64, 12, 2, 0), (128, 128, 4, 4, 0), (256, 256, 12, 2, 0)]
for ql, kvl, qh, kvh, qsp in cases:
    torch.manual_seed(ql * 7 + kvl)
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    actual = fwd(Q, K, V, qsp)
    expected = reference_causal(Q, K, V, qsp)
    rel = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
    nan = not torch.isfinite(actual).all().item()
    ok = rel < 0.02 and not nan
    if not ok: pass3 = False
    print(f"  Q={ql:3d} KV={kvl:3d} qsp={qsp} H={qh}/{kvh}: rel_err={rel:.6f}  [{'PASS' if ok else 'FAIL'}]")
print(f"  {'ALL PASS' if pass3 else 'FAILED'}")

# === Performance ===
print("\n=== Performance (causal, Qwen3.5 config) ===")
bench = [
    (128, 128, 12, 2, 0),      # first chunk
    (128, 1024, 12, 2, 896),   # late chunk (most KV visible)
    (256, 2048, 12, 2, 1792),  # large late chunk
]
for ql, kvl, qh, kvh, qsp in bench:
    Q = torch.randn(ql, qh, 256, device=device, dtype=torch.float16)
    K = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    V = torch.randn(kvl, kvh, 256, device=device, dtype=torch.float16)
    N = 10
    for _ in range(3): fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fwd(Q, K, V, qsp)
    torch.xpu.synchronize()
    t = (time.perf_counter() - t0) / N
    print(f"  Q={ql:3d} KV={kvl:4d} qsp={qsp:4d}: {t*1000:.2f}ms")

# Summary
overall = pass1 and pass2 and pass3
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
