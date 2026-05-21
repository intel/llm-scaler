"""Task 1 verification: single-thread flash attention with online softmax.

Tests:
  1. Correctness vs torch SDPA across various KV lengths
  2. Numerical stability at long KV (simulating 32K/128K via streaming)
  3. Edge cases (kv_len not multiple of KV_TILE=32)

Usage:
  KERNEL=flash_attn_minimal TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  python3 test_flash_attn_minimal.py
"""
import os
import sys
import ctypes
import glob
import argparse
import time

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--bench", action="store_true")
args = parser.parse_args()

# Load extension
script_dir = os.path.dirname(os.path.abspath(__file__))
so_files = glob.glob(os.path.join(script_dir, "flash_attn_minimal*.so"))
if not so_files:
    print("ERROR: Build first with:")
    print("  KERNEL=flash_attn_minimal TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace")
    sys.exit(1)
ctypes.CDLL(so_files[0])
forward = torch.ops.flash_attn_minimal.forward

device = torch.device("xpu:0")


def reference_sdpa(Q, K, V):
    """Compute reference using torch SDPA. Q/K/V are [seq, 256] fp16."""
    # SDPA expects [B, H, S, D]
    q = Q.unsqueeze(0).unsqueeze(0).float()  # [1, 1, q_len, 256]
    k = K.unsqueeze(0).unsqueeze(0).float()  # [1, 1, kv_len, 256]
    v = V.unsqueeze(0).unsqueeze(0).float()  # [1, 1, kv_len, 256]
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.squeeze(0).squeeze(0).half()   # [q_len, 256] fp16


# ─── Test 1: Basic correctness ──────────────────────────────────────
print("=== Test 1: Basic Correctness ===")
all_pass = True
test_cases = [
    (1, 32),     # minimal: exactly 1 KV tile
    (1, 64),     # 2 tiles
    (4, 32),     # full Q_ROWS, 1 tile
    (4, 64),     # full Q_ROWS, 2 tiles
    (4, 128),    # 4 tiles
    (4, 256),    # 8 tiles
    (2, 100),    # non-aligned kv_len
    (3, 33),     # odd kv_len (1 full tile + 1 partial)
    (4, 1),      # kv_len = 1
    (1, 1),      # minimal possible
]

for q_len, kv_len in test_cases:
    torch.manual_seed(42 + q_len * 100 + kv_len)
    Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
    K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
    V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

    actual = forward(Q, K, V)
    expected = reference_sdpa(Q, K, V)

    max_diff = (actual.float() - expected.float()).abs().max().item()
    mean_diff = (actual.float() - expected.float()).abs().mean().item()
    ref_scale = expected.float().abs().max().item() + 1e-6
    rel_err = max_diff / ref_scale

    passed = rel_err < 0.02  # fp16 output, allow 2% relative error
    status = "PASS" if passed else "FAIL"
    print(f"  Q={q_len:2d} KV={kv_len:4d}: rel_err={rel_err:.6f} max_diff={max_diff:.4f}  [{status}]")
    if not passed:
        all_pass = False
        print(f"    expected[0,:4] = {expected[0,:4].tolist()}")
        print(f"    actual[0,:4]   = {actual[0,:4].tolist()}")

print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")

# ─── Test 2: Longer KV sequences (stress online softmax) ────────────
print("=== Test 2: Long KV Sequences ===")
long_pass = True
long_cases = [512, 1024, 2048, 4096]

for kv_len in long_cases:
    torch.manual_seed(123 + kv_len)
    Q = torch.randn(4, 256, device=device, dtype=torch.float16)
    K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
    V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

    actual = forward(Q, K, V)
    expected = reference_sdpa(Q, K, V)

    max_diff = (actual.float() - expected.float()).abs().max().item()
    ref_scale = expected.float().abs().max().item() + 1e-6
    rel_err = max_diff / ref_scale

    # Check no NaN/Inf
    has_nan = not torch.isfinite(actual).all().item()
    passed = rel_err < 0.02 and not has_nan
    status = "PASS" if passed else "FAIL"
    extra = " [NaN!]" if has_nan else ""
    print(f"  KV={kv_len:5d}: rel_err={rel_err:.6f}{extra}  [{status}]")
    if not passed:
        long_pass = False

print(f"\n  Result: {'ALL PASS' if long_pass else 'SOME FAILED'}\n")

# ─── Test 3: Numerical stability (large scale values) ────────────────
print("=== Test 3: Numerical Edge Cases ===")
edge_pass = True

# Case: Q and K highly correlated → large QK scores
torch.manual_seed(999)
Q = torch.randn(4, 256, device=device, dtype=torch.float16)
K = Q[:1].expand(64, -1).contiguous()  # All K rows identical to Q[0]
V = torch.randn(64, 256, device=device, dtype=torch.float16)

actual = forward(Q, K, V)
has_nan = not torch.isfinite(actual).all().item()
print(f"  Correlated Q/K (kv=64): NaN={has_nan}  [{'FAIL' if has_nan else 'PASS'}]")
if has_nan:
    edge_pass = False

# Case: Very small values
Q_small = torch.randn(4, 256, device=device, dtype=torch.float16) * 0.001
K_small = torch.randn(128, 256, device=device, dtype=torch.float16) * 0.001
V_small = torch.randn(128, 256, device=device, dtype=torch.float16)
actual = forward(Q_small, K_small, V_small)
expected = reference_sdpa(Q_small, K_small, V_small)
rel_err = (actual.float() - expected.float()).abs().max().item() / (expected.float().abs().max().item() + 1e-6)
print(f"  Small Q/K (×0.001):     rel_err={rel_err:.6f}  [{'PASS' if rel_err < 0.02 else 'FAIL'}]")
if rel_err >= 0.02:
    edge_pass = False

print(f"\n  Result: {'ALL PASS' if edge_pass else 'SOME FAILED'}\n")

# ─── Benchmark (optional) ───────────────────────────────────────────
if args.bench:
    print("=== Benchmark (single thread — measures algorithm overhead, not final perf) ===")
    bench_cases = [(4, 32), (4, 128), (4, 512), (4, 1024), (4, 2048)]
    for q_len, kv_len in bench_cases:
        Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
        K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        # Warmup
        for _ in range(3):
            forward(Q, K, V)
        torch.xpu.synchronize()
        N = 10 if kv_len >= 1024 else 20
        t0 = time.perf_counter()
        for _ in range(N):
            forward(Q, K, V)
        torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) / N
        tiles = (kv_len + 31) // 32
        print(f"  Q={q_len} KV={kv_len:5d} ({tiles:3d} tiles): {elapsed*1000:.2f} ms")

# ─── Summary ────────────────────────────────────────────────────────
overall = all_pass and long_pass and edge_pass
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
