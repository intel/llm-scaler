"""Task 2 verification: multi-thread parallel Q flash attention.

Key validation:
  1. Multiple threads produce same result as single-thread reference
  2. Q lengths larger than Q_GROUP=32 work correctly (multiple workgroups)
  3. Non-aligned Q lengths (not multiple of 32) work correctly
  4. Performance improvement vs single-thread (Task 1)

Usage:
  KERNEL=flash_attn_parallel_q TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  python3 test_flash_attn_parallel_q.py
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
so_files = glob.glob(os.path.join(script_dir, "flash_attn_parallel_q*.so"))
if not so_files:
    print("ERROR: Build first with:")
    print("  KERNEL=flash_attn_parallel_q TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace")
    sys.exit(1)
ctypes.CDLL(so_files[0])
forward = torch.ops.flash_attn_parallel_q.forward

device = torch.device("xpu:0")
Q_GROUP = 32


def reference_sdpa(Q, K, V):
    """Reference using torch SDPA. Q/K/V are [seq, 256] fp16."""
    q = Q.unsqueeze(0).unsqueeze(0).float()
    k = K.unsqueeze(0).unsqueeze(0).float()
    v = V.unsqueeze(0).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.squeeze(0).squeeze(0).half()


# ─── Test 1: Basic correctness (small Q, single workgroup) ──────────
print("=== Test 1: Single Workgroup (Q <= 32) ===")
pass1 = True
cases1 = [
    (1, 64),
    (4, 64),
    (8, 64),
    (16, 128),
    (32, 128),    # exactly 1 workgroup
    (31, 100),    # partial last thread
    (1, 1),
]
for q_len, kv_len in cases1:
    torch.manual_seed(q_len * 1000 + kv_len)
    Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
    K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
    V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

    actual = forward(Q, K, V)
    expected = reference_sdpa(Q, K, V)

    max_diff = (actual.float() - expected.float()).abs().max().item()
    ref_scale = expected.float().abs().max().item() + 1e-6
    rel_err = max_diff / ref_scale
    passed = rel_err < 0.02
    print(f"  Q={q_len:3d} KV={kv_len:4d}: rel_err={rel_err:.6f}  [{'PASS' if passed else 'FAIL'}]")
    if not passed:
        pass1 = False

print(f"\n  Result: {'ALL PASS' if pass1 else 'SOME FAILED'}\n")

# ─── Test 2: Multiple workgroups (Q > 32) ───────────────────────────
print("=== Test 2: Multiple Workgroups (Q > 32) ===")
pass2 = True
cases2 = [
    (33, 128),    # 2 workgroups, second has only 1 row
    (64, 256),    # exactly 2 workgroups
    (128, 256),   # 4 workgroups
    (256, 512),   # 8 workgroups
    (100, 200),   # non-aligned
    (512, 1024),  # larger
]
for q_len, kv_len in cases2:
    torch.manual_seed(q_len * 1000 + kv_len)
    Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
    K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
    V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

    actual = forward(Q, K, V)
    expected = reference_sdpa(Q, K, V)

    max_diff = (actual.float() - expected.float()).abs().max().item()
    ref_scale = expected.float().abs().max().item() + 1e-6
    rel_err = max_diff / ref_scale
    passed = rel_err < 0.02
    print(f"  Q={q_len:3d} KV={kv_len:4d}: rel_err={rel_err:.6f}  [{'PASS' if passed else 'FAIL'}]")
    if not passed:
        pass2 = False
        print(f"    max_diff={max_diff:.6f} ref_scale={ref_scale:.2f}")

print(f"\n  Result: {'ALL PASS' if pass2 else 'SOME FAILED'}\n")

# ─── Test 3: Large Q with long KV ───────────────────────────────────
print("=== Test 3: Large Q + Long KV ===")
pass3 = True
cases3 = [
    (256, 2048),
    (512, 4096),
]
for q_len, kv_len in cases3:
    torch.manual_seed(q_len + kv_len)
    Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
    K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
    V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

    actual = forward(Q, K, V)
    expected = reference_sdpa(Q, K, V)

    max_diff = (actual.float() - expected.float()).abs().max().item()
    ref_scale = expected.float().abs().max().item() + 1e-6
    rel_err = max_diff / ref_scale
    has_nan = not torch.isfinite(actual).all().item()
    passed = rel_err < 0.02 and not has_nan
    extra = " [NaN!]" if has_nan else ""
    print(f"  Q={q_len:3d} KV={kv_len:4d}: rel_err={rel_err:.6f}{extra}  [{'PASS' if passed else 'FAIL'}]")
    if not passed:
        pass3 = False

print(f"\n  Result: {'ALL PASS' if pass3 else 'SOME FAILED'}\n")

# ─── Benchmark ──────────────────────────────────────────────────────
if args.bench:
    print("=== Benchmark ===")
    print("  (8 threads/WG, each thread independent KV loop)")
    bench_cases = [
        (32, 256),
        (32, 1024),
        (128, 1024),
        (256, 1024),
        (256, 4096),
        (512, 4096),
    ]
    for q_len, kv_len in bench_cases:
        Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
        K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        num_wg = (q_len + Q_GROUP - 1) // Q_GROUP
        # Warmup
        for _ in range(3):
            forward(Q, K, V)
        torch.xpu.synchronize()
        N = 5 if kv_len >= 4096 else 10
        t0 = time.perf_counter()
        for _ in range(N):
            forward(Q, K, V)
        torch.xpu.synchronize()
        elapsed = (time.perf_counter() - t0) / N
        print(f"  Q={q_len:4d} KV={kv_len:4d} ({num_wg:2d} WGs): {elapsed*1000:.2f} ms")

    # Compare with Task 1 (single-thread) for same workload
    print("\n  Comparison with Task 1 (single thread, Q=4 KV=1024):")
    # Load task1 if available
    so1 = glob.glob(os.path.join(script_dir, "flash_attn_minimal*.so"))
    if so1:
        ctypes.CDLL(so1[0])
        fwd1 = torch.ops.flash_attn_minimal.forward
        Q4 = torch.randn(4, 256, device=device, dtype=torch.float16)
        Kb = torch.randn(1024, 256, device=device, dtype=torch.float16)
        Vb = torch.randn(1024, 256, device=device, dtype=torch.float16)
        for _ in range(3):
            fwd1(Q4, Kb, Vb)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fwd1(Q4, Kb, Vb)
        torch.xpu.synchronize()
        t_task1 = (time.perf_counter() - t0) / 10

        # Task 2: same total Q=32 (1 WG), KV=1024
        Q32 = torch.randn(32, 256, device=device, dtype=torch.float16)
        for _ in range(3):
            forward(Q32, Kb, Vb)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            forward(Q32, Kb, Vb)
        torch.xpu.synchronize()
        t_task2 = (time.perf_counter() - t0) / 10

        print(f"    Task 1: Q=4  KV=1024: {t_task1*1000:.2f} ms (1 thread)")
        print(f"    Task 2: Q=32 KV=1024: {t_task2*1000:.2f} ms (8 threads, 8x Q)")
        print(f"    Task 2 processes 8x more Q rows in {t_task2/t_task1:.1f}x time")
    else:
        print("    (Task 1 .so not found, skip comparison)")


# ─── Summary ────────────────────────────────────────────────────────
overall = pass1 and pass2 and pass3
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
