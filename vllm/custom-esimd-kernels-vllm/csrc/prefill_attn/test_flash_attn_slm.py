"""Task 3 verification: SLM cooperative V load.

Key validation:
  1. Results must be identical to Task 2 (same algorithm, only V load path changed)
  2. No hangs (barrier correctness)
  3. Performance comparison vs Task 2

Usage:
  KERNEL=flash_attn_slm TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace
  python3 test_flash_attn_slm.py
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

script_dir = os.path.dirname(os.path.abspath(__file__))
so_files = glob.glob(os.path.join(script_dir, "flash_attn_slm*.so"))
if not so_files:
    print("ERROR: Build first with:")
    print("  KERNEL=flash_attn_slm TORCH_XPU_ARCH_LIST=bmg-g21 python3 setup_test.py build_ext --inplace")
    sys.exit(1)
ctypes.CDLL(so_files[0])
forward = torch.ops.flash_attn_slm.forward

device = torch.device("xpu:0")
Q_GROUP = 32


def reference_sdpa(Q, K, V):
    q = Q.unsqueeze(0).unsqueeze(0).float()
    k = K.unsqueeze(0).unsqueeze(0).float()
    v = V.unsqueeze(0).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return out.squeeze(0).squeeze(0).half()


# ─── Test 1: Basic correctness ──────────────────────────────────────
print("=== Test 1: Correctness vs SDPA ===")
pass1 = True
cases = [
    (1, 32),
    (4, 64),
    (32, 128),
    (32, 256),
    (64, 256),
    (128, 512),
    (256, 1024),
    (31, 100),    # non-aligned Q
    (33, 65),     # non-aligned both
    (512, 2048),
]
for q_len, kv_len in cases:
    torch.manual_seed(q_len * 1000 + kv_len)
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
        pass1 = False
        if not has_nan:
            print(f"    max_diff={max_diff:.6f}")

print(f"\n  Result: {'ALL PASS' if pass1 else 'SOME FAILED'}\n")

# ─── Test 2: Compare with Task 2 (bit-level comparison) ─────────────
print("=== Test 2: Compare with Task 2 (should be near-identical) ===")
pass2 = True
so_t2 = glob.glob(os.path.join(script_dir, "flash_attn_parallel_q*.so"))
if so_t2:
    ctypes.CDLL(so_t2[0])
    fwd_t2 = torch.ops.flash_attn_parallel_q.forward

    cases_cmp = [(32, 128), (64, 256), (128, 512), (256, 1024)]
    for q_len, kv_len in cases_cmp:
        torch.manual_seed(q_len * 100 + kv_len)
        Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
        K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)

        out_slm = forward(Q, K, V)
        out_t2 = fwd_t2(Q, K, V)

        max_diff = (out_slm.float() - out_t2.float()).abs().max().item()
        # Should be very close (same algorithm, just different V load path)
        passed = max_diff < 0.001
        print(f"  Q={q_len:3d} KV={kv_len:4d}: max_diff_vs_task2={max_diff:.8f}  [{'PASS' if passed else 'FAIL'}]")
        if not passed:
            pass2 = False
else:
    print("  (Task 2 .so not found, skipping comparison)")

print(f"\n  Result: {'ALL PASS' if pass2 else 'SOME FAILED'}\n")

# ─── Benchmark ──────────────────────────────────────────────────────
if args.bench:
    print("=== Benchmark: Task 3 (SLM V) vs Task 2 (global V) ===")
    bench_cases = [
        (32, 256),
        (32, 1024),
        (128, 1024),
        (256, 4096),
        (512, 4096),
    ]
    for q_len, kv_len in bench_cases:
        Q = torch.randn(q_len, 256, device=device, dtype=torch.float16)
        K = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        V = torch.randn(kv_len, 256, device=device, dtype=torch.float16)
        num_wg = (q_len + Q_GROUP - 1) // Q_GROUP

        # Task 3 (SLM)
        for _ in range(3):
            forward(Q, K, V)
        torch.xpu.synchronize()
        N = 5 if kv_len >= 4096 else 10
        t0 = time.perf_counter()
        for _ in range(N):
            forward(Q, K, V)
        torch.xpu.synchronize()
        t_slm = (time.perf_counter() - t0) / N

        # Task 2 (global)
        if so_t2:
            for _ in range(3):
                fwd_t2(Q, K, V)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                fwd_t2(Q, K, V)
            torch.xpu.synchronize()
            t_global = (time.perf_counter() - t0) / N
            speedup = t_global / t_slm
            print(f"  Q={q_len:4d} KV={kv_len:4d} ({num_wg:2d} WGs): "
                  f"SLM={t_slm*1000:.2f}ms  Global={t_global*1000:.2f}ms  "
                  f"Speedup={speedup:.2f}x")
        else:
            print(f"  Q={q_len:4d} KV={kv_len:4d} ({num_wg:2d} WGs): SLM={t_slm*1000:.2f}ms")

# ─── Summary ────────────────────────────────────────────────────────
overall = pass1 and pass2
print(f"\n{'='*50}")
print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
print(f"{'='*50}")
sys.exit(0 if overall else 1)
