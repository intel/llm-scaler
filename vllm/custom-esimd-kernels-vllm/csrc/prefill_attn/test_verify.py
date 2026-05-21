"""Task 0 verification: DPAS HD=256 correctness and register pressure.

Run:
  python3 test_verify.py           # correctness only
  python3 test_verify.py --bench   # + timing benchmark
"""
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--bench", action="store_true", help="Run benchmark")
args = parser.parse_args()

# Load the compiled extension via torch.ops (TORCH_LIBRARY registration)
import ctypes, glob, os
script_dir = os.path.dirname(os.path.abspath(__file__))
so_files = glob.glob(os.path.join(script_dir, "test_dpas_hd256*.so"))
if not so_files:
    raise FileNotFoundError("Build test_dpas_hd256 first: python3 setup_test.py build_ext --inplace")
ctypes.CDLL(so_files[0])
forward = torch.ops.test_dpas_hd256.forward

device = torch.device("xpu:0")

# ─── Correctness test ───────────────────────────────────────────────
print("=== Correctness Test ===")

torch.manual_seed(42)
Q = torch.randn(4, 256, device=device, dtype=torch.float16)
K = torch.randn(32, 256, device=device, dtype=torch.float16)

# Reference: torch matmul (fp32 for precision)
expected = (Q.float() @ K.float().T)  # [4, 32] fp32

# Our kernel
actual = forward(Q, K)  # [4, 32] fp32

# Compare
max_diff = (actual - expected).abs().max().item()
mean_diff = (actual - expected).abs().mean().item()
rel_err = max_diff / expected.abs().max().item()

print(f"  Max absolute diff:  {max_diff:.6f}")
print(f"  Mean absolute diff: {mean_diff:.6f}")
print(f"  Max relative error: {rel_err:.6f}")
print(f"  Expected max value: {expected.abs().max().item():.2f}")

# fp16 input precision: matmul of 256-dim vectors, each multiplication
# has fp16 rounding. Expected error ~ sqrt(256) * eps_fp16 ~ 16 * 0.001 ~ 0.016
# We use fp32 accumulation so error should be much smaller.
passed = rel_err < 0.01
print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: rel_err < 0.01)")

if not passed:
    print("\n  First few values:")
    print(f"    expected[0,:8] = {expected[0,:8].tolist()}")
    print(f"    actual[0,:8]   = {actual[0,:8].tolist()}")

# ─── Multiple random tests ──────────────────────────────────────────
print("\n=== Random Input Tests ===")
all_pass = True
for trial in range(10):
    Q = torch.randn(4, 256, device=device, dtype=torch.float16)
    K = torch.randn(32, 256, device=device, dtype=torch.float16)
    expected = Q.float() @ K.float().T
    actual = forward(Q, K)
    rel_err = (actual - expected).abs().max().item() / (expected.abs().max().item() + 1e-6)
    if rel_err > 0.01:
        print(f"  Trial {trial}: FAIL (rel_err={rel_err:.6f})")
        all_pass = False
    else:
        print(f"  Trial {trial}: PASS (rel_err={rel_err:.6f})")

print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

# ─── Benchmark ──────────────────────────────────────────────────────
if args.bench:
    print("\n=== Benchmark ===")
    Q = torch.randn(4, 256, device=device, dtype=torch.float16)
    K = torch.randn(32, 256, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(20):
        forward(Q, K)
    torch.xpu.synchronize()

    # Time it
    N = 200
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        forward(Q, K)
    torch.xpu.synchronize()
    elapsed = (time.perf_counter() - t0) / N

    print(f"  Kernel time: {elapsed*1e6:.1f} us")
    print(f"  (This is 1 thread doing Q[4,256]×K[32,256]^T — not representative of final perf)")
    print(f"  Purpose: verify no register spill penalty)")

    # Compare with torch matmul overhead
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = Q.float() @ K.float().T
    torch.xpu.synchronize()
    elapsed_torch = (time.perf_counter() - t0) / N
    print(f"  torch matmul: {elapsed_torch*1e6:.1f} us (for reference)")
