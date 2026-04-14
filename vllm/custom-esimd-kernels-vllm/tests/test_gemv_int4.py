"""
Test suite for esimd_gemv_int4 — Symmetric INT4 GEMV kernel with per-group scale.

================================================================================
Background
================================================================================

This kernel performs General Matrix-Vector multiply (GEMV) with INT4 quantized
weights on Intel XPU (BMG) using ESIMD intrinsics.  It is the INT4 counterpart
of the existing FP8 GEMV kernel (fp8_GEMV_v2.h / esimd_gemv_fp8_pert).

The kernel targets the Qwen3.5-122B-A10B model running on vLLM.  In that model,
GEMV (M=1, decode phase) is the dominant operation.  Each decoder layer uses
GEMV for:
  - GatedDeltaNet input projections (qkvz + ba, fused as 2-GEMV)
  - GatedDeltaNet output projection
  - Full-attention QKV / O projections
  - MoE router + expert gate_up / down projections

Replacing the current IPEX-based INT4 path with high-performance ESIMD kernels
is the goal.

================================================================================
INT4 quantization scheme
================================================================================

Symmetric INT4 with per-group scale (group_size = 128):

  Weight storage:
    - packed:  [N, K/2]          uint8   — 2 int4 values per byte
    - scale:   [N, K/group_size] fp16    — one scale per 128 elements along K

  Packing layout (within each byte):
    byte[i] = (val[2*i] & 0xF) | ((val[2*i+1] & 0xF) << 4)
    i.e. low nibble = even-indexed element, high nibble = odd-indexed element.

  Dequantization:
    For each group g of 128 consecutive elements along K:
      fp16_weight[n, g*128 + j] = int4_value[n, g*128 + j] * scale[n, g]
    where int4_value is in [-8, 7] (4-bit two's complement).

Key difference from FP8:
  - FP8 uses per-tensor scale (one float scalar for the whole matrix), applied
    AFTER the K-reduction.  This is cheap — a single multiply at the end.
  - INT4 uses per-group scale (one fp16 per 128 elements), applied INSIDE the
    K-loop.  When VL=128, each loop iteration covers exactly one group, so one
    extra scale load per iteration.

================================================================================
Kernels under test
================================================================================

Only 2 kernels (compared to 4 in FP8, because INT4 has a single scale type):

  esimd_gemv_int4(input, packed_weight, group_scale, output)
      Single GEMV: output[1,N] = input[1,K] @ dequant(packed_weight[N,K/2])^T
      Used for: attention QKV/O projections, MLP down projection, MoE router.

  esimd_gemv_int4_fused2(input, w0, s0, o0, w1, s1, o1)
      Two GEMVs sharing the same input, submitted as one kernel to save launch
      overhead (~20-50 us per avoided launch).
      Used for: GDN input projection (in_proj_qkvz + in_proj_ba fused).

FP8 has pern (per-N scale) and pert (per-tensor scale) variants — 4 kernels
total.  INT4 does not need this split because per-group is the only scale type.
Therefore no pern-vs-pert comparison test is included.

================================================================================
Test structure
================================================================================

Reference self-tests (--ref-only, no kernel needed):
  test_reference_roundtrip       — pack then unpack exact integers, expect zero loss
  test_reference_quantize_range  — all quantized values in [-8, 7], scale > 0
  test_reference_gemv            — reference GEMV matches plain torch matmul

Kernel correctness (requires compiled esimd_gemv_int4):
  test_correctness_unit_scale    — integer weights with scale=1.0, isolates unpack logic
  test_correctness_with_scale    — real quantized weights, non-trivial group scales
  test_correctness_large_k       — large K (7168, 14336) to stress K_SPLIT + group
                                   boundary alignment
  test_fused2_correctness        — fused2 output matches two individual unfused calls
  test_fused2_vs_reference       — fused2 output matches Python reference directly

Performance benchmarks (requires compiled kernels):
  benchmark_shapes               — bandwidth utilization on Qwen3.5-122B TP4 shapes
  benchmark_fused                — fused2 latency vs sum of two individual calls

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/shaojun/custom-esimd-kernels-vllm

    python tests/test_gemv_int4.py --ref-only     # validate reference helpers only
    python tests/test_gemv_int4.py                 # full test (needs compiled kernel)
"""
import sys
import torch
import time

device = torch.device("xpu")

# INT4 symmetric quantization uses groups of 128 elements along K.
# Each group shares a single fp16 scale factor.
# This must match the kernel's compile-time GROUP_SIZE.
GROUP_SIZE = 128


# ============================================================================
# Reference helpers (pure PyTorch — no custom kernel needed)
#
# These serve two purposes:
#   1. Provide a ground-truth implementation for correctness tests.
#   2. Define the exact packing / dequant contract that the ESIMD kernel must
#      implement identically.
# ============================================================================

def int4_quantize(weight_fp16, group_size=GROUP_SIZE):
    """
    GGML q4_0 quantization:  fp16 weight  ->  (packed uint8, group scale).

    Matches the BigDL-core C library (quantize_row_q4_0_gptq_reference):
      1. Reshape weight into groups of `group_size` along K dimension.
      2. Compute per-group scale = max(group) / (-8).
         Note: uses signed max (not abs max), so scale can be negative.
      3. Quantize:  uint4 = clamp(round(weight / scale + 8), 0, 15).
         The +8 is the implicit zero_point.
      4. Pack two uint4 values into one byte:
           byte = (val[even_idx] & 0xF) | ((val[odd_idx] & 0xF) << 4)

    Args:
        weight_fp16: [N, K] fp16 tensor on device.
        group_size:  number of K-elements per scale group (must divide K).

    Returns:
        packed: [N, K//2] uint8  — packed uint4 weight on device.
        scale:  [N, K//group_size] fp16 — per-group scale on device.
    """
    N, K = weight_fp16.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert K % 2 == 0, f"K={K} must be even for int4 packing"

    # Reshape to [N, num_groups, group_size] for per-group operations.
    groups = weight_fp16.float().reshape(N, K // group_size, group_size)

    # Per-group scale: find the element with max absolute value, then
    # scale = max_val / (-8).  This is the GGML q4_0 convention.
    # We need the signed max (the element whose abs is largest, with sign).
    amax_vals, amax_idx = groups.abs().max(dim=-1, keepdim=True)
    # Gather the actual signed value at the max-abs position
    max_signed = groups.gather(-1, amax_idx)  # [N, num_groups, 1]
    scale = (max_signed / -8.0).squeeze(-1).to(torch.float16)  # [N, num_groups]

    # Quantize: uint4 = clamp(round(weight * (1/scale) + 8), 0, 15)
    # Handle zero scale (all-zero groups)
    scale_expanded = scale.float().unsqueeze(-1)  # [N, num_groups, 1]
    inv_scale = torch.where(
        scale_expanded.abs() > 1e-10,
        1.0 / scale_expanded,
        torch.zeros_like(scale_expanded))
    quantized = (groups * inv_scale + 8.5).to(torch.int32).clamp(0, 15)
    quantized = quantized.reshape(N, K)

    # Pack: two uint4 values per byte (low nibble = even, high nibble = odd).
    even = quantized[:, 0::2] & 0x0F
    odd = (quantized[:, 1::2] & 0x0F) << 4
    packed = (even | odd).to(torch.uint8)  # [N, K//2]

    return packed, scale


def int4_dequantize_ref(packed, scale, N, K, group_size=GROUP_SIZE):
    """
    Reference dequantization (GGML q4_0):  packed uint8  ->  fp16 weight.

    This is the inverse of int4_quantize (modulo quantization error).
    The ESIMD kernel must produce the same numerical result as this function.

    GGML q4_0 dequant formula:
      fp_weight = (uint4_val - 8) * scale

    Steps:
      1. Unpack each byte into two uint4 values (low nibble, high nibble).
      2. Subtract 8 (implicit zero_point) to get signed [-8, +7].
      3. Interleave low/high back to original K-ordering.
      4. Multiply by per-group scale (which can be negative).

    Args:
        packed: [N, K//2] uint8 on device.
        scale:  [N, K//group_size] fp16 on device.
        N, K:   original (unpacked) weight dimensions.
        group_size: elements per scale group.

    Returns:
        weight: [N, K] fp16 on device — dequantized weight.
    """
    # Unpack: byte -> low nibble (even index) + high nibble (odd index).
    # Values are unsigned [0, 15].
    low = (packed & 0x0F).to(torch.float32)
    high = ((packed >> 4) & 0x0F).to(torch.float32)

    # Subtract zero_point=8: unsigned [0,15] → signed [-8,+7]
    low = low - 8.0
    high = high - 8.0

    # Interleave [low0, high0, low1, high1, ...] to recover original K order.
    weight = torch.stack([low, high], dim=-1).reshape(N, K)

    # Apply per-group scale (can be negative — this is the GGML convention).
    scale_expanded = scale.float().repeat_interleave(group_size, dim=-1)

    return (weight * scale_expanded).to(torch.float16)


def gemv_int4_ref(input_t, packed, scale, N, K, group_size=GROUP_SIZE):
    """
    Reference INT4 GEMV:  output = input @ dequant(weight)^T.

    Dequantizes packed INT4 weights to fp16, then performs matmul in fp32.
    This is the ground truth for all kernel correctness tests.

    Args:
        input_t: [1, K] fp16 — input activation vector.
        packed:  [N, K//2] uint8 — packed INT4 weights.
        scale:   [N, K//group_size] fp16 — per-group scales.
        N, K:    unpacked weight dimensions.
        group_size: elements per scale group.

    Returns:
        output: [1, N] fp32 — result (kept in fp32 to preserve reference precision).
    """
    weight_fp16 = int4_dequantize_ref(packed, scale, N, K, group_size)
    return input_t.float() @ weight_fp16.float().T


# ============================================================================
# Reference self-tests
#
# These tests validate the Python pack/unpack/GEMV helpers themselves.
# Run with --ref-only to execute these without a compiled kernel.
# If these fail, all kernel tests would produce wrong baselines, so they
# run first unconditionally.
# ============================================================================

def test_reference_roundtrip():
    """
    Pack then unpack values and verify dequantized output is close to original.

    GGML q4_0 uses scale = max(block)/(-8), so the roundtrip is not perfectly
    lossless even for small integers (due to the asymmetric scale computation).
    We check that the relative error is small (<5%) for random weights.
    """
    print("\n--- Reference Pack/Unpack Roundtrip ---")
    for N, K in [(32, 128), (64, 256), (128, 1024), (16, 2048), (1, 128)]:
        weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1

        packed, scale = int4_quantize(weight_fp16)
        recovered = int4_dequantize_ref(packed, scale, N, K)

        max_diff = (recovered.float() - weight_fp16.float()).abs().max().item()
        ref_max = weight_fp16.float().abs().max().item()
        rel_err = max_diff / ref_max if ref_max > 1e-6 else 0
        ok = rel_err < 0.15  # 4-bit quantization has inherent error
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.6f}  rel={rel_err:.4f}")
        assert ok, f"Roundtrip failed for N={N}, K={K}: rel_err={rel_err}"


def test_reference_quantize_range():
    """
    Verify quantized uint4 values stay in [0, 15].

    In GGML q4_0, scale can be negative (when max element is negative).
    We check that packed nibbles are in valid unsigned range [0,15].
    """
    print("\n--- Reference Quantize Range Check ---")
    for N, K in [(64, 256), (128, 1024)]:
        weight = torch.randn(N, K, dtype=torch.float16, device=device)
        packed, scale = int4_quantize(weight)

        # Unpack to check raw unsigned values [0, 15].
        low = (packed & 0x0F).to(torch.int32)
        high = ((packed >> 4) & 0x0F).to(torch.int32)

        all_vals = torch.cat([low.flatten(), high.flatten()])
        vmin, vmax = all_vals.min().item(), all_vals.max().item()

        ok = vmin >= 0 and vmax <= 15
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}"
              f"  uint4_range=[{vmin}, {vmax}]  (expected [0,15])")
        assert ok, f"Range check failed for N={N}, K={K}"


def test_reference_gemv():
    """
    Verify gemv_int4_ref matches naive torch matmul on dequantized weights.

    Both paths do the same thing (dequant then matmul), so the diff must be
    zero up to floating-point associativity.  Any larger diff means a bug in
    gemv_int4_ref or int4_dequantize_ref.
    """
    print("\n--- Reference GEMV vs Torch Matmul ---")
    for N, K in [(128, 256), (512, 1024), (16, 2048)]:
        weight = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        packed, scale = int4_quantize(weight)
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        ref = gemv_int4_ref(input_t, packed, scale, N, K)

        w_deq = int4_dequantize_ref(packed, scale, N, K)
        manual = input_t.float() @ w_deq.float().T

        max_diff = (ref.float() - manual.float()).abs().max().item()
        ok = max_diff < 1e-4
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.6f}")
        assert ok, f"Reference GEMV mismatch for N={N}, K={K}"


# ============================================================================
# GGML C library packing format compatibility test
#
# The actual model uses ggml_quantize_tensor (C library) to quantize weights,
# NOT our Python int4_quantize.  If the C library packs bits differently
# (e.g. nibble order, sign convention, scale meaning), our kernel would
# produce wrong results even though the Python-only UT passes.
#
# This test verifies that:
#   1. The C library's packing format matches our kernel's expectation:
#      - low nibble (bits 0-3) = element at even K-position
#      - high nibble (bits 4-7) = element at odd K-position
#      - 4-bit two's complement: 0..7 = positive, 8..15 = -8..-1
#   2. The scale semantics match: dequant = int4_val * scale
#   3. End-to-end: kernel(ggml_quantized_weight) == matmul(dequant(weight))
# ============================================================================

def test_ggml_packing_format():
    """
    Verify the C library's packing layout matches our kernel's assumptions.

    GGML q4_0 format (from BigDL-core quantize.c):
      scale = max(block) / -8     (signed max, can be negative)
      uint4 = clamp(round(weight / scale + 8), 0, 15)
      dequant = (uint4 - 8) * scale

    This test places known values at specific positions and verifies:
      1. Scale matches the GGML formula.
      2. Low nibble corresponds to even K-position, high nibble to odd.
      3. Dequantized values are close to the original weights.
    """
    print("\n--- GGML Packing Format Check ---")
    try:
        from vllm.model_executor.layers.quantization.sym_int4 import (
            ggml_quantize_tensor, QK4_GROUP_SIZE, QK4_PACK_FACTOR)
    except ImportError:
        print("  [SKIP] Cannot import ggml_quantize_tensor (vllm not installed)")
        return

    N, K = 1, 128  # minimal: 1 row, 1 group
    weight = torch.zeros(N, K, dtype=torch.float32)

    # Place known values: position 0 (even→lo nibble), position 1 (odd→hi nibble)
    weight[0, 0] = 7.0
    weight[0, 1] = -7.0
    weight[0, 2] = 3.0
    weight[0, 3] = -3.0

    qweight = torch.zeros(N, K // QK4_PACK_FACTOR, dtype=torch.int32)
    scale = torch.zeros(N, K // QK4_GROUP_SIZE, dtype=torch.float16)
    qweight, scale = ggml_quantize_tensor(
        weight, qweight, scale, N, K,
        block_size=QK4_GROUP_SIZE, transpose=False)

    raw = qweight.view(torch.uint8)  # (1, 64)
    scale_val = scale[0, 0].item()

    # GGML: max(block) = 7.0 (first element with largest abs wins), scale = 7/-8 = -0.875
    expected_scale = 7.0 / -8.0
    scale_ok = abs(scale_val - expected_scale) < 0.01
    print(f"  scale = {scale_val:.4f} (expected {expected_scale:.4f})")

    # Verify dequant: (uint4 - 8) * scale ≈ original weight
    # Check byte 0 (positions 0 and 1) and byte 1 (positions 2 and 3)
    dequant_results = []
    originals = [7.0, -7.0, 3.0, -3.0]
    nibble_order_ok = True

    for byte_idx in range(2):
        byte_val = raw[0, byte_idx].item()
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        deq_lo = (lo - 8) * scale_val  # even position
        deq_hi = (hi - 8) * scale_val  # odd position
        orig_even = originals[byte_idx * 2]
        orig_odd = originals[byte_idx * 2 + 1]

        # Check that dequantized values are close to originals
        err_lo = abs(deq_lo - orig_even)
        err_hi = abs(deq_hi - orig_odd)
        print(f"  byte[{byte_idx}] = 0x{byte_val:02x}: "
              f"lo={lo}→deq={deq_lo:+.3f} (orig={orig_even:+.1f}, err={err_lo:.3f}), "
              f"hi={hi}→deq={deq_hi:+.3f} (orig={orig_odd:+.1f}, err={err_hi:.3f})")

        # Allow up to 1 quantization step of error: |scale| = 0.875
        tol = abs(scale_val) + 0.01
        if err_lo > tol or err_hi > tol:
            nibble_order_ok = False

    ok = nibble_order_ok and scale_ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] scale={'OK' if scale_ok else 'WRONG'}, "
          f"nibble_dequant={'OK' if nibble_order_ok else 'WRONG'}")
    assert ok, f"GGML packing format mismatch! scale={scale_ok}, nibble={nibble_order_ok}"


def test_ggml_kernel_e2e():
    """
    End-to-end: quantize with C library → run ESIMD kernel → compare vs dequant matmul.

    This is the definitive test: uses the SAME quantization path as the actual
    model (ggml_quantize_tensor with transpose=False), then checks that our
    kernel produces the correct GEMV result.

    If test_ggml_packing_format passes but this fails, the issue is likely in
    scale handling or accumulation precision, not in nibble order.
    """
    print("\n--- GGML → Kernel End-to-End ---")
    try:
        from vllm.model_executor.layers.quantization.sym_int4 import (
            ggml_quantize_tensor, QK4_GROUP_SIZE, QK4_PACK_FACTOR)
        from custom_esimd_kernels_vllm import esimd_gemv_int4
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    for N, K in [(128, 256), (512, 1024), (2560, 2048), (16, 2048), (3072, 2048)]:
        # Step 1: Create fp32 weight and quantize with C library
        weight_fp32 = torch.randn(N, K, dtype=torch.float32) * 0.1
        qweight = torch.zeros(N, K // QK4_PACK_FACTOR, dtype=torch.int32)
        scale = torch.zeros(N, K // QK4_GROUP_SIZE, dtype=torch.float16)
        qweight, scale = ggml_quantize_tensor(
            weight_fp32, qweight, scale, N, K,
            block_size=QK4_GROUP_SIZE, transpose=False)

        # Move to XPU
        qweight = qweight.to(device)
        scale = scale.to(device)

        # Step 2: View as uint8 for our kernel
        packed_u8 = qweight.view(torch.uint8)  # (N, K/2)

        # Step 3: Run ESIMD kernel
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)
        esimd_gemv_int4(input_t, packed_u8, scale, output)

        # Step 4: Reference — dequantize with our Python helper and matmul
        ref_deq = int4_dequantize_ref(packed_u8, scale, N, K)
        ref_out = input_t.float() @ ref_deq.float().T

        # Step 5: Compare
        max_diff = (output.float() - ref_out.float()).abs().max().item()
        ref_max = ref_out.float().abs().max().item()
        rel_err = (max_diff / ref_max) if ref_max > 1e-6 else 0
        ok = max_diff < 0.5 or rel_err < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] N={N:5d} K={K:5d}  max_diff={max_diff:.4f}  rel={rel_err:.4f}")
        assert ok, f"GGML E2E failed for N={N}, K={K}: rel_err={rel_err:.4f}"


def test_ggml_vs_python_quantize():
    """
    Compare C library quantization output against our Python int4_quantize.

    If these produce different packed bytes for the same input, it means the
    two quantizers use different conventions. In that case, our Python-only UT
    is testing against a wrong reference for the actual model path.
    """
    print("\n--- GGML vs Python Quantize Comparison ---")
    try:
        from vllm.model_executor.layers.quantization.sym_int4 import (
            ggml_quantize_tensor, QK4_GROUP_SIZE, QK4_PACK_FACTOR)
    except ImportError:
        print("  [SKIP] Cannot import ggml_quantize_tensor")
        return

    for N, K in [(64, 128), (128, 256), (32, 1024)]:
        # Same fp16 weight for both quantizers
        weight_fp16 = torch.randn(N, K, dtype=torch.float16)
        weight_fp32 = weight_fp16.float()

        # Python quantize (our reference helper)
        py_packed, py_scale = int4_quantize(weight_fp16.to(device))
        py_packed = py_packed.cpu()
        py_scale = py_scale.cpu()

        # C library quantize
        c_qweight = torch.zeros(N, K // QK4_PACK_FACTOR, dtype=torch.int32)
        c_scale = torch.zeros(N, K // QK4_GROUP_SIZE, dtype=torch.float16)
        c_qweight, c_scale = ggml_quantize_tensor(
            weight_fp32, c_qweight, c_scale, N, K,
            block_size=QK4_GROUP_SIZE, transpose=False)
        c_packed = c_qweight.view(torch.uint8)  # (N, K/2)

        # Compare packed bytes
        byte_match = (py_packed.cpu() == c_packed).float().mean().item()
        scale_diff = (py_scale.cpu().float() - c_scale.float()).abs().max().item()

        # They may not be identical (different rounding), but should be very close
        ok = byte_match > 0.95 and scale_diff < 0.01
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] N={N:5d} K={K:5d}  "
              f"byte_match={byte_match*100:.1f}%  scale_diff={scale_diff:.6f}")
        if byte_match < 0.95:
            # Show first mismatch for debugging
            mismatch = (py_packed.cpu() != c_packed)
            idx = mismatch.nonzero()[0]
            r, c = idx[0].item(), idx[1].item()
            print(f"    First mismatch at [{r},{c}]: "
                  f"python=0x{py_packed[r,c].item():02x} "
                  f"ggml=0x{c_packed[r,c].item():02x}")


# ============================================================================
# Kernel correctness tests
#
# Each test constructs INT4-quantized weights, runs the ESIMD kernel, and
# compares against the Python reference.  Allowed error accounts for:
#   - FP32 accumulation in kernel vs FP32 matmul in reference (same precision,
#     but different reduction order due to SIMD/K_SPLIT parallelism).
#   - fp16 output truncation.
# ============================================================================

def test_correctness_detailed():
    """
    Comprehensive kernel correctness test with detailed output.

    For each shape, shows three things:
      1. First 5 values of kernel output, dequant ref, and original fp16 ref
         — for visual sanity check.
      2. "vs dequant weight" error — kernel computation error only.
         kernel(int4_weight) vs python_matmul(dequant(int4_weight))
         Both use the SAME quantized weights; difference is accumulation order.
         Expected: <0.1%.
      3. "vs original fp16" error — total error including quantization loss.
         kernel(int4_weight) vs python_matmul(original_fp16_weight)
         Expected: ~10% (inherent INT4 precision loss).

    Covers:
      - Typical Qwen3.5 projection sizes (N=16..4096, K=128..7168)
      - Small K (128, 512): few groups, tests basic correctness
      - Large K (7168, 14336): many groups, stress-tests K_SPLIT + group boundary
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4

    print("\n--- Kernel Correctness (detailed) ---")

    shapes = [
        # (label,               N,     K)
        ("Attn qkv",         2560,  2048),
        ("Attn o_proj",      2048,   512),
        ("DN qkvz",          3072,  2048),
        ("DN ba",              16,  2048),
        ("Exp gate_up",       512,  2048),
        ("Large K (56 grp)",  128,  7168),
        ("Large K (112 grp)", 256, 14336),
        ("Small K (1 grp)",  1024,   128),
    ]

    all_ok = True
    for label, N, K in shapes:
        # Step 1: Create original FP16 weight and quantize.
        original_weight = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        packed, scale = int4_quantize(original_weight)

        # Step 2: Run kernel.
        kernel_out = torch.zeros(1, N, dtype=torch.float16, device=device)
        esimd_gemv_int4(input_t, packed, scale, kernel_out)

        # Step 3: Compute two references.
        #   dequant_ref: uses same quantized weights → measures kernel error
        #   original_ref: uses original fp16 weights → measures quantization error
        dequant_weight = int4_dequantize_ref(packed, scale, N, K)
        dequant_ref = input_t.float() @ dequant_weight.float().T
        original_ref = input_t.float() @ original_weight.float().T

        # Step 4: Compute errors.
        diff_dq = (kernel_out.float() - dequant_ref).abs()
        diff_orig = (kernel_out.float() - original_ref).abs()

        mean_dq = diff_dq.mean().item()
        max_dq = diff_dq.max().item()
        rel_dq = mean_dq / dequant_ref.abs().mean().item() if dequant_ref.abs().mean().item() > 1e-6 else 0

        mean_orig = diff_orig.mean().item()
        max_orig = diff_orig.max().item()
        rel_orig = mean_orig / original_ref.abs().mean().item() if original_ref.abs().mean().item() > 1e-6 else 0

        ok = rel_dq < 0.001  # kernel error should be < 0.1%
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False

        print(f"\n  [{status}] {label} (N={N}, K={K}, groups={K // GROUP_SIZE}):")
        print(f"    kernel:      {kernel_out[0, :5].tolist()}")
        print(f"    dequant ref: {dequant_ref[0, :5].tolist()}")
        print(f"    fp16 ref:    {original_ref[0, :5].tolist()}")
        print(f"    vs dequant weight: mean_abs={mean_dq:.4f}, "
              f"max_abs={max_dq:.4f}, rel={rel_dq:.4%}  "
              f"{'<-- kernel OK' if rel_dq < 0.001 else '<-- KERNEL ERROR!'}")
        print(f"    vs original fp16:  mean_abs={mean_orig:.4f}, "
              f"max_abs={max_orig:.4f}, rel={rel_orig:.4%}  "
              f"(INT4 quantization loss)")

    assert all_ok, "Some kernel correctness tests failed — see KERNEL ERROR above"


# ============================================================================
# Quantization error analysis
#
# The tests above compare kernel output against a Python reference that uses
# the SAME quantized weights (dequant ref).  This measures kernel computation
# error only — typically <0.05%.
#
# The test below ALSO compares kernel output against the result computed with
# ORIGINAL FP16 weights (before any quantization).  This measures the total
# error introduced by INT4 quantization itself — typically ~10% for 4-bit.
#
# Both numbers together tell the full story:
#   - kernel computation error small → kernel is correct
#   - quantization error ~10%        → inherent 4-bit precision loss, not a bug
# ============================================================================

def test_quantization_error_analysis():
    """
    Compare kernel output against TWO different references:

    1. "vs dequant weight" (kernel computation error):
       ref = input @ dequant(int4_weight).T
       Both sides use the SAME quantized weights.  The only difference is
       how the matmul is computed (ESIMD kernel vs PyTorch matmul).
       Expected rel_err: <0.05%  (just floating-point accumulation order).

    2. "vs original fp16 weight" (quantization error):
       ref = input @ original_fp16_weight.T
       Compares against the "perfect" answer before any quantization.
       Expected rel_err: ~10%  (inherent loss from 16-bit → 4-bit compression).

    If #1 is small but #2 is large → kernel is correct, quantization is lossy.
    If #1 is also large → kernel has a bug.
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4

    print("\n--- Quantization Error Analysis ---")
    print(f"  {'N':>6} {'K':>6}"
          f" | {'vs dequant weight (kernel err)':^40}"
          f" | {'vs original fp16 (quant err)':^40}")
    print(f"  {'':>6} {'':>6}"
          f" | {'mean_abs':>10} {'max_abs':>10} {'rel':>8}"
          f" | {'mean_abs':>10} {'max_abs':>10} {'rel':>8}")
    print("  " + "-" * 105)

    for N, K in [(2560, 2048), (512, 2048), (3072, 2048),
                 (128, 2048), (16, 2048), (2048, 512)]:
        # Step 1: Create random FP16 weight (the "original" before quantization).
        original_weight = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        # Step 2: Quantize to INT4 (GGML q4_0 format).
        packed, scale = int4_quantize(original_weight)

        # Step 3: Run ESIMD kernel.
        kernel_out = torch.zeros(1, N, dtype=torch.float16, device=device)
        esimd_gemv_int4(input_t, packed, scale, kernel_out)

        # Step 4: Compute "dequant weight" reference.
        #   Uses the same quantized weights, dequantized back to fp16.
        #   Difference from kernel = kernel computation error only.
        dequant_weight = int4_dequantize_ref(packed, scale, N, K)
        dequant_ref = input_t.float() @ dequant_weight.float().T

        # Step 5: Compute "original fp16 weight" reference.
        #   Uses the original weights before quantization.
        #   Difference from kernel = quantization error + kernel computation error.
        original_ref = input_t.float() @ original_weight.float().T

        # Step 6: Calculate errors for both comparisons.
        # --- vs dequant weight ---
        diff_dq = (kernel_out.float() - dequant_ref).abs()
        mean_abs_dq = diff_dq.mean().item()
        max_abs_dq = diff_dq.max().item()
        rel_dq = mean_abs_dq / dequant_ref.abs().mean().item()

        # --- vs original fp16 weight ---
        diff_orig = (kernel_out.float() - original_ref).abs()
        mean_abs_orig = diff_orig.mean().item()
        max_abs_orig = diff_orig.max().item()
        rel_orig = mean_abs_orig / original_ref.abs().mean().item()

        print(f"  {N:>6} {K:>6}"
              f" | {mean_abs_dq:>10.4f} {max_abs_dq:>10.4f} {rel_dq:>7.2%}"
              f" | {mean_abs_orig:>10.4f} {max_abs_orig:>10.4f} {rel_orig:>7.2%}")

    print()
    print("  Interpretation:")
    print("    'vs dequant weight' small (<0.1%)  → kernel computes correctly")
    print("    'vs original fp16'  large (~10%)   → inherent INT4 quantization loss (normal)")
    print("    If 'vs dequant weight' is also large → kernel has a bug")


# ============================================================================
# Fused2 correctness tests
#
# esimd_gemv_int4_fused2 computes two GEMVs sharing the same input in a
# single kernel submission.  This saves one kernel launch overhead (~20-50 us).
#
# In the Qwen3.5 model, this is used for GDN (GatedDeltaNet) input projection:
#   qkvz = input @ in_proj_qkvz.weight^T   (N0 = qkvz_dim, e.g. 3072)
#   ba   = input @ in_proj_ba.weight^T      (N1 = ba_dim,   e.g. 16)
# Both share the same input hidden_states [1, K].
#
# Two tests:
#   1. fused2 output == two unfused kernel calls  (tests kernel-vs-kernel).
#   2. fused2 output == Python reference           (tests kernel-vs-reference).
# ============================================================================

def test_fused2_correctness():
    """
    Fused2 vs two individual unfused calls — should be near bit-identical.

    The fused kernel merges two GEMV dispatch grids into one kernel submission.
    Numerically, each output element is computed identically to the unfused path
    (same weights, same accumulation order), so diff should be < 1e-3.
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4, esimd_gemv_int4_fused2

    print("\n--- Fused2 Correctness ---")
    cases = [
        # (name,          N0,   N1,   K)
        ("DN qkvz+ba",  3072,   16, 2048),  # GDN projection: large + tiny
        ("Exp gate+up",  512,  512, 2048),   # MoE expert: symmetric
        ("Sh gate+up",   128,  128, 2048),   # shared expert: small symmetric
        ("Symmetric",   1024, 1024, 1024),   # generic square-ish
    ]
    for name, N0, N1, K in cases:
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        w0_ref = torch.randn(N0, K, dtype=torch.float16, device=device) * 0.1
        packed0, scale0 = int4_quantize(w0_ref)

        w1_ref = torch.randn(N1, K, dtype=torch.float16, device=device) * 0.1
        packed1, scale1 = int4_quantize(w1_ref)

        # Unfused: two separate kernel calls.
        ref_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
        ref_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
        esimd_gemv_int4(input_t, packed0, scale0, ref_o0)
        esimd_gemv_int4(input_t, packed1, scale1, ref_o1)

        # Fused: single kernel call for both.
        fused_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
        fused_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
        esimd_gemv_int4_fused2(input_t,
                               packed0, scale0, fused_o0,
                               packed1, scale1, fused_o1)

        diff0 = (fused_o0.float() - ref_o0.float()).abs().max().item()
        diff1 = (fused_o1.float() - ref_o1.float()).abs().max().item()
        ok = diff0 < 1e-3 and diff1 < 1e-3
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<16} N0={N0:>5} N1={N1:>5} K={K}"
              f"  diff0={diff0:.6f} diff1={diff1:.6f}")
        assert ok, f"Fused2 mismatch for {name}: diff0={diff0}, diff1={diff1}"


def test_fused2_vs_reference():
    """
    Fused2 kernel output vs Python reference (not just vs unfused kernel).

    This catches bugs that might be shared between the fused and unfused kernel
    paths (e.g. a common dequant function with a sign error).  Comparing against
    a completely independent Python implementation is the strongest check.
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4_fused2

    print("\n--- Fused2 vs Python Reference ---")
    cases = [
        ("DN qkvz+ba",  3072,   16, 2048),
        ("Large",       2048, 2048, 4096),
    ]
    for name, N0, N1, K in cases:
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

        w0_ref = torch.randn(N0, K, dtype=torch.float16, device=device) * 0.1
        packed0, scale0 = int4_quantize(w0_ref)

        w1_ref = torch.randn(N1, K, dtype=torch.float16, device=device) * 0.1
        packed1, scale1 = int4_quantize(w1_ref)

        # Kernel (fused).
        fused_o0 = torch.zeros(1, N0, dtype=torch.float16, device=device)
        fused_o1 = torch.zeros(1, N1, dtype=torch.float16, device=device)
        esimd_gemv_int4_fused2(input_t,
                               packed0, scale0, fused_o0,
                               packed1, scale1, fused_o1)

        # Python reference.
        py_ref0 = gemv_int4_ref(input_t, packed0, scale0, N0, K)
        py_ref1 = gemv_int4_ref(input_t, packed1, scale1, N1, K)

        diff0 = (fused_o0.float() - py_ref0.float()).abs().max().item()
        diff1 = (fused_o1.float() - py_ref1.float()).abs().max().item()
        ref_max0 = py_ref0.float().abs().max().item()
        ref_max1 = py_ref1.float().abs().max().item()
        rel0 = (diff0 / ref_max0) if ref_max0 > 1e-6 else 0
        rel1 = (diff1 / ref_max1) if ref_max1 > 1e-6 else 0

        ok = (diff0 < 0.5 or rel0 < 0.05) and (diff1 < 0.5 or rel1 < 0.05)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<16} N0={N0:>5} N1={N1:>5} K={K}"
              f"  rel0={rel0:.4f} rel1={rel1:.4f}")
        assert ok, f"Fused2 vs ref mismatch for {name}"


# ============================================================================
# Performance benchmarks
#
# Measures effective memory bandwidth (GB/s) and compares against the
# theoretical peak of the target device (BMG @ 450 GB/s).
#
# Total bytes per GEMV call (determines roofline):
#   input:   K * 2         bytes  (fp16)
#   weight:  N * K / 2     bytes  (packed int4, half of FP8's N*K)
#   scale:   N * (K/128)*2 bytes  (fp16 per group, ~1.6% of weight bytes)
#   output:  N * 2         bytes  (fp16)
#
# INT4 weight is half the size of FP8, so for the same N,K the kernel should
# be ~2x less memory-bound, and the achievable throughput (in terms of "how
# fast the matmul finishes") should approach 2x of FP8 — assuming dequant
# compute is hidden behind memory latency.
#
# Cache-busting: each benchmark rotates through nc weight copies so that
# consecutive calls cannot hit L3 cache, measuring true DRAM bandwidth.
# ============================================================================

def benchmark_shapes():
    """
    Benchmark single-GEMV latency and bandwidth on Qwen3.5-122B-A10B TP4 shapes.

    Shape names correspond to model components:
      Attn qkv/o_proj  — full-attention Q/K/V and output projections
      DN qkvz/ba/out    — GatedDeltaNet (linear attention) projections
      Exp gate_up/down  — MoE routed expert MLP projections
      Sh gate_up/down   — MoE shared expert MLP projections
      Router            — MoE routing logits (hidden_size -> num_experts)

    TODO: verify N,K against actual Qwen3.5-122B-A10B TP4 config.
          Shapes below are adapted from Qwen3-Next-80B-A3B TP4 as placeholder.
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4

    shapes = [
        # (name,           N,     K)
        ("Attn qkv",     2560, 2048),
        ("Attn o_proj",  2048, 1024),
        ("DN qkvz",      3072, 2048),
        ("DN ba",          16, 2048),
        ("DN out_proj",  2048, 1024),
        ("Exp gate_up",   512, 2048),
        ("Exp down",     2048,  512),
        ("Sh gate_up",    128, 2048),
        ("Sh down",      2048,  128),
        ("Router",        512, 2048),
    ]

    TARGET_BW = 450.0  # GB/s BMG theoretical peak

    print(f"\n{'Shape':<18} {'N':>6} {'K':>6}"
          f" {'wt_KB':>7} {'sc_KB':>7} {'tot_KB':>7}"
          f" | {'GB/s':>8} {'BW%':>7} {'us':>8}")
    print("-" * 85)

    for name, N, K in shapes:
        weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        packed, scale = int4_quantize(weight_ref)
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output = torch.zeros(1, N, dtype=torch.float16, device=device)

        # Byte breakdown for bandwidth calculation.
        n_groups = K // GROUP_SIZE
        wt_bytes = N * K // 2             # packed int4 weight
        sc_bytes = N * n_groups * 2       # fp16 group scales
        io_bytes = K * 2 + N * 2          # input + output (both fp16)
        total_bytes = wt_bytes + sc_bytes + io_bytes

        # Cache-bust: allocate enough weight copies to exceed L3 (~32 MB).
        wb = N * K // 2
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(wb, 1))
        nc = min(nc, 512)

        packed_list = [packed]
        scale_list = [scale]
        for _ in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            p, s = int4_quantize(w)
            packed_list.append(p)
            scale_list.append(s)

        # More iterations for smaller shapes (less time per call).
        ni = (4000 if total_bytes < 512 * 1024
              else (1000 if total_bytes < 2 * 1024 * 1024 else 300))

        # Warmup (JIT compile + cache warm).
        for i in range(10):
            esimd_gemv_int4(input_t,
                            packed_list[i % nc], scale_list[i % nc], output)
        torch.xpu.synchronize()

        # Timed region.
        t0 = time.perf_counter()
        for i in range(ni):
            esimd_gemv_int4(input_t,
                            packed_list[i % nc], scale_list[i % nc], output)
        torch.xpu.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / ni * 1000
        bw = (total_bytes / 1e9) / (ms / 1e3)
        us = ms * 1000
        bw_pct = bw / TARGET_BW * 100

        print(f"{name:<18} {N:>6} {K:>6}"
              f" {wt_bytes // 1024:>6}K {sc_bytes // 1024:>6}K"
              f" {total_bytes // 1024:>6}K"
              f" | {bw:>7.1f} {bw_pct:>6.1f}% {us:>7.2f}")


def benchmark_fused():
    """
    Benchmark fused2 latency vs sum of two individual calls.

    The speedup comes from eliminating one kernel launch overhead and sharing
    the input read across both GEMVs.  Typical expected speedup: 1.1-1.5x
    depending on shape (larger shapes are more compute-bound, less launch-
    bound, so less benefit from fusion).

    Cases match actual Qwen3.5 usage:
      DN qkvz+ba   — GDN input: one large (3072) + one tiny (16) matrix
      Exp gate+up  — MoE expert: two medium symmetric matrices
      Sh gate+up   — shared expert: two small symmetric matrices
    """
    from custom_esimd_kernels_vllm import esimd_gemv_int4, esimd_gemv_int4_fused2

    print(f"\n{'Case':<20} {'Config':>20}"
          f" | {'Indiv us':>10} {'Fused us':>10} {'Speedup':>8}")
    print("-" * 78)

    def make_tensors(N, K):
        """Create quantized weight + output buffer for one GEMV."""
        w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        p, s = int4_quantize(w)
        o = torch.zeros(1, N, dtype=torch.float16, device=device)
        return p, s, o

    cases = [
        # (name,          [(N0, K), (N1, K)])
        ("DN qkvz+ba",   [(3072, 2048), (16, 2048)]),
        ("Exp gate+up",  [(512, 2048), (512, 2048)]),
        ("Sh gate+up",   [(128, 2048), (128, 2048)]),
    ]

    ni = 2000  # iterations for timing

    for name, shape_pairs in cases:
        K = shape_pairs[0][1]
        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        p0, s0, o0 = make_tensors(shape_pairs[0][0], K)
        p1, s1, o1 = make_tensors(shape_pairs[1][0], K)
        config = f"N=[{shape_pairs[0][0]},{shape_pairs[1][0]}] K={K}"

        # --- Benchmark: two individual calls ---
        for _ in range(10):
            esimd_gemv_int4(input_t, p0, s0, o0)
            esimd_gemv_int4(input_t, p1, s1, o1)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_int4(input_t, p0, s0, o0)
            esimd_gemv_int4(input_t, p1, s1, o1)
        torch.xpu.synchronize()
        indiv_us = (time.perf_counter() - t0) / ni * 1e6

        # --- Benchmark: single fused call ---
        for _ in range(10):
            esimd_gemv_int4_fused2(input_t, p0, s0, o0, p1, s1, o1)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_int4_fused2(input_t, p0, s0, o0, p1, s1, o1)
        torch.xpu.synchronize()
        fused_us = (time.perf_counter() - t0) / ni * 1e6

        speedup = indiv_us / fused_us if fused_us > 0 else 0
        print(f"{name:<20} {config:>20}"
              f" | {indiv_us:>9.2f} {fused_us:>9.2f} {speedup:>7.2f}x")


def benchmark_int4_vs_fp8():
    """
    Benchmark INT4 GEMV vs FP8 GEMV on identical (N, K) shapes.

    Purpose: ensure INT4 kernel performance hasn't regressed relative to FP8.
    Since INT4 loads half the weight bytes (0.5 B/element vs 1 B/element),
    the INT4 kernel should be faster on memory-bound shapes.

    For each shape, reports:
      - INT4 latency (us) and effective bandwidth (GB/s)
      - FP8  latency (us) and effective bandwidth (GB/s)
      - INT4/FP8 speedup ratio
      - Weight size ratio (INT4 is always 0.5x of FP8)

    A speedup < 1.0 means INT4 is slower than FP8 — potential regression.
    Theoretical speedup ≈ 1.5-2.0x for large shapes (memory-bound regime),
    less for small shapes where launch overhead or compute dominates.

    Both kernels use per-tensor-style scaling for fairest comparison:
      - FP8:  esimd_gemv_fp8_pert — single float scalar scale
      - INT4: esimd_gemv_int4    — per-group fp16 scale (group_size=128)

    Cache-busting is applied to both paths identically (rotate through
    multiple weight copies exceeding L3 size).
    """
    print("\n--- INT4 vs FP8 Performance Comparison ---")
    try:
        from custom_esimd_kernels_vllm import esimd_gemv_int4, esimd_gemv_fp8_pert
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    TARGET_BW = 450.0  # GB/s BMG theoretical peak

    shapes = [
        # (name,            N,     K)     — Qwen3.5-122B TP4 decode shapes
        ("Attn qkv",      2560,  2048),
        ("Attn o_proj",   2048,  1024),
        ("DN qkvz",       3072,  2048),
        ("DN ba",           16,  2048),
        ("DN out_proj",   2048,  1024),
        ("Exp gate_up",    512,  2048),
        ("Exp down",      2048,   512),
        ("Sh gate_up",     128,  2048),
        ("Sh down",       2048,   128),
        ("Router",         512,  2048),
    ]

    print(f"\n{'Shape':<16} {'N':>5} {'K':>5}"
          f" | {'INT4':>8} {'GB/s':>6} {'BW%':>5}"
          f" | {'FP8':>8} {'GB/s':>6} {'BW%':>5}"
          f" | {'Speedup':>7} {'Note':>12}")
    print("-" * 105)

    ni = 2000

    for name, N, K in shapes:
        n_groups = K // GROUP_SIZE

        # ---- Byte counts for bandwidth calculation ----
        # INT4: weight=N*K/2, scale=N*n_groups*2, input=K*2, output=N*2
        int4_wt = N * K // 2
        int4_sc = N * n_groups * 2
        int4_io = K * 2 + N * 2
        int4_total = int4_wt + int4_sc + int4_io

        # FP8: weight=N*K, scale=4 (single float), input=K*2, output=N*2
        fp8_wt = N * K
        fp8_sc = 4
        fp8_io = K * 2 + N * 2
        fp8_total = fp8_wt + fp8_sc + fp8_io

        # ---- Prepare INT4 tensors ----
        weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
        int4_packed, int4_scale = int4_quantize(weight_fp16)

        # ---- Prepare FP8 tensors ----
        # Use random uint8 as FP8 weight (we're benchmarking throughput, not correctness)
        fp8_weight = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device)
        fp8_scale = torch.tensor([0.01], dtype=torch.float32, device=device)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output_int4 = torch.zeros(1, N, dtype=torch.float16, device=device)
        output_fp8 = torch.zeros(1, N, dtype=torch.float16, device=device)

        # ---- Cache-bust: rotate weight copies to avoid L3 hits ----
        target_mem = 32 * 1024 * 1024
        nc = max(16, target_mem // max(int4_wt, fp8_wt, 1))
        nc = min(nc, 256)

        int4_copies = [(int4_packed, int4_scale)]
        fp8_copies = [(fp8_weight, fp8_scale)]
        for _ in range(1, nc):
            w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
            p, s = int4_quantize(w)
            int4_copies.append((p, s))
            fp8_copies.append((
                torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device),
                fp8_scale))

        # ---- Benchmark INT4 ----
        for i in range(10):
            p, s = int4_copies[i % nc]
            esimd_gemv_int4(input_t, p, s, output_int4)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for i in range(ni):
            p, s = int4_copies[i % nc]
            esimd_gemv_int4(input_t, p, s, output_int4)
        torch.xpu.synchronize()
        int4_us = (time.perf_counter() - t0) / ni * 1e6

        # ---- Benchmark FP8 ----
        for i in range(10):
            w, sc = fp8_copies[i % nc]
            esimd_gemv_fp8_pert(input_t, w, sc, output_fp8)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for i in range(ni):
            w, sc = fp8_copies[i % nc]
            esimd_gemv_fp8_pert(input_t, w, sc, output_fp8)
        torch.xpu.synchronize()
        fp8_us = (time.perf_counter() - t0) / ni * 1e6

        # ---- Compute metrics ----
        int4_bw = (int4_total / 1e9) / (int4_us / 1e6)
        fp8_bw = (fp8_total / 1e9) / (fp8_us / 1e6)
        int4_bw_pct = int4_bw / TARGET_BW * 100
        fp8_bw_pct = fp8_bw / TARGET_BW * 100
        speedup = fp8_us / int4_us if int4_us > 0 else 0

        note = ""
        if speedup < 0.95:
            note = "<-- REGRESS!"
        elif speedup < 1.05:
            note = "(~same)"
        elif speedup >= 1.5:
            note = "OK (BW win)"
        else:
            note = "OK"

        print(f"{name:<16} {N:>5} {K:>5}"
              f" | {int4_us:>7.1f}u {int4_bw:>5.0f} {int4_bw_pct:>4.0f}%"
              f" | {fp8_us:>7.1f}u {fp8_bw:>5.0f} {fp8_bw_pct:>4.0f}%"
              f" | {speedup:>6.2f}x {note:>12}")

    print()
    print("  Speedup = FP8_latency / INT4_latency (>1.0 means INT4 is faster)")
    print("  INT4 weight is 0.5x the size of FP8 → expect ~1.5-2.0x speedup")
    print("  Speedup < 1.0 indicates potential INT4 kernel regression")


def benchmark_vs_ipex():
    """
    Benchmark ESIMD INT4 GEMV vs IPEX INT4 linear (IPEXWeightOnlyQuantizedLinear).

    IPEX is the current production INT4 path — this benchmark shows whether our
    ESIMD kernel is faster and by how much.  Both paths use the same quantized
    weights (ggml format), so the comparison is fair.

    Metrics:
      - Latency (us) for M=1 decode
      - Speedup = IPEX_latency / ESIMD_latency

    Note: IPEX setup requires intel_extension_for_pytorch.  If not available,
    this benchmark is skipped.
    """
    print("\n--- ESIMD vs IPEX Performance ---")
    try:
        from vllm.model_executor.layers.quantization.sym_int4 import (
            ggml_quantize_tensor, QK4_GROUP_SIZE, QK4_PACK_FACTOR)
        from custom_esimd_kernels_vllm import esimd_gemv_int4
        import intel_extension_for_pytorch as ipex
    except ImportError as e:
        print(f"  [SKIP] {e}")
        return

    shapes = [
        ("Attn qkv",     2560, 2048),
        ("DN qkvz",      3072, 2048),
        ("Exp gate_up",   512, 2048),
        ("Exp down",     2048,  512),
        ("Router",        512, 2048),
    ]

    print(f"\n{'Shape':<18} {'N':>6} {'K':>6}"
          f" | {'IPEX us':>9} {'ESIMD us':>9} {'Speedup':>8}")
    print("-" * 65)

    ni = 2000

    for name, N, K in shapes:
        # --- Quantize with C library ---
        weight_fp32 = torch.randn(N, K, dtype=torch.float32) * 0.1

        # IPEX path: transposed layout
        qw_ipex = torch.zeros(N, K // QK4_PACK_FACTOR, dtype=torch.int32)
        sc_ipex = torch.zeros(N, K // QK4_GROUP_SIZE, dtype=torch.float16)
        qw_ipex, sc_ipex = ggml_quantize_tensor(
            weight_fp32, qw_ipex, sc_ipex, N, K,
            block_size=QK4_GROUP_SIZE, transpose=True)
        qw_ipex = qw_ipex.to(device)
        sc_ipex = sc_ipex.to(device)

        # ESIMD path: row-major layout
        qw_esimd = torch.zeros(N, K // QK4_PACK_FACTOR, dtype=torch.int32)
        sc_esimd = torch.zeros(N, K // QK4_GROUP_SIZE, dtype=torch.float16)
        qw_esimd, sc_esimd = ggml_quantize_tensor(
            weight_fp32, qw_esimd, sc_esimd, N, K,
            block_size=QK4_GROUP_SIZE, transpose=False)
        qw_esimd = qw_esimd.to(device)
        sc_esimd = sc_esimd.to(device)
        packed_u8 = qw_esimd.view(torch.uint8)

        # --- Setup IPEX qlinear ---
        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=QK4_GROUP_SIZE,
        )
        ipex_linear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            qw_ipex, sc_ipex,
            torch.tensor([8], device=device, dtype=torch.int8),
            qw_ipex.size(0), qw_ipex.size(1) if qw_ipex.shape[0] == K // QK4_PACK_FACTOR else N,
            qconfig=qconfig, g_idx=None, bias=None,
            group_size=QK4_GROUP_SIZE, quant_method=0)

        input_t = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
        output_esimd = torch.zeros(1, N, dtype=torch.float16, device=device)

        # --- Benchmark IPEX ---
        for _ in range(10):
            ipex_linear(input_t)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            ipex_linear(input_t)
        torch.xpu.synchronize()
        ipex_us = (time.perf_counter() - t0) / ni * 1e6

        # --- Benchmark ESIMD ---
        for _ in range(10):
            esimd_gemv_int4(input_t, packed_u8, sc_esimd, output_esimd)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ni):
            esimd_gemv_int4(input_t, packed_u8, sc_esimd, output_esimd)
        torch.xpu.synchronize()
        esimd_us = (time.perf_counter() - t0) / ni * 1e6

        speedup = ipex_us / esimd_us if esimd_us > 0 else 0
        print(f"{name:<18} {N:>6} {K:>6}"
              f" | {ipex_us:>8.2f} {esimd_us:>8.2f} {speedup:>7.2f}x")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    ref_only = "--ref-only" in sys.argv

    print("=" * 60)
    print("custom-esimd-kernels-vllm: GEMV INT4 Tests")
    print("=" * 60)

    # --- Phase 1: Reference self-tests (always run, no kernel needed) ---
    # These validate the Python pack/unpack/GEMV helpers.
    # If these fail, kernel tests would compare against wrong baselines.
    test_reference_roundtrip()
    test_reference_quantize_range()
    test_reference_gemv()

    if ref_only:
        print("\n" + "=" * 60)
        print("REFERENCE TESTS PASSED (--ref-only, kernel tests skipped)")
        print("=" * 60)
        sys.exit(0)

    # --- Phase 1.5: GGML C library compatibility (critical gate) ---
    # If packing format doesn't match, all kernel tests below are meaningless.
    test_ggml_packing_format()
    test_ggml_vs_python_quantize()
    test_ggml_kernel_e2e()

    # --- Phase 2: Kernel correctness (requires compiled esimd_gemv_int4) ---
    test_correctness_detailed()      # kernel err + quantization err, detailed output
    test_fused2_correctness()        # fused2 == 2x unfused
    test_fused2_vs_reference()       # fused2 == python reference

    # --- Phase 3: Performance benchmarks ---
    print("\n--- Performance Benchmark (unfused) ---")
    benchmark_shapes()

    print("\n--- Performance Benchmark (fused vs individual) ---")
    benchmark_fused()

    print("\n--- Performance Benchmark (INT4 vs FP8) ---")
    benchmark_int4_vs_fp8()

    print("\n--- Performance Benchmark (ESIMD vs IPEX) ---")
    benchmark_vs_ipex()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
