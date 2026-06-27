#!/usr/bin/env python3
"""
Bug H diagnostic: test INT4 MoE GEMM with different tile policies.

Tests whether the DEVICE_LOST after attention is caused by the INT4 kernel's
large 256×256 GEMM workgroup tiles vs MXFP4's 128×128 tiles.

The xetla kernel policy is selected based on average_m (total_m / num_experts):
  average_m ≤ 4   → GEMV    (wg 8×64)   — same for INT4 and MXFP4
  average_m ≤ 32  → GEMV_16 (wg 16×64)  — same for INT4 and MXFP4
  average_m ≤ 128 → GEMV_32 (wg 32×64)  — same for INT4 and MXFP4
  average_m > 128 → GEMM    (wg 256×256 INT4 vs 128×128 MXFP4)

If GEMV policy works after attention but GEMM policy crashes, it confirms
the tile size is the root cause of Bug H.

Usage (inside vLLM apply() after attention, or standalone):
    python test_int4_tile_policy.py
"""
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch: {torch.__version__}")
print(f"IPEX: {ipex.__version__}")
print(f"XPU device: {torch.xpu.get_device_name(0)}")
print()

# Model-like dimensions (Qwen3.5-35B-A3B scale)
num_experts = 64
hidden_dim = 2048
intermediate_dim = 1024  # smaller for test
group_size = 128
K_packed = intermediate_dim // 8

# Create weights once
W_int4 = torch.randint(0, 2**31, (num_experts, K_packed, hidden_dim),
                        dtype=torch.int32, device="xpu")
scale = torch.ones(num_experts, intermediate_dim // group_size, hidden_dim,
                   dtype=torch.float16, device="xpu")  # Note: INT4 uses FP16 scale


def test_policy(policy_name, total_tokens, top_k=4):
    """Test moe_gemm with a specific number of tokens to force a tile policy."""
    # Compute what C++ will see as average_m
    average_m = (total_tokens + num_experts - 1) // num_experts
    print(f"\n{'='*60}")
    print(f"Policy: {policy_name}")
    print(f"  total_tokens={total_tokens}, num_experts={num_experts}, "
          f"average_m≈{average_m}")

    hidden = torch.randn(total_tokens, hidden_dim, dtype=torch.bfloat16,
                          device="xpu")

    # Distribute tokens roughly evenly across top_k experts
    rows = torch.zeros(num_experts, dtype=torch.int64, device="xpu")
    tokens_per_expert = total_tokens // top_k
    for i in range(top_k):
        rows[i] = tokens_per_expert
    remainder = total_tokens - tokens_per_expert * top_k
    if remainder > 0:
        rows[0] += remainder

    # Pad rows to multiple of 8 (required by IPEX)
    aligned = ((num_experts + 7) // 8) * 8
    if rows.shape[0] < aligned:
        rows = torch.nn.functional.pad(rows, (0, aligned - rows.shape[0]))

    try:
        result = torch.xpu.moe_gemm(
            hidden, W_int4, rows, num_experts,
            None, scale,
            bias=None, is_mxfp4=False, is_fp8=False,
            is_int4=True, use_native=False,
        )
        torch.xpu.synchronize()
        print(f"  PASSED — output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# Test each tile policy threshold
print("="*60)
print("Testing INT4 MoE GEMM tile policies in ISOLATION")
print("(No attention ops have run — all should pass)")
print("="*60)

# GEMV: average_m ≤ 4 → total_tokens ≤ 4 * num_experts = 256
test_policy("GEMV (8×64)", total_tokens=4)

# GEMV_16: average_m ≤ 32
test_policy("GEMV_16 (16×64)", total_tokens=32 * num_experts)

# GEMV_32: average_m ≤ 128
test_policy("GEMV_32 (32×64)", total_tokens=128 * num_experts)

# GEMM: average_m > 128 — this is the 256×256 tile
test_policy("GEMM (256×256)", total_tokens=256 * num_experts)

print()
print("="*60)
print("DTYPE CHECK")
print("="*60)
print(f"  Input (hidden_states) dtype: {torch.bfloat16}")
print(f"  INT4 C++ expects: sycl::half (FP16)")
print(f"  MXFP4 C++ expects: sycl::ext::oneapi::bfloat16 (BF16)")
print(f"  Weight scale dtype: {scale.dtype} (FP16)")
print()
print("If INT4 kernel receives BF16 input but hardcodes sycl::half,")
print("there may be an implicit cast that interacts with post-attention state.")
print()
print("To test Bug H hypothesis: run this AFTER attention ops in vLLM pipeline.")
print("If GEMV passes but GEMM crashes → tile size is root cause.")
print("If ALL crash → it's the INT4 compute_policy or dispatch model, not tiles.")
