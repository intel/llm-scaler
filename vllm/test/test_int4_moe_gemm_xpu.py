#!/usr/bin/env python3
"""
Minimal reproducer for Bug H: INT4 MoE GEMM DEVICE_LOST on Lunar Lake Xe2-LPG.

Tests whether torch.xpu.moe_gemm(..., is_int4=True) works at all on the device.
Expected result on Lunar Lake: DEVICE_LOST on the first call.
Expected result on discrete Arc (B580, etc.): should work.

Usage:
    source /opt/intel/oneapi/setvars.sh --force
    source ~/llm-scaler-vllm/venv/bin/activate  # or wherever vLLM venv is
    python test_int4_moe_gemm_xpu.py
"""
import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch: {torch.__version__}")
print(f"IPEX: {ipex.__version__}")
print(f"XPU device: {torch.xpu.get_device_name(0)}")
print(f"XPU memory: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
print()

# Small MoE dimensions — 4 experts, tiny hidden size
num_experts = 4
hidden_dim = 256
intermediate_dim = 512
group_size = 128  # typical GPTQ group size

# INT4 packing: 8 INT4 values per int32 word
# Weight shape for INT4: [num_experts, intermediate_dim // 8, hidden_dim]
# (packed along the K dimension)
K_packed = intermediate_dim // 8  # 64

print("=" * 60)
print("Test 1: MXFP4 MoE GEMM (expected to WORK)")
print("=" * 60)

try:
    # MXFP4 weights — same kernel path as GPT-OSS-20B
    W_mxfp4 = torch.randint(0, 255, (num_experts, intermediate_dim, hidden_dim),
                             dtype=torch.uint8, device="xpu")
    hidden = torch.randn(8, hidden_dim, dtype=torch.bfloat16, device="xpu")
    rows = torch.tensor([2, 2, 2, 2], dtype=torch.int64, device="xpu")

    result = torch.xpu.moe_gemm(
        hidden, W_mxfp4, rows, num_experts,
        is_mxfp4=True, is_fp8=False, is_int4=False,
        use_native=False,
    )
    torch.xpu.synchronize()
    print(f"  PASSED — output shape: {result.shape}")
    del W_mxfp4, hidden, rows, result
    torch.xpu.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("Test 2: INT4 MoE GEMM (expected to CRASH on Lunar Lake)")
print("=" * 60)

try:
    # INT4 packed weights — same kernel path as AutoRound INT4
    # Shape: [num_experts, K_packed, hidden_dim] where K_packed = K // 8
    W_int4 = torch.randint(0, 2**31, (num_experts, K_packed, hidden_dim),
                            dtype=torch.int32, device="xpu")
    hidden = torch.randn(8, hidden_dim, dtype=torch.bfloat16, device="xpu")
    rows = torch.tensor([2, 2, 2, 2], dtype=torch.int64, device="xpu")

    # Scale tensor for INT4 dequantization
    num_groups = intermediate_dim // group_size  # 4
    scale = torch.ones(num_experts, num_groups, hidden_dim,
                       dtype=torch.bfloat16, device="xpu")

    result = torch.xpu.moe_gemm(
        hidden, W_int4, rows, num_experts,
        weight_scale_inv=scale,
        is_mxfp4=False, is_fp8=False, is_int4=True,
        use_native=False,
    )
    torch.xpu.synchronize()
    print(f"  PASSED — output shape: {result.shape}")
    del W_int4, hidden, rows, scale, result
    torch.xpu.empty_cache()
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("Test 3: INT4 MoE GEMM with use_native=True (fallback path)")
print("=" * 60)

try:
    W_int4 = torch.randint(0, 2**31, (num_experts, K_packed, hidden_dim),
                            dtype=torch.int32, device="xpu")
    hidden = torch.randn(8, hidden_dim, dtype=torch.bfloat16, device="xpu")
    rows = torch.tensor([2, 2, 2, 2], dtype=torch.int64, device="xpu")
    scale = torch.ones(num_experts, intermediate_dim // group_size, hidden_dim,
                       dtype=torch.bfloat16, device="xpu")

    result = torch.xpu.moe_gemm(
        hidden, W_int4, rows, num_experts,
        weight_scale_inv=scale,
        is_mxfp4=False, is_fp8=False, is_int4=True,
        use_native=True,  # per-expert loop fallback
    )
    torch.xpu.synchronize()
    print(f"  PASSED — output shape: {result.shape}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("Done. If Test 2 failed with DEVICE_LOST, the INT4 xetla kernel")
print("(group_mm_int4_out_marlin) does not support Xe2-LPG architecture.")
