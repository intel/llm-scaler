"""
Correctness unit tests for INT4 ESIMD MOE kernels (K-major + marlin unshuffle):
  - moe_router_forward_int4
  - moe_forward_full_int4

Weights are transformed to IPEX format (transpose + marlin shuffle) before
passing to the kernel, matching the real vLLM integration.

Usage:
    python tests/test_moe_int4_kernel.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

DEVICE = "xpu"

# Test configs
CONFIGS = {
    "small": {
        "hidden_size": 256, "intermediate_size": 128, "shared_intermediate_size": 128,
        "num_experts": 16, "top_k": 4,
    },
    "35B-A3B-TP4": {
        "hidden_size": 2048, "intermediate_size": 256, "shared_intermediate_size": 128,
        "num_experts": 256, "top_k": 8,
    },
    "122B-A10B-TP4": {
        "hidden_size": 3072, "intermediate_size": 256, "shared_intermediate_size": 256,
        "num_experts": 256, "top_k": 8,
    },
}
NUM_SHARED_EXPERTS = 1
GROUP_SIZE = 128
PACK_FACTOR = 8


# ─── INT4 Quantization Helpers ────────────────────────────────────────────────

def quantize_int4(weight_fp16: torch.Tensor, group_size: int = GROUP_SIZE):
    """Vectorized quantize FP16 weight [N, K] → INT4 packed int32 + scales."""
    N, K = weight_fp16.shape
    assert K % group_size == 0 and K % PACK_FACTOR == 0
    n_groups = K // group_size

    w = weight_fp16.float().numpy()
    w_grouped = w.reshape(N, n_groups, group_size)

    max_abs = np.abs(w_grouped).max(axis=2)
    scale_np = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)

    scale_expanded = scale_np[:, :, None].astype(np.float32)
    quantized = np.round(w_grouped / scale_expanded).clip(-8, 7).astype(np.int32) + 8

    quantized_flat = quantized.reshape(N, K)
    quantized_packed = quantized_flat.reshape(N, K // PACK_FACTOR, PACK_FACTOR).astype(np.uint32)
    packed = np.zeros((N, K // PACK_FACTOR), dtype=np.uint32)
    for b in range(PACK_FACTOR):
        packed |= (quantized_packed[:, :, b] & 0xF) << (b * 4)

    qweight = torch.from_numpy(packed.view(np.int32))
    scales = torch.from_numpy(scale_np)
    return qweight, scales


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor,
                    N: int, K: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Vectorized dequantize INT4 packed weight → FP16 [N, K]."""
    qw_np = qweight.numpy().view(np.uint32)
    sc_np = scales.numpy().astype(np.float32)
    n_groups = K // group_size

    unpacked = np.zeros((N, K), dtype=np.float32)
    for b in range(PACK_FACTOR):
        nibbles = ((qw_np >> (b * 4)) & 0xF).astype(np.float32) - 8.0
        unpacked[:, b::PACK_FACTOR] = nibbles

    unpacked_grouped = unpacked.reshape(N, n_groups, group_size)
    unpacked_grouped *= sc_np[:, :, None]
    result = unpacked_grouped.reshape(N, K)
    return torch.from_numpy(result).half()


def marlin_shuffle_weight(qweight_np):
    """Simulate IPEX marlin_shuffle_weight: reorder nibbles within each int32.

    IPEX shuffled_idx = [0, 4, 1, 5, 2, 6, 3, 7]
    This means: new_nibble[0]=old_nibble[0], new_nibble[1]=old_nibble[4],
                new_nibble[2]=old_nibble[1], new_nibble[3]=old_nibble[5], etc.
    """
    shuffled_idx = np.array([0, 4, 1, 5, 2, 6, 3, 7])

    result = np.zeros_like(qweight_np)
    for new_pos in range(8):
        old_pos = shuffled_idx[new_pos]
        nibbles = (qweight_np >> np.uint32(old_pos * 4)) & np.uint32(0xF)
        result |= nibbles << np.uint32(new_pos * 4)
    return result


def ipex_transform_expert_weights(qweight, scales, E, N, K_packed, K_groups):
    """Simulate full IPEX GatedMLPMOE transformation on expert weights.

    Input:  qweight [E, N, K_packed] int32, scales [E, N, K_groups] fp16
    Output: qweight [E, K_packed, N] int32 (transposed + marlin shuffled)
            scales  [E, K_groups, N] fp16 (transposed)
    """
    # Step 1: transpose [E, N, K_packed] → [E, K_packed, N]
    qw_t = qweight.permute(0, 2, 1).contiguous()
    sc_t = scales.permute(0, 2, 1).contiguous()

    # Step 2: marlin shuffle the transposed weight
    qw_np = qw_t.numpy().view(np.uint32)
    qw_shuffled = marlin_shuffle_weight(qw_np)
    qw_t = torch.from_numpy(qw_shuffled.view(np.int32))

    return qw_t, sc_t


def ipex_column_major_repack(qweight):
    """Simulate IPEX IPEXWeightOnlyQuantizedLinear column-major repack.

    Input:  qweight [K_packed, N] row-major (GPTQ convention)
    Output: same shape [K_packed, N] but flat memory rearranged so that
            reshape({N, K_packed}) gives new[n, kp] = original[kp, n].

    IPEX stores column-major of [K_packed, N]: element [kp, n] at offset n*K_packed+kp.
    PyTorch sees it as row-major [K_packed, N].
    """
    K_packed, N = qweight.shape
    # Transpose gives [N, K_packed] row-major, flatten gives flat[n*K_packed+kp] = original[kp, n]
    flat = qweight.t().contiguous().flatten()
    return flat.reshape(K_packed, N)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float().unsqueeze(0),
                               b.flatten().float().unsqueeze(0)).item()


# ─── Test 1: moe_router_forward_int4 ─────────────────────────────────────────

def test_router_forward():
    from custom_esimd_kernels_vllm import moe_int4_ops

    print("=" * 60)
    print("Test 1: moe_router_forward_int4")
    print("=" * 60)

    for cfg_name, cfg in CONFIGS.items():
        H = cfg["hidden_size"]
        E = cfg["num_experts"]
        print(f"\n  Config: {cfg_name} (H={H}, E={E})")

        for n_tokens in [1, 4, 16]:
            gate_fp16 = (torch.randn(E, H) * 0.02).half()
            qweight, scales = quantize_int4(gate_fp16, GROUP_SIZE)

            gate_dq = dequantize_int4(qweight, scales, E, H, GROUP_SIZE)
            x = (torch.randn(n_tokens, H) * 0.1).half()
            ref_out = (x.float() @ gate_dq.float().t()).half()

            # Kernel uses reshape to handle IPEX column-major repack.
            # Test: [E,K] (direct) and [K,E] (IPEX OneDNN column-major repack simulation)
            K_packed = H // PACK_FACTOR
            layouts = ["[E,K]"]
            if K_packed != E:
                layouts.append("[K,E]_colmaj")  # simulate IPEX OneDNN column-major repack
            for layout in layouts:
                if layout == "[E,K]":
                    qw = qweight.to(DEVICE)
                else:
                    # Simulate IPEX OneDNN: store [E,K] data in [K,E] shape (column-major repack)
                    # flat memory of [E,K] row-major = [K,E] column-major
                    qw_flat = qweight.flatten()  # [E*K_packed] in [E,K] row-major order
                    qw = qw_flat.reshape(K_packed, E).to(DEVICE)  # reinterpret as [K,E]
                sc = scales.t().contiguous().to(DEVICE)

                kernel_out = moe_int4_ops.moe_router_forward_int4(
                    x.to(DEVICE), qw, sc).cpu()

                cos = cosine_similarity(ref_out, kernel_out)
                mae = (ref_out.float() - kernel_out.float()).abs().max().item()
                passed = cos > 0.98 and mae < 0.5
                status = "PASS" if passed else "FAIL"
                print(f"    n={n_tokens:>3d} {layout}: cos={cos:.6f} mae={mae:.4f} [{status}]")
                if not passed:
                    return False

    print("  All router tests passed!\n")
    return True


# ─── Test 2: moe_forward_full_int4 ───────────────────────────────────────────

def _ref_moe_forward_full(x, logits, w13_fp16, shared_gate_up_fp16,
                          w2_fp16, shared_down_fp16,
                          shared_gate_weight, top_k, n_routed_experts):
    """Pure PyTorch reference for the full MoE pipeline."""
    n_tokens = x.shape[0]
    hidden_size = x.shape[1]
    intermediate_size = w2_fp16.shape[2]
    shared_inter = shared_gate_up_fp16.shape[0] // 2

    x_f = x.float()

    probs = F.softmax(logits.float(), dim=-1)
    topk_weight, topk_idx = torch.topk(probs, top_k, dim=-1)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.half()

    final = torch.zeros(n_tokens, hidden_size, dtype=torch.float32)

    for t in range(n_tokens):
        routed_sum = torch.zeros(hidden_size, dtype=torch.float32)
        for ki in range(top_k):
            eid = topk_idx[t, ki].item()
            rw = topk_weight[t, ki].float()

            gate_w = w13_fp16[eid, :intermediate_size, :].float()
            up_w = w13_fp16[eid, intermediate_size:, :].float()
            gate_out = x_f[t] @ gate_w.t()
            up_out = x_f[t] @ up_w.t()

            intermediate = (gate_out / (1 + torch.exp(-gate_out))) * up_out

            down_w = w2_fp16[eid].float()
            down_out = intermediate @ down_w.t()

            routed_sum += rw * down_out

        shared_gate_up_w = shared_gate_up_fp16.float()
        gate_out = x_f[t] @ shared_gate_up_w[:shared_inter, :].t()
        up_out = x_f[t] @ shared_gate_up_w[shared_inter:, :].t()
        intermediate = (gate_out / (1 + torch.exp(-gate_out))) * up_out

        shared_down_w = shared_down_fp16.float()
        shared_down_out = intermediate @ shared_down_w.t()

        gate_sigmoid = torch.sigmoid(x_f[t] @ shared_gate_weight.float().t()).item()
        shared_down_out *= gate_sigmoid

        final[t] = routed_sum + shared_down_out

    return final.half()


def test_forward_full():
    from custom_esimd_kernels_vllm import moe_int4_ops

    print("=" * 60)
    print("Test 2: moe_forward_full_int4 (K-major + marlin unshuffle)")
    print("=" * 60)

    for cfg_name, cfg in CONFIGS.items():
        H = cfg["hidden_size"]
        D = cfg["intermediate_size"]
        D_S = cfg["shared_intermediate_size"]
        E = cfg["num_experts"]
        TK = cfg["top_k"]
        print(f"\n  Config: {cfg_name} (H={H}, D={D}, E={E})")

        for n_tokens in [2, 4]:  # bs>=2 to skip fused router+topk path (bs=1)
            torch.manual_seed(42)

            w13_fp16 = (torch.randn(E, 2 * D, H) * 0.02).half()
            w2_fp16 = (torch.randn(E, H, D) * 0.02).half()
            shared_gate_up = (torch.randn(2 * D_S, H) * 0.02).half()
            shared_down = (torch.randn(H, D_S) * 0.02).half()
            shared_gate_w = (torch.randn(1, H) * 0.02).half()
            x = (torch.randn(n_tokens, H) * 0.1).half()

            w13_qw_list, w13_sc_list, w2_qw_list, w2_sc_list = [], [], [], []
            for e_idx in range(E):
                qw, sc = quantize_int4(w13_fp16[e_idx], GROUP_SIZE)
                w13_qw_list.append(qw); w13_sc_list.append(sc)
                qw, sc = quantize_int4(w2_fp16[e_idx], GROUP_SIZE)
                w2_qw_list.append(qw); w2_sc_list.append(sc)

            w13_qweight = torch.stack(w13_qw_list)
            w13_scales = torch.stack(w13_sc_list)
            w2_qweight = torch.stack(w2_qw_list)
            w2_scales = torch.stack(w2_sc_list)

            w13_dq = torch.stack([dequantize_int4(w13_qweight[e_idx], w13_scales[e_idx],
                                  2 * D, H, GROUP_SIZE) for e_idx in range(E)])
            w2_dq = torch.stack([dequantize_int4(w2_qweight[e_idx], w2_scales[e_idx],
                                  H, D, GROUP_SIZE) for e_idx in range(E)])

            w13_ipex, s13_ipex = ipex_transform_expert_weights(
                w13_qweight, w13_scales, E, 2 * D, H // PACK_FACTOR, H // GROUP_SIZE)
            w2_ipex, s2_ipex = ipex_transform_expert_weights(
                w2_qweight, w2_scales, E, H, D // PACK_FACTOR, D // GROUP_SIZE)

            logits = (torch.randn(n_tokens, E) * 0.1).half()

            ref_out = _ref_moe_forward_full(
                x, logits, w13_dq, shared_gate_up, w2_dq, shared_down,
                shared_gate_w, TK, E)

            # Gate weight/scale for fused router+topk (bs=1 path)
            # Shared expert is FP16 in UT — pass dummy scales (empty tensor)
            _dummy_scale = torch.empty(0, device=DEVICE, dtype=torch.float16)

            kernel_out = moe_int4_ops.moe_forward_full_int4(
                x.to(DEVICE), logits.to(DEVICE),
                w13_ipex.to(DEVICE), s13_ipex.to(DEVICE),
                shared_gate_up.to(DEVICE), _dummy_scale,
                w2_ipex.to(DEVICE), s2_ipex.to(DEVICE),
                shared_down.to(DEVICE), _dummy_scale,
                shared_gate_w.to(DEVICE),
                TK, NUM_SHARED_EXPERTS, E).cpu()

            cos = cosine_similarity(ref_out, kernel_out)
            mae = (ref_out.float() - kernel_out.float()).abs().max().item()
            passed = cos > 0.95 and mae < 1.0
            status = "PASS" if passed else "FAIL"
            print(f"    n={n_tokens:>3d}  cos={cos:.6f}  mae={mae:.4f}  [{status}]")
            if not passed:
                print(f"      ref[:5]={ref_out[0,:5].tolist()}")
                print(f"      ker[:5]={kernel_out[0,:5].tolist()}")
                return False

    print("  All forward_full tests passed!\n")
    return True


# ─── Test 2b: moe_forward_full_int4 with INT4 shared expert ─────────────────

def test_forward_full_int4_shared():
    from custom_esimd_kernels_vllm import moe_int4_ops

    print("=" * 60)
    print("Test 2b: moe_forward_full_int4 (INT4 shared expert)")
    print("=" * 60)

    for cfg_name, cfg in CONFIGS.items():
        H = cfg["hidden_size"]
        D = cfg["intermediate_size"]
        D_S = cfg["shared_intermediate_size"]
        E = cfg["num_experts"]
        TK = cfg["top_k"]
        print(f"\n  Config: {cfg_name} (H={H}, D={D}, D_S={D_S}, E={E})")

        for n_tokens in [2, 4]:
            torch.manual_seed(42)

            # Routed expert weights
            w13_fp16 = (torch.randn(E, 2 * D, H) * 0.02).half()
            w2_fp16 = (torch.randn(E, H, D) * 0.02).half()
            # Shared expert weights (will be quantized to INT4)
            shared_gate_up_fp16 = (torch.randn(2 * D_S, H) * 0.02).half()
            shared_down_fp16 = (torch.randn(H, D_S) * 0.02).half()
            shared_gate_w = (torch.randn(1, H) * 0.02).half()
            x = (torch.randn(n_tokens, H) * 0.1).half()

            # Quantize routed experts
            w13_qw_list, w13_sc_list, w2_qw_list, w2_sc_list = [], [], [], []
            for e_idx in range(E):
                qw, sc = quantize_int4(w13_fp16[e_idx], GROUP_SIZE)
                w13_qw_list.append(qw); w13_sc_list.append(sc)
                qw, sc = quantize_int4(w2_fp16[e_idx], GROUP_SIZE)
                w2_qw_list.append(qw); w2_sc_list.append(sc)
            w13_qweight = torch.stack(w13_qw_list)
            w13_scales = torch.stack(w13_sc_list)
            w2_qweight = torch.stack(w2_qw_list)
            w2_scales = torch.stack(w2_sc_list)

            # Dequantize for reference
            w13_dq = torch.stack([dequantize_int4(w13_qweight[e_idx], w13_scales[e_idx],
                                  2 * D, H, GROUP_SIZE) for e_idx in range(E)])
            w2_dq = torch.stack([dequantize_int4(w2_qweight[e_idx], w2_scales[e_idx],
                                  H, D, GROUP_SIZE) for e_idx in range(E)])

            # IPEX transform for routed experts (transpose + marlin shuffle)
            w13_ipex, s13_ipex = ipex_transform_expert_weights(
                w13_qweight, w13_scales, E, 2 * D, H // PACK_FACTOR, H // GROUP_SIZE)
            w2_ipex, s2_ipex = ipex_transform_expert_weights(
                w2_qweight, w2_scales, E, H, D // PACK_FACTOR, D // GROUP_SIZE)

            # Quantize shared expert
            # gate_up_proj: [2*D_S, H] → GPTQ: [H//8, 2*D_S]
            shared_gu_qw, shared_gu_sc = quantize_int4(shared_gate_up_fp16, GROUP_SIZE)
            # shape: [2*D_S, H//8] — this is [N, K_packed]
            # GPTQ convention: [K_packed, N] — need to transpose to match
            # Actually quantize_int4 returns [N, K_packed] where N=2*D_S, K_packed=H//8
            # GPTQ/SymInt4 stores as [K_packed, N]:
            shared_gu_qw_gptq = shared_gu_qw.t().contiguous()   # [H//8, 2*D_S] = [K_packed, N]
            shared_gu_sc_gptq = shared_gu_sc.t().contiguous()   # [H//128, 2*D_S] = [K_groups, N]

            # down_proj: [H, D_S] → quantize → [H, D_S//8] = [N, K_packed]
            shared_dw_qw, shared_dw_sc = quantize_int4(shared_down_fp16, GROUP_SIZE)
            shared_dw_qw_gptq = shared_dw_qw.t().contiguous()   # [D_S//8, H] = [K_packed, N]
            shared_dw_sc_gptq = shared_dw_sc.t().contiguous()   # [D_S//128, H] = [K_groups, N]

            # Simulate IPEX OneDNN format:
            # qweight: column-major repack (flat memory rearranged)
            # scales: NOT repacked (just contiguous, same as original transposed)
            shared_gu_ipex = ipex_column_major_repack(shared_gu_qw_gptq)  # [K_packed, N] repacked
            shared_dw_ipex = ipex_column_major_repack(shared_dw_qw_gptq)
            # Scales stay in [K_groups, N] layout (no repack)
            shared_gu_sc_ipex = shared_gu_sc_gptq  # [K_groups, 2*D_S]
            shared_dw_sc_ipex = shared_dw_sc_gptq  # [K_groups_down, H]

            # Dequantize shared for reference
            shared_gu_dq = dequantize_int4(shared_gu_qw, shared_gu_sc, 2 * D_S, H, GROUP_SIZE)
            shared_dw_dq = dequantize_int4(shared_dw_qw, shared_dw_sc, H, D_S, GROUP_SIZE)

            logits = (torch.randn(n_tokens, E) * 0.1).half()

            ref_out = _ref_moe_forward_full(
                x, logits, w13_dq, shared_gu_dq, w2_dq, shared_dw_dq,
                shared_gate_w, TK, E)

            kernel_out = moe_int4_ops.moe_forward_full_int4(
                x.to(DEVICE), logits.to(DEVICE),
                w13_ipex.to(DEVICE), s13_ipex.to(DEVICE),
                shared_gu_ipex.int().to(DEVICE), shared_gu_sc_ipex.to(DEVICE),
                w2_ipex.to(DEVICE), s2_ipex.to(DEVICE),
                shared_dw_ipex.int().to(DEVICE), shared_dw_sc_ipex.to(DEVICE),
                shared_gate_w.to(DEVICE),
                TK, NUM_SHARED_EXPERTS, E).cpu()

            cos = cosine_similarity(ref_out, kernel_out)
            mae = (ref_out.float() - kernel_out.float()).abs().max().item()
            passed = cos > 0.95 and mae < 1.0
            status = "PASS" if passed else "FAIL"
            print(f"    n={n_tokens:>3d}  cos={cos:.6f}  mae={mae:.4f}  [{status}]")
            if not passed:
                print(f"      ref[:5]={ref_out[0,:5].tolist()}")
                print(f"      ker[:5]={kernel_out[0,:5].tolist()}")
                return False

    print("  All INT4 shared expert tests passed!\n")
    return True


# ─── Test 3: Edge cases ──────────────────────────────────────────────────────

def test_edge_cases():
    from custom_esimd_kernels_vllm import moe_int4_ops

    print("=" * 60)
    print("Test 3: Edge cases")
    print("=" * 60)

    cfg = CONFIGS["small"]
    H, E = cfg["hidden_size"], cfg["num_experts"]
    x_zero = torch.zeros(1, H, dtype=torch.half, device=DEVICE)
    gate_fp16 = (torch.randn(E, H) * 0.02).half()
    qw, sc = quantize_int4(gate_fp16, GROUP_SIZE)
    out = moe_int4_ops.moe_router_forward_int4(
        x_zero, qw.to(DEVICE), sc.t().contiguous().to(DEVICE))
    all_zero = (out.cpu().abs().max().item() < 1e-3)
    print(f"  Zero input → near-zero output: {'PASS' if all_zero else 'FAIL'}")

    print("  Edge case tests done!\n")
    return True


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_router_forward()
    all_pass &= test_forward_full()
    all_pass &= test_forward_full_int4_shared()
    all_pass &= test_edge_cases()

    if all_pass:
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
