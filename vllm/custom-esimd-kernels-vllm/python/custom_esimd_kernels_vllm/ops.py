"""Python wrappers for custom ESIMD kernels."""
import torch
import torch.nn.functional as F

_ops = torch.ops.custom_esimd_kernels_vllm


def esimd_gemv_fp8_pern(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
    N: int, K: int,
) -> torch.Tensor:
    """FP8 weight GEMV with per-N scale, FP32 accumulation, deferred scale.

    input: [1, K] fp16, weight: [N, K] fp8_e4m3, scale: [N] fp16, output: [1, N] fp16.
    K must be 256-aligned. N must be 8-aligned.
    """
    return _ops.esimd_gemv_fp8_pern(input, weight, weight_scale, output, N, K)


def esimd_gemv_fp8_pern_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor, N0: int,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor, N1: int,
    K: int,
) -> torch.Tensor:
    """Fused FP8 GEMV for 2 weight matrices sharing the same input and K.

    Single kernel submit: eliminates redundant launch overhead.
    Each weight/scale/output is independent; results written to o0 and o1.
    Returns o0 (first output tensor).
    """
    return _ops.esimd_gemv_fp8_pern_fused2(input, w0, s0, o0, N0, w1, s1, o1, N1, K)


def esimd_gemv_fp8_pern_fused3(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor, N0: int,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor, N1: int,
    w2: torch.Tensor, s2: torch.Tensor, o2: torch.Tensor, N2: int,
    K: int,
) -> torch.Tensor:
    """Fused FP8 GEMV for 3 weight matrices sharing the same input and K.

    Single kernel submit: eliminates redundant launch overhead.
    Each weight/scale/output is independent; results written to o0, o1, o2.
    Returns o0 (first output tensor).
    """
    return _ops.esimd_gemv_fp8_pern_fused3(input, w0, s0, o0, N0, w1, s1, o1, N1, w2, s2, o2, N2, K)


# ---- Per-tensor scale variants (N/K auto-detected from weight shape) ----

def esimd_gemv_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 weight GEMV with per-tensor scale (fp32 scalar).

    input: [1, K] fp16, weight: [N, K] fp8_e4m3, scale: fp32 scalar, output: [1, N] fp16.
    N and K are inferred from weight shape.
    """
    return _ops.esimd_gemv_fp8_pert(input, weight, weight_scale, output)


def esimd_gemv_fp8_pert_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
) -> torch.Tensor:
    """Fused FP8 GEMV for 2 weight matrices with per-tensor scale.

    N0, N1 inferred from w0.size(0), w1.size(0). K from w0.size(1).
    """
    return _ops.esimd_gemv_fp8_pert_fused2(input, w0, s0, o0, w1, s1, o1)


def esimd_gemv_fp8_pert_fused3(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
    w2: torch.Tensor, s2: torch.Tensor, o2: torch.Tensor,
) -> torch.Tensor:
    """Fused FP8 GEMV for 3 weight matrices with per-tensor scale.

    N0, N1, N2 inferred from w0/w1/w2.size(0). K from w0.size(1).
    """
    return _ops.esimd_gemv_fp8_pert_fused3(input, w0, s0, o0, w1, s1, o1, w2, s2, o2)


# ---- INT4 GEMV with per-group scale (group_size=128) ----

def esimd_gemv_int4(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Symmetric INT4 weight GEMV with per-group scale, FP32 accumulation.

    Computes: output[1, N] = input[1, K] @ dequant(weight)^T
    where dequant unpacks int4 values and multiplies by per-group scale.

    input:        [1, K]            fp16  — input activation vector
    weight:       [N, K/2]          uint8 — packed INT4 (2 values per byte,
                                            low nibble = even index)
    weight_scale: [N, K/128]        fp16  — per-group scale (group_size=128)
    output:       [1, N]            fp16  — pre-allocated output buffer

    N inferred from weight.size(0), K inferred from weight.size(1) * 2.
    K must be a multiple of 128 (group_size).
    """
    return _ops.esimd_gemv_int4(input, weight, weight_scale, output)


def esimd_gemv_int4_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
) -> torch.Tensor:
    """Fused 2-matrix INT4 GEMV: two GEMVs sharing the same input, single kernel.

    Saves one kernel launch overhead (~20-50 us) compared to two separate calls.
    Used for GDN input projection: in_proj_qkvz (w0) + in_proj_ba (w1).

    input: [1, K]       fp16 — shared input
    w0:    [N0, K/2]    uint8, s0: [N0, K/128] fp16, o0: [1, N0] fp16
    w1:    [N1, K/2]    uint8, s1: [N1, K/128] fp16, o1: [1, N1] fp16

    Returns o0. Both o0 and o1 are written.
    """
    return _ops.esimd_gemv_int4_fused2(input, w0, s0, o0, w1, s1, o1)


# ---- Fused QKV Split + RMSNorm + RoPE ----

def esimd_qkv_split_norm_rope(
    qkv_state: torch.Tensor,
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    norm_wq: torch.Tensor,
    norm_wk: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    attn_output_gate: bool,
    rotary_dim: int = 256,
    cos_sin_cache: torch.Tensor = None,
) -> torch.Tensor:
    """Fused QKV Split + RMSNorm(weight+1.0, eps=1e-6) + RoPE.

    qkv_state:     [nTokens, hiddenDim] fp16 — packed QKV projection output
    q_out:         [nTokens, qHead*256] fp16
    gate_out:      [nTokens, qHead*256] fp16 (unused if not attn_output_gate)
    k_out:         [nTokens, kvHead*256] fp16
    v_out:         [nTokens, kvHead*256] fp16
    norm_wq/wk:    [256] fp16 — RMSNorm weights (Qwen3 weight+1.0 convention)
    positions:     [nTokens] int32 — RoPE position indices
    rotary_dim:    number of dimensions to apply RoPE.
    cos_sin_cache: [max_pos, rotary_dim] fp16 — from rotary_emb.cos_sin_cache.
                   Layout: [cos(rotary_dim/2), sin(rotary_dim/2)] per row.
    headDim=256 only.
    """
    return _ops.esimd_qkv_split_norm_rope(
        qkv_state, q_out, gate_out, k_out, v_out,
        norm_wq, norm_wk, positions,
        q_heads, kv_heads, attn_output_gate, rotary_dim, cos_sin_cache)


# ---- Fused Conv1d + GDN (doubleGRF, LGRF module) ----

def esimd_gdn_conv_fused(
    qkvz: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ba: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    output: torch.Tensor,
    z_out: torch.Tensor,
    N: int, H: int, HV: int,
    K: int, V: int,
    scale: float,
) -> torch.Tensor:
    """Fused Conv1d + GDN for Qwen3-Next-80B-A3B decode.

    Reads directly from projection outputs — zero extra submits.
    Phase 1: Conv1d with SiLU, reads x from qkvz at mapped offsets.
    Phase 2: GDN recurrent update.
    Phase 3: conv_state shift + z extraction from qkvz.

    qkvz:               [N, qkvz_dim] fp16 — projected_states_qkvz (read-only)
    conv_state:         [num_cache, 3, 2048] fp16, strided dim0
    conv_weight:        [2048, 4] fp16
    conv_bias:          [2048] fp16 (zeros if no bias)
    conv_state_indices: [N] int32
    A_log:              [HV] fp16
    dt_bias:            [HV] fp16
    ba:                 [N, 2*HV] fp16 — projected_states_ba, interleaved layout
    ssm_state:          [num_states, HV, V, K] fp16, strided dim0
    ssm_state_indices:  [N] int32
    output:             [N, HV, V] fp16 — GDN output (core_attn_out)
    z_out:              [N, HV, V] fp16 — z gate extracted from qkvz
    """
    return _ops.esimd_gdn_conv_fused(
        qkvz, conv_state, conv_weight, conv_bias, conv_state_indices,
        A_log, dt_bias, ba,
        ssm_state, ssm_state_indices, output, z_out,
        N, H, HV, K, V, scale)


def esimd_fused_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused residual add + RMSNorm (Gemma-style).

    residual = hidden_states + residual  (in-place)
    hidden_states = rmsnorm(residual) * weight  (output)
    weight must be pre-adjusted (w+1.0).
    """
    return _ops.esimd_fused_add_rms_norm(hidden_states, residual, weight, eps)


def esimd_fused_add_rms_norm_batched(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Batched fused residual add + RMSNorm (Gemma-style).

    residual[i] = hidden_states[i] + residual[i]  (in-place)
    hidden_states[i] = rmsnorm(residual[i]) * weight  (output)
    weight must be pre-adjusted (w+1.0). Works for any number of rows.
    """
    return _ops.esimd_fused_add_rms_norm_batched(hidden_states, residual, weight, eps)


def esimd_rms_norm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """ESIMD RMSNormGated: output = rmsnorm(x) * weight * silu(z).

    x, z: [rows, V] fp16. weight: [V] fp16. output: [rows, V] fp16.
    Single kernel replaces ~6 PyTorch dispatches (87us → ~5us).
    """
    return _ops.esimd_rms_norm_gated(x, z, weight, output, eps)


def esimd_resadd_norm_gemv_fp8_pert(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    normed_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ResidualAdd + RMSNorm + FP8 GEMV.

    Combines post_attention_layernorm + MoE router GEMV:
      1. residual = hidden_states + residual  (in-place)
      2. normed = rmsnorm(residual) * norm_weight  (Gemma-style, w+1 pre-applied)
      3. output = normed @ dequant(gemv_weight^T) * scale
      4. normed_out = normed  (for MoE expert consumption)

    hidden_states: [1, K] fp16
    residual:      [1, K] fp16 (updated in-place)
    norm_weight:   [K] fp16 (Gemma _gemma_w)
    gemv_weight:   [N, K] FP8
    gemv_scale:    [1] fp32
    output:        [1, N] fp16 — router logits
    normed_out:    [1, K] fp16 — normed hidden for experts
    """
    return _ops.esimd_resadd_norm_gemv_fp8_pert(
        hidden_states, residual, norm_weight,
        gemv_weight, gemv_scale, output, normed_out, eps)


def esimd_resadd_norm_gemv_int4_pert(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    normed_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ResidualAdd + RMSNorm + INT4 GEMV.

    Combines post_attention_layernorm + MoE router GEMV (INT4 quantized):
      1. residual = hidden_states + residual  (in-place)
      2. normed = rmsnorm(residual) * norm_weight
      3. output = normed @ dequant(int4_weight^T) (per-block scale)
      4. normed_out = normed  (for MoE expert consumption)

    hidden_states: [1, K] fp16
    residual:      [1, K] fp16 (updated in-place)
    norm_weight:   [K] fp16
    gemv_weight:   [N, K//8] int32 packed INT4
    gemv_scale:    [N, K//128] fp16 — per-block scale
    output:        [1, N] fp16 — router logits
    normed_out:    [1, K] fp16 — normed hidden for experts
    """
    return _ops.esimd_resadd_norm_gemv_int4_pert(
        hidden_states, residual, norm_weight,
        gemv_weight, gemv_scale, output, normed_out, eps)


def esimd_resadd_norm_gemv2_fp8_pert(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ResidualAdd + RMSNorm + 2-matrix FP8 GEMV.

    For input_layernorm + GDN in_proj (qkvz + ba projections).
    residual updated in-place. o0/o1 are output buffers.
    """
    return _ops.esimd_resadd_norm_gemv2_fp8_pert(
        hidden_states, residual, norm_weight,
        w0, s0, o0, w1, s1, o1, eps)


def esimd_norm_gemv_fp8_pert(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    HV: int,
    V: int,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNormGated + FP8 GEMV for GDN out_proj decode path.

    Combines norm(x, z) + out_proj(normed) into a single kernel.
    Eliminates norm kernel launch, torch.empty, reshape overhead.

    x:            [HV, V] fp16 — core_attn_out
    z:            [HV, V] fp16 — z_out
    norm_weight:  [V] fp16 — RMSNorm weight
    gemv_weight:  [N, K] FP8, K = HV*V — out_proj weight
    gemv_scale:   [1] fp32 — per-tensor scale
    output:       [1, N] fp16 — pre-allocated output buffer
    """
    return _ops.esimd_norm_gemv_fp8_pert(
        x, z, norm_weight, gemv_weight, gemv_scale, output,
        HV, V, eps)


def esimd_norm_gemv_int4_pert(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    gemv_weight: torch.Tensor,
    gemv_scale: torch.Tensor,
    output: torch.Tensor,
    HV: int,
    V: int,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNormGated + INT4 GEMV for GDN out_proj decode path.

    Combines norm(x, z) + out_proj(normed) into a single kernel.
    INT4 analogue of esimd_norm_gemv_fp8_pert.

    x:            [HV, V] fp16 — core_attn_out
    z:            [HV, V] fp16 — z_out
    norm_weight:  [V] fp16 — RMSNorm weight
    gemv_weight:  [N, K//8] int32 packed INT4, K = HV*V — out_proj weight
    gemv_scale:   [N, K//128] fp16 — per-block INT4 scale
    output:       [1, N] fp16 — pre-allocated output buffer
    """
    return _ops.esimd_norm_gemv_int4_pert(
        x, z, norm_weight, gemv_weight, gemv_scale, output,
        HV, V, eps)


def esimd_gdn_conv_fused_seq(
    qkvz: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ba: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    output: torch.Tensor,
    z_out: torch.Tensor,
    N: int, H: int, HV: int,
    K: int, V: int,
    scale: float,
) -> torch.Tensor:
    """Fused Conv1d + GDN for SEQUENTIAL qkvz layout [q|k|v|z].

    Same as esimd_gdn_conv_fused but reads qkvz in sequential order
    instead of GQA-interleaved. For models like Qwen3.5-35B-A3B where
    MergedColumnParallelLinear outputs [q_all|k_all|v_all|z_all].

    ba is also sequential: [b_all(HV) | a_all(HV)].

    Eliminates ALL host-side rearrangement (no cat, reshape, gather).
    """
    return _ops.esimd_gdn_conv_fused_seq(
        qkvz, conv_state, conv_weight, conv_bias, conv_state_indices,
        A_log, dt_bias, ba,
        ssm_state, ssm_state_indices, output, z_out,
        N, H, HV, K, V, scale)


# ---- MoE Auxiliary Ops (doubleGRF, LGRF module) ----

def esimd_moe_topk(
    router_logits: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Fused softmax + top-8 selection + normalize.

    router_logits: [T, 128] fp16
    top_values:    [T, 8] fp16 (output)
    top_indices:   [T, 8] int32 (output)
    """
    return _ops.esimd_moe_topk(router_logits, top_values, top_indices, T)


def esimd_moe_scatter(
    hidden_states: torch.Tensor,
    router_top_value: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    scattered_hidden: torch.Tensor,
    scattered_weights: torch.Tensor,
    K: int,
    topk: int,
    total_expanded: int,
) -> torch.Tensor:
    """Scatter hidden_states by expert grouping.

    hidden_states:    [T, K] fp16
    router_top_value: [T, topk] fp16
    sorted_token_ids: [total_expanded] int32
    scattered_hidden: [total_expanded, K] fp16 (output)
    scattered_weights:[total_expanded] fp16 (output)
    """
    return _ops.esimd_moe_scatter(
        hidden_states, router_top_value, sorted_token_ids,
        scattered_hidden, scattered_weights, K, topk, total_expanded)


def esimd_moe_scatter_fused(
    hidden_states: torch.Tensor,
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    scattered_hidden: torch.Tensor,
    scattered_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: torch.Tensor,
    max_tokens_out: torch.Tensor,
    K: int,
    topk: int,
    T: int,
    num_experts: int,
) -> torch.Tensor:
    """Fused GPU scatter: atomic counting + prefix-sum + copy. No CPU preprocessing.

    hidden_states:    [T, K] fp16
    top_values:       [T, topk] fp16
    top_indices:      [T, topk] int32
    scattered_hidden: [T*topk, K] fp16 (output)
    scattered_weights:[T*topk] fp16 (output)
    topk_ids:         [T*topk] int32 (output — reverse map for Gather)
    expert_start:     [num_experts+1] uint32 (output)
    max_tokens_out:   [1] int32 (output)
    """
    return _ops.esimd_moe_scatter_fused(
        hidden_states, top_values, top_indices,
        scattered_hidden, scattered_weights,
        topk_ids, expert_start, max_tokens_out,
        K, topk, T, num_experts)


def esimd_moe_silu_mul(
    input: torch.Tensor,
    output: torch.Tensor,
    N_gate_up: int,
    N_half: int,
    total_rows: int,
) -> torch.Tensor:
    """SiLU(gate) * up activation.

    input:  [total_rows, N_gate_up] fp16
    output: [total_rows, N_half] fp16
    """
    return _ops.esimd_moe_silu_mul(input, output, N_gate_up, N_half, total_rows)


def esimd_moe_gather(
    moe_output: torch.Tensor,
    topk_ids: torch.Tensor,
    scattered_weights: torch.Tensor,
    final_hidden: torch.Tensor,
    K: int,
    topk: int,
    T: int,
) -> torch.Tensor:
    """Weighted gather/reduce from scattered expert outputs.

    moe_output:       [total_expanded, K] fp16
    topk_ids:         [T, topk] int32
    scattered_weights:[total_expanded] fp16
    final_hidden:     [T, K] fp16 (output)
    """
    return _ops.esimd_moe_gather(
        moe_output, topk_ids, scattered_weights, final_hidden, K, topk, T)


def esimd_moe_gemm_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    expert_idx: torch.Tensor,
    N: int,
    K: int,
    num_experts: int,
    max_tokens_per_expert: int,
) -> torch.Tensor:
    """MoE grouped GEMM — FP8 E5M2 with per-N scale.

    input:      [total_tokens, K] fp16
    weight:     [num_experts, N, K] uint8 FP8 E5M2
    scale:      [num_experts, N] float32
    output:     [total_tokens, N] fp16
    expert_idx: [num_experts+1] uint32 — token start offsets per expert
    """
    return _ops.esimd_moe_gemm_fp8(
        input, weight, scale, output, expert_idx,
        N, K, num_experts, max_tokens_per_expert)


def esimd_gemm_fp8_pert(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP8 GEMM with per-tensor scale — handles any M (auto-dispatches).

    input:  [M, K] fp16, weight: [N, K] fp8, scale: fp32 scalar, output: [M, N] fp16.
    N and K are inferred from weight shape. M from input shape.

    Auto-dispatch:
      M=1-3  → batched GEMV (BW-bound, K-split SLM reduction)
      M>=2   → DPAS V9 (E4M3, K%64==0) or DPAS V7 (E5M2) or WS fallback
    """
    return _ops.esimd_gemm_fp8_pert(input, weight, weight_scale, output)


def esimd_moe_gemm_fp8_pert(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    expert_idx: torch.Tensor,
    N: int,
    K: int,
    num_experts: int,
    max_tokens_per_expert: int,
) -> torch.Tensor:
    """MoE grouped GEMM — FP8 E5M2 with per-tensor scale (one per expert).

    input:      [total_tokens, K] fp16
    weight:     [num_experts, N, K] uint8 FP8 E5M2
    scale:      [num_experts] float32 — one scalar per expert
    output:     [total_tokens, N] fp16
    expert_idx: [num_experts+1] uint32 — token start offsets per expert
    """
    return _ops.esimd_moe_gemm_fp8_pert(
        input, weight, scale, output, expert_idx,
        N, K, num_experts, max_tokens_per_expert)


# ---- Eagle Ops (GDN + Page Attention) ----

_eagle_ops = torch.ops.eagle_ops


def eagle_gdn(
    qkvz: torch.Tensor,
    z_out: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    conv_state: torch.Tensor,
    accepted_tokens: torch.Tensor,
    ba: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_in: torch.Tensor,
    ssm_state_idx: torch.Tensor,
    norm_w: torch.Tensor,
    max_query_len: int,
) -> torch.Tensor:
    """Eagle GDN fused kernel: Conv1d + SSM + Attention.

    qkvz:            [batches, dim] fp16 — packed projection output
    z_out:           [batches, HV*V] fp16 — z gate output
    conv_w:          [dim, kernel_size] fp16
    conv_b:          [dim] fp16 or None
    conv_state:      [num_cache, kernel_size-1, dim] fp16
    accepted_tokens: [batches] int32
    ba:              [batches, 2*HV] fp16
    a_log:           [HV] fp16
    dt_bias:         [HV] fp16
    state_in:        [num_states, HV, V, K] fp16
    ssm_state_idx:   [batches] int32
    norm_w:          [dim] fp16
    max_query_len:   int
    """
    return _eagle_ops.gdn_eagle(
        qkvz, z_out, conv_w, conv_b, conv_state,
        accepted_tokens, ba, a_log, dt_bias,
        state_in, ssm_state_idx, norm_w, max_query_len)


def eagle_page_attn_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
) -> None:
    """Eagle paged attention decode.

    query:       [batches, num_heads, head_dim] fp16
    kv_cache:    paged KV cache tensor
    block_table: [batches, max_blocks] int32
    seq_lens:    [batches] int32
    out:         [batches, num_heads, head_dim] fp16 (output)
    """
    return _eagle_ops.page_attn_decode(
        query, kv_cache, block_table, seq_lens, out,
        max_query_len, max_seq_len)


# ---- MoE Batch Ops (Router, TopK, Up/Down, Accumulate) ----

_moe_batch = torch.ops.moe_ops


def moe_router_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """MoE router forward: batched GEMV with weight reuse.

    x:      [n_tokens, hidden_size] fp16
    weight: [num_experts, hidden_size] fp8
    scale:  [num_experts] fp32
    Returns: [n_tokens, num_experts] fp16
    """
    return _moe_batch.moe_router_forward(x, weight, scale)


def moe_batch_topk(
    logits: torch.Tensor,
    top_k: int,
    norm: bool = True,
) -> tuple:
    """MoE fused softmax + top-k selection + normalize.

    logits: [n_tokens, num_experts] fp16
    top_k:  number of experts to select
    norm:   whether to normalize top-k weights
    Returns: (top_values [n_tokens, top_k] fp16, top_indices [n_tokens, top_k] int32)
    """
    return _moe_batch.moe_topk(logits, top_k, norm)


def moe_up_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    selected_experts: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
) -> torch.Tensor:
    """MoE gate+up projection with SiLU (routed + shared experts).

    x:                    [n_tokens, hidden_size] fp16
    gate_up_weight:       [num_experts, hidden_size, 2*intermediate_size] fp8
    gate_up_scale:        [num_experts] fp32
    shared_gate_up_weight:[num_shared, 2*intermediate_size, hidden_size] fp8
    shared_gate_up_scale: [num_shared] fp32
    selected_experts:     [n_tokens, top_k] int32
    Returns: [n_tokens * (top_k + num_shared), intermediate_size] fp16
    """
    return _moe_batch.moe_up_forward(
        x, gate_up_weight, gate_up_scale,
        shared_gate_up_weight, shared_gate_up_scale,
        selected_experts, top_k, num_shared_experts)


def moe_down_forward(
    x: torch.Tensor,
    intermediates: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
) -> torch.Tensor:
    """MoE down projection (routed + shared experts).

    x:                        [n_tokens, hidden_size] fp16
    intermediates:            [n_tokens * (top_k + num_shared), intermediate_size] fp16
    down_weight:              [num_experts, intermediate_size, hidden_size] fp8
    down_scale:               [num_experts] fp32
    shared_down_weight:       [num_shared, hidden_size, intermediate_size] fp8
    shared_down_scale:        [num_shared] fp32
    shared_expert_gate_weight:[num_shared, hidden_size] fp16
    routing_weights:          [n_tokens, top_k] fp16
    selected_experts:         [n_tokens, top_k] int32
    Returns: [n_tokens * (top_k + num_shared), hidden_size] fp16
    """
    return _moe_batch.moe_down_forward(
        x, intermediates, down_weight, down_scale,
        shared_down_weight, shared_down_scale,
        shared_expert_gate_weight, routing_weights,
        selected_experts, top_k, num_shared_experts)


def moe_accumulate(
    partials: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
) -> torch.Tensor:
    """Accumulate expert outputs per token.

    partials: [n_tokens * (top_k + num_shared), hidden_size] fp16
    Returns:  [n_tokens, hidden_size] fp16
    """
    return _moe_batch.moe_accumulate(partials, top_k, num_shared_experts)


def moe_forward_fused(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
) -> torch.Tensor:
    """MoE fused forward: up + down_routed + down_finalize in one C++ call.

    Requires routing_weights and selected_experts to be pre-computed.
    """
    return _moe_batch.moe_forward_fused(
        x, gate_up_weight, gate_up_scale,
        shared_gate_up_weight, shared_gate_up_scale,
        down_weight, down_scale,
        shared_down_weight, shared_down_scale,
        shared_expert_gate_weight,
        routing_weights, selected_experts,
        top_k, num_shared_experts)


def moe_forward_full(
    x: torch.Tensor,
    logits: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """MoE full forward: topk + up + down_routed + down_finalize in one C++ call.

    Pre-allocates buffers to eliminate torch::empty overhead.
    """
    return _moe_batch.moe_forward_full(
        x, logits, gate_up_weight, gate_up_scale,
        shared_gate_up_weight, shared_gate_up_scale,
        down_weight, down_scale,
        shared_down_weight, shared_down_scale,
        shared_expert_gate_weight,
        top_k, num_shared_experts, n_routed_experts)


# ═══════════════════════════════════════════════════════════════════════════════
# MoE INT4 Batch ops
# ═══════════════════════════════════════════════════════════════════════════════

_moe_int4 = torch.ops.moe_int4_ops


def moe_router_forward_int4(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    use_ggml_layout: bool = False,
) -> torch.Tensor:
    """INT4 router GEMV: x @ dequant(weight).T → logits.

    x:      [n_tokens, hidden_size] fp16
    weight: [num_experts, hidden_size//8] int32 (or uint8 viewed as int32)
    scale:  fp16

    use_ggml_layout=False (IPEX): weight [E, K_packed] after IPEX repack,
        scale [K_groups, E] (kernel reads with stride).
    use_ggml_layout=True (GGML): weight_esimd [E, K/2] uint8 → [E, K/8] int32,
        scale_esimd [E, K_groups] contiguous (kernel reads row-major).
    Returns: [n_tokens, num_experts] fp16
    """
    return _moe_int4.moe_router_forward_int4(x, weight, scale, use_ggml_layout)


def moe_router_topk_int4(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    use_ggml_layout: bool,
    top_k: int,
    n_routed_experts: int,
    norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT4 router followed by the same C++ TopK path as ``moe_forward_full_int4``."""
    return _moe_int4.moe_router_topk_int4(
        x.contiguous(), weight, scale, use_ggml_layout,
        top_k, n_routed_experts, norm)


def moe_forward_full_int4(
    x: torch.Tensor,
    logits: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    n_routed_experts: int,
    use_ggml_layout: bool = False,
) -> torch.Tensor:
    """INT4 MoE full forward: topk + up + down + finalize in one C++ call.

    Supports both INT4 and FP16 shared expert weights (auto-detected by dtype).
    When shared expert is INT4: shared_gate_up_scale/shared_down_scale are used.
    When shared expert is FP16: pass dummy tensors for scales (ignored).

    use_ggml_layout: if True, routed expert weights are in GGML N-major layout
        [E, N, K_packed] with natural nibble order (transpose=False from ggml_quantize_tensor).
        If False (default), expects IPEX K-major layout [E, K_packed, N] with marlin shuffled nibbles.
    """
    return _moe_int4.moe_forward_full_int4(
        x, logits,
        gate_up_qweight, gate_up_scales,
        shared_gate_up_weight, shared_gate_up_scale,
        down_qweight, down_scales,
        shared_down_weight, shared_down_scale,
        shared_expert_gate_weight,
        top_k, num_shared_experts, n_routed_experts,
        use_ggml_layout)


def moe_shared_expert_forward_int4_nmajor(
    x: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scale: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Shared expert forward with CUTLASS N-major uint8 INT4 weights.

    x:                [n_tokens, H] fp16
    gate_up_qweight:  [2*I, H/2] uint8 (implement_zp signed encoding)
    gate_up_scale:    [2*I, H/GS] fp16
    down_qweight:     [H, I/2] uint8 (implement_zp signed encoding)
    down_scale:       [H, I/GS] fp16
    gate_weight:      [num_shared, H] fp16

    Returns: [n_tokens, H] fp16
    """
    return _moe_int4.moe_shared_expert_forward_int4_nmajor(
        x, gate_up_qweight, gate_up_scale,
        down_qweight, down_scale, gate_weight)


def moe_topk_int4(
    logits: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
    norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT4 MoE TopK using the same C++ kernel path as ``moe_forward_full_int4``."""
    return _moe_int4.moe_topk_int4(logits.contiguous(), top_k, n_routed_experts, norm)


def to_cutlass_nmajor_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Convert INT4 weights to CUTLASS-style N-major uint8 packing.

    Input can be GGML/test-style int32 ``[E, N, K/8]`` or ``[N, K/8]`` with
    8 unsigned int4 values per int32. The output is uint8 ``[E, N, K/2]`` or
    ``[N, K/2]`` with low nibble = even K and high nibble = odd K.

    If ``qweight`` is already uint8, this returns a contiguous copy/view.
    """
    if qweight.dtype == torch.uint8:
        return qweight.contiguous()
    if qweight.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"unsupported qweight dtype: {qweight.dtype}")
    if qweight.dim() not in (2, 3):
        raise ValueError(f"expected [N,K/8] or [E,N,K/8], got {tuple(qweight.shape)}")

    q_u32 = qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int64) * 4
    nibbles = ((q_u32.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    nibbles = nibbles.reshape(*qweight.shape[:-1], qweight.shape[-1] * 8)
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).contiguous()


def cutlass_nmajor_int4_to_signed(qweight_u4: torch.Tensor) -> torch.Tensor:
    """Convert unsigned CUTLASS N-major uint4 bytes to signed compact int4.

    This mirrors ``vllm_xpu_kernels.fused_moe_interface.implement_zp`` and is
    intended to be run once during weight preparation, not inside decode.
    """
    if qweight_u4.dtype != torch.uint8:
        raise TypeError(f"expected uint8 qweight, got {qweight_u4.dtype}")
    try:
        from vllm_xpu_kernels.fused_moe_interface import implement_zp
    except Exception as exc:
        raise RuntimeError("vllm_xpu_kernels is required for signed INT4 packing") from exc

    if qweight_u4.dim() == 2:
        return implement_zp(qweight_u4.contiguous())
    if qweight_u4.dim() != 3:
        raise ValueError(f"expected [N,K/2] or [E,N,K/2], got {tuple(qweight_u4.shape)}")

    qweight_s4 = torch.empty_like(qweight_u4)
    for expert in range(qweight_u4.shape[0]):
        qweight_s4[expert] = implement_zp(qweight_u4[expert].contiguous())
    return qweight_s4.contiguous()


def prepare_cutlass_nmajor_int4_weight(qweight: torch.Tensor) -> torch.Tensor:
    """Prepare a routed expert INT4 weight for CUTLASS grouped GEMM.

    Converts GGML/test int32 N-major ``[E,N,K/8]`` to CUTLASS uint8 N-major
    ``[E,N,K/2]`` and then applies the signed-s4 zero-point transform expected
    by ``cutlass_grouped_gemm_xe2``.
    """
    return cutlass_nmajor_int4_to_signed(to_cutlass_nmajor_int4(qweight))


def precompute_moe_route(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token routes by expert for grouped GEMM.

    Returns ``sorted_rows``, ``sorted_weights`` and ``rows_per_expert``. This is
    a Python/Torch prototype; a production decode path should replace it with a
    fused C++/SYCL prologue to avoid many tiny launches.
    """
    if topk_weights.device.type == "xpu" and topk_ids.device.type == "xpu":
        try:
            return _moe_int4.moe_route_precompute_int4(
                topk_weights.contiguous(), topk_ids.contiguous(), num_experts)
        except (AttributeError, RuntimeError):
            pass

    num_rows = topk_ids.shape[0]
    top_k = topk_ids.shape[1]
    flat_experts = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)
    flat_rows = torch.arange(num_rows, device=topk_ids.device, dtype=torch.int64)
    flat_rows = flat_rows.repeat_interleave(top_k)

    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_rows = flat_rows[order]
    sorted_weights = flat_weights[order]
    rows_per_expert = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    return sorted_rows.contiguous(), sorted_weights.contiguous(), rows_per_expert.contiguous()


def moe_silu_mul_int4(gate_up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up for routed MoE intermediate tensors."""
    if gate_up.device.type == "xpu":
        try:
            return _moe_int4.moe_silu_mul_int4(gate_up.contiguous())
        except (AttributeError, RuntimeError):
            pass
    inter_size = gate_up.shape[1] // 2
    return (F.silu(gate_up[:, :inter_size].float()) *
            gate_up[:, inter_size:].float()).to(gate_up.dtype).contiguous()


def moe_route_gather_int4(
    route_output: torch.Tensor,
    sorted_rows: torch.Tensor,
    sorted_weights: torch.Tensor,
    n_tokens: int,
) -> torch.Tensor:
    """Gather weighted routed outputs back to token-major order."""
    if route_output.device.type == "xpu":
        try:
            return _moe_int4.moe_route_gather_int4(
                route_output.contiguous(), sorted_rows.contiguous(),
                sorted_weights.contiguous(), n_tokens)
        except (AttributeError, RuntimeError):
            pass
    output = torch.zeros(n_tokens, route_output.shape[1], dtype=route_output.dtype,
                         device=route_output.device)
    output.index_add_(0, sorted_rows, route_output * sorted_weights.unsqueeze(-1))
    return output


def moe_forward_routed_cutlass_nmajor_int4(
    hidden_states: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_experts: int,
    route: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    logits: torch.Tensor | None = None,
    top_k: int = 8,
) -> torch.Tensor:
    """Prototype routed MoE forward using CUTLASS N-major INT4 grouped GEMM.

    ``w13_qweight_s4`` and ``w2_qweight_s4`` must already be prepared by
    ``prepare_cutlass_nmajor_int4_weight``. Shared experts are not included;
    this isolates the routed path for bring-up and benchmarking.
    """
    if topk_weights is None or topk_ids is None:
        if logits is None:
            raise ValueError("pass either topk_weights/topk_ids or logits")
        topk_weights, topk_ids = moe_topk_int4(logits, top_k, num_experts)

    if route is None and hidden_states.shape[0] <= 4:
        return moe_forward_tiny_cutlass_nmajor_int4(
            hidden_states, w13_qweight_s4, w13_scales,
            w2_qweight_s4, w2_scales, topk_weights, topk_ids)

    try:
        from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm_xe2
    except Exception as exc:
        raise RuntimeError("vllm_xpu_kernels is required for CUTLASS grouped GEMM") from exc

    num_rows, hidden_size = hidden_states.shape
    inter_size = w2_qweight_s4.shape[2] * 2
    if route is None:
        sorted_rows, sorted_weights, rows_per_expert = precompute_moe_route(
            topk_weights, topk_ids, num_experts)
    else:
        sorted_rows, sorted_weights, rows_per_expert = route

    gemm1_input = hidden_states.index_select(0, sorted_rows).contiguous()
    gemm1_output = torch.empty(
        gemm1_input.shape[0], w13_qweight_s4.shape[1],
        dtype=hidden_states.dtype, device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        gemm1_input, w13_qweight_s4, w13_scales, None, gemm1_output,
        rows_per_expert, w13_qweight_s4.shape[1], hidden_size, num_experts,
        True, False)

    act_output = moe_silu_mul_int4(gemm1_output)
    gemm2_output = torch.empty(
        gemm1_input.shape[0], hidden_size,
        dtype=hidden_states.dtype, device=hidden_states.device)
    cutlass_grouped_gemm_xe2(
        act_output, w2_qweight_s4, w2_scales, None, gemm2_output,
        rows_per_expert, hidden_size, inter_size, num_experts, True, False)

    return moe_route_gather_int4(gemm2_output, sorted_rows, sorted_weights, num_rows)


def _moe_topk_from_logits(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.device.type == "xpu":
        try:
            topk_weights, topk_ids = moe_topk_int4(logits.contiguous(), top_k, logits.shape[-1], True)
            return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()
        except (AttributeError, RuntimeError):
            pass
    probs = F.softmax(logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(logits.dtype).contiguous(), topk_ids.to(torch.int32).contiguous()


def moe_forward_full_cutlass_nmajor_int4(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    num_experts: int,
) -> torch.Tensor:
    """Prototype full MoE forward using CUTLASS N-major INT4 routed GEMMs.

    This path owns TopK internally so timing is comparable to
    ``moe_forward_full_int4``. Routed experts use CUTLASS grouped GEMM with
    pre-packed signed-s4 N-major weights. Shared experts are currently the
    FP16 path used by Qwen3.5 decode bring-up.
    """
    if num_shared_experts != 1:
        raise NotImplementedError("CUTLASS N-major full prototype currently supports one FP16 shared expert")

    shared_inter_size = shared_down_weight.shape[-1]
    if shared_gate_up_weight.dim() == 3:
        shared_gate_up = shared_gate_up_weight[0]
    else:
        shared_gate_up = shared_gate_up_weight
    if shared_down_weight.dim() == 3:
        shared_down = shared_down_weight[0]
    else:
        shared_down = shared_down_weight
    if shared_expert_gate_weight.dim() == 3:
        shared_gate_weight = shared_expert_gate_weight[0]
    else:
        shared_gate_weight = shared_expert_gate_weight

    if hidden_states.shape[0] <= 32:
        return moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
            hidden_states, logits, w13_qweight_s4, w13_scales,
            w2_qweight_s4, w2_scales, shared_gate_up, shared_down,
            shared_gate_weight, top_k, num_shared_experts, num_experts)

    routed = moe_forward_routed_cutlass_nmajor_int4(
        hidden_states, w13_qweight_s4, w13_scales, w2_qweight_s4, w2_scales,
        None, None, num_experts, logits=logits, top_k=top_k)

    shared_gu = hidden_states @ shared_gate_up.t()
    shared_act = F.silu(shared_gu[:, :shared_inter_size].float()) * shared_gu[:, shared_inter_size:].float()
    shared_out = shared_act.to(hidden_states.dtype) @ shared_down.t()
    gate = torch.sigmoid((hidden_states @ shared_gate_weight.t()).float()).to(hidden_states.dtype)
    return routed + shared_out * gate


def moe_forward_full_cutlass_nmajor_int4_with_router(
    hidden_states: torch.Tensor,
    router_qweight: torch.Tensor,
    router_scales: torch.Tensor,
    router_use_ggml_layout: bool,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    num_experts: int,
) -> torch.Tensor:
    """CUTLASS N-major full MoE path with INT4 router logits computed first."""
    if num_shared_experts != 1:
        raise NotImplementedError("CUTLASS N-major full prototype currently supports one FP16 shared expert")

    shared_inter_size = shared_down_weight.shape[-1]
    if shared_gate_up_weight.dim() == 3:
        shared_gate_up = shared_gate_up_weight[0]
    else:
        shared_gate_up = shared_gate_up_weight
    if shared_down_weight.dim() == 3:
        shared_down = shared_down_weight[0]
    else:
        shared_down = shared_down_weight
    if shared_expert_gate_weight.dim() == 3:
        shared_gate_weight = shared_expert_gate_weight[0]
    else:
        shared_gate_weight = shared_expert_gate_weight

    topk_weights, topk_ids = moe_router_topk_int4(
        hidden_states, router_qweight, router_scales, router_use_ggml_layout,
        top_k, num_experts, True)

    if hidden_states.shape[0] <= 32:
        return moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
            hidden_states, w13_qweight_s4, w13_scales,
            w2_qweight_s4, w2_scales, topk_weights, topk_ids,
            shared_gate_up, shared_down, shared_gate_weight,
            num_shared_experts)

    routed = moe_forward_routed_cutlass_nmajor_int4(
        hidden_states, w13_qweight_s4, w13_scales, w2_qweight_s4, w2_scales,
        topk_weights, topk_ids, num_experts)

    shared_gu = hidden_states @ shared_gate_up.t()
    shared_act = F.silu(shared_gu[:, :shared_inter_size].float()) * shared_gu[:, shared_inter_size:].float()
    shared_out = shared_act.to(hidden_states.dtype) @ shared_down.t()
    gate = torch.sigmoid((hidden_states @ shared_gate_weight.t()).float()).to(hidden_states.dtype)
    return routed + shared_out * gate


def moe_forward_tiny_cutlass_nmajor_int4(
    hidden_states: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """bs1 tiny-M routed MoE using local CUTLASS N-major INT4 kernels."""
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny CUTLASS N-major INT4 path requires XPU")
    return _moe_int4.moe_forward_tiny_cutlass_nmajor_int4(
        hidden_states.contiguous(),
        w13_qweight_s4.contiguous(), w13_scales.contiguous(),
        w2_qweight_s4.contiguous(), w2_scales.contiguous(),
        topk_weights.contiguous(), topk_ids.contiguous())


def moe_tiny_cutlass_nmajor_int4_up(
    hidden_states: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny CUTLASS N-major INT4 path requires XPU")
    return _moe_int4.moe_tiny_cutlass_nmajor_int4_up(
        hidden_states.contiguous(), w13_qweight_s4.contiguous(),
        w13_scales.contiguous(), topk_ids.contiguous())


def moe_tiny_cutlass_nmajor_int4_down(
    intermediates: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if intermediates.device.type != "xpu":
        raise RuntimeError("tiny CUTLASS N-major INT4 path requires XPU")
    return _moe_int4.moe_tiny_cutlass_nmajor_int4_down(
        intermediates.contiguous(), w2_qweight_s4.contiguous(),
        w2_scales.contiguous(), topk_weights.contiguous(), topk_ids.contiguous())


def moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
    hidden_states: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    num_shared_experts: int,
) -> torch.Tensor:
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny CUTLASS N-major INT4 path requires XPU")
    return _moe_int4.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
        hidden_states.contiguous(),
        w13_qweight_s4.contiguous(), w13_scales.contiguous(),
        w2_qweight_s4.contiguous(), w2_scales.contiguous(),
        topk_weights.contiguous(), topk_ids.contiguous(),
        shared_gate_up_weight.contiguous(), shared_down_weight.contiguous(),
        shared_expert_gate_weight.contiguous(), num_shared_experts)


def moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    num_experts: int,
) -> torch.Tensor:
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny CUTLASS N-major INT4 path requires XPU")
    return _moe_int4.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
        hidden_states.contiguous(), logits.contiguous(),
        w13_qweight_s4.contiguous(), w13_scales.contiguous(),
        w2_qweight_s4.contiguous(), w2_scales.contiguous(),
        shared_gate_up_weight.contiguous(), shared_down_weight.contiguous(),
        shared_expert_gate_weight.contiguous(), top_k, num_shared_experts,
        num_experts)


def moe_tiny_fp16_shared_up(
    hidden_states: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    num_shared_experts: int,
) -> torch.Tensor:
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny shared FP16 path requires XPU")
    return _moe_int4.moe_tiny_fp16_shared_up(
        hidden_states.contiguous(), shared_gate_up_weight.contiguous(),
        num_shared_experts)


def moe_tiny_fp16_shared_finalize(
    hidden_states: torch.Tensor,
    shared_intermediates: torch.Tensor,
    routed_output: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    num_shared_experts: int,
) -> torch.Tensor:
    if hidden_states.device.type != "xpu":
        raise RuntimeError("tiny shared FP16 path requires XPU")
    return _moe_int4.moe_tiny_fp16_shared_finalize(
        hidden_states.contiguous(), shared_intermediates.contiguous(),
        routed_output.contiguous(), shared_down_weight.contiguous(),
        shared_expert_gate_weight.contiguous(), num_shared_experts)


def moe_forward_cutlass_nmajor_int4_full(
    x: torch.Tensor,
    logits: torch.Tensor,
    w13: torch.Tensor, w13_scales: torch.Tensor,
    w2: torch.Tensor, w2_scales: torch.Tensor,
    shared_gu_w: torch.Tensor,
    shared_d_w: torch.Tensor,
    shared_gate_w: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Full fused MoE decode: topk + routed INT4 + shared FP16, M>=1."""
    return _moe_int4.moe_forward_cutlass_nmajor_int4_full(
        x, logits, w13, w13_scales, w2, w2_scales,
        shared_gu_w, shared_d_w, shared_gate_w,
        top_k, num_shared_experts, n_routed_experts)
