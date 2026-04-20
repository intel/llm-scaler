"""Python wrappers for custom ESIMD kernels."""
import torch

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
