"""Numerical and state-contract tests for speculative sequential GDN."""

import pytest
import torch

if not torch.xpu.is_available():
    pytest.skip("requires an Intel XPU", allow_module_level=True)

import custom_esimd_kernels_vllm as esimd
import vllm_xpu_kernels._xpu_C  # noqa: F401


def _seq_to_interleaved_qkvz(qkvz_seq, k_heads, v_heads, head_dim):
    """Match Qwen's GQA layout accepted by the native GDN kernel."""
    heads_per_group = v_heads // k_heads
    q_base = 0
    k_base = k_heads * head_dim
    v_base = 2 * k_heads * head_dim
    z_base = v_base + v_heads * head_dim
    parts = []
    for head in range(k_heads):
        parts.append(qkvz_seq[:, q_base + head * head_dim:q_base + (head + 1) * head_dim])
        parts.append(qkvz_seq[:, k_base + head * head_dim:k_base + (head + 1) * head_dim])
        for lane in range(heads_per_group):
            v_head = head * heads_per_group + lane
            parts.append(qkvz_seq[:, v_base + v_head * head_dim:v_base + (v_head + 1) * head_dim])
        for lane in range(heads_per_group):
            v_head = head * heads_per_group + lane
            parts.append(qkvz_seq[:, z_base + v_head * head_dim:z_base + (v_head + 1) * head_dim])
    return torch.cat(parts, dim=1)


def _seq_to_interleaved_ba(ba_seq, k_heads, v_heads):
    heads_per_group = v_heads // k_heads
    parts = []
    for head in range(k_heads):
        start = head * heads_per_group
        parts.append(ba_seq[:, start:start + heads_per_group])
        parts.append(ba_seq[:, v_heads + start:v_heads + start + heads_per_group])
    return torch.cat(parts, dim=1)


def test_spec_gdn_does_not_mutate_the_null_rollback_slot():
    """State index zero is the vLLM null block, not a writable checkpoint.

    The native speculative FLA kernel skips an invalid checkpoint but continues
    the current sequence with its register-resident state. Writing slot zero
    makes its shared sentinel state visible to a later padded sequence.
    """

    torch.manual_seed(708)
    tokens, k_heads, v_heads, head_dim = 4, 8, 24, 128
    conv_dim = 2 * k_heads * head_dim + v_heads * head_dim
    slots, conv_state_len = 5, tokens + 2
    device = "xpu"

    qkvz = torch.zeros(
        tokens, conv_dim + v_heads * head_dim, dtype=torch.float16, device=device
    )
    conv_state = torch.zeros(
        slots, conv_state_len, conv_dim, dtype=torch.float16, device=device
    )
    ssm_state = torch.zeros(
        slots, v_heads, head_dim, head_dim, dtype=torch.float16, device=device
    )
    # Use distinct sentinels so a store to either null cache is observable.
    conv_state[0].fill_(3)
    ssm_state[0].fill_(5)
    null_conv_before = conv_state[0].clone()
    null_ssm_before = ssm_state[0].clone()

    esimd.esimd_gdn_conv_fused_seq_spec(
        qkvz,
        conv_state,
        torch.zeros(conv_dim, 4, dtype=torch.float16, device=device),
        torch.zeros(conv_dim, dtype=torch.float16, device=device),
        torch.tensor([[1, 0, 3, 4]], dtype=torch.int32, device=device),
        torch.zeros(v_heads, dtype=torch.float16, device=device),
        torch.zeros(v_heads, dtype=torch.float16, device=device),
        torch.zeros(tokens, 2 * v_heads, dtype=torch.float16, device=device),
        ssm_state,
        torch.empty(tokens, v_heads, head_dim, dtype=torch.float16, device=device),
        torch.empty(tokens, v_heads, head_dim, dtype=torch.float16, device=device),
        torch.arange(tokens, dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        1,
        tokens,
        k_heads,
        v_heads,
        head_dim,
        head_dim,
        head_dim**-0.5,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(conv_state[0], null_conv_before, rtol=0, atol=0)
    torch.testing.assert_close(ssm_state[0], null_ssm_before, rtol=0, atol=0)


def test_spec_gdn_nonpacked_conv_does_not_mutate_null_rollback_slot():
    """The legacy three-row conv cache must also treat state ID zero as null."""

    tokens, k_heads, v_heads, head_dim = 4, 8, 24, 128
    conv_dim = 2 * k_heads * head_dim + v_heads * head_dim
    slots, conv_state_len = 5, 3
    device = "xpu"

    qkvz = torch.zeros(
        tokens, conv_dim + v_heads * head_dim, dtype=torch.float16, device=device
    )
    conv_state = torch.zeros(
        slots, conv_state_len, conv_dim, dtype=torch.float16, device=device
    )
    ssm_state = torch.zeros(
        slots, v_heads, head_dim, head_dim, dtype=torch.float16, device=device
    )
    conv_state[0].fill_(7)
    null_conv_before = conv_state[0].clone()

    esimd.esimd_gdn_conv_fused_seq_spec(
        qkvz,
        conv_state,
        torch.zeros(conv_dim, 4, dtype=torch.float16, device=device),
        torch.zeros(conv_dim, dtype=torch.float16, device=device),
        torch.tensor([[1, 0, 3, 4]], dtype=torch.int32, device=device),
        torch.zeros(v_heads, dtype=torch.float16, device=device),
        torch.zeros(v_heads, dtype=torch.float16, device=device),
        torch.zeros(tokens, 2 * v_heads, dtype=torch.float16, device=device),
        ssm_state,
        torch.empty(tokens, v_heads, head_dim, dtype=torch.float16, device=device),
        torch.empty(tokens, v_heads, head_dim, dtype=torch.float16, device=device),
        torch.arange(tokens, dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        1,
        tokens,
        k_heads,
        v_heads,
        head_dim,
        head_dim,
        head_dim**-0.5,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(conv_state[0], null_conv_before, rtol=0, atol=0)


@pytest.mark.parametrize("num_accepted", [1, 2, 3])
def test_spec_gdn_matches_native_qwen35_tp2(num_accepted):
    """Exercise TP2/MTP3 rollback offsets against the native XPU kernel."""

    torch.manual_seed(709)
    tokens, k_heads, v_heads, head_dim = 4, 8, 24, 128
    conv_dim = 2 * k_heads * head_dim + v_heads * head_dim
    slots, conv_state_len = 6, tokens + 2
    device = "xpu"

    qkvz = torch.randn(
        tokens, conv_dim + v_heads * head_dim, dtype=torch.float16, device=device
    ) * 0.1
    ba = torch.randn(tokens, 2 * v_heads, dtype=torch.float16, device=device) * 0.1
    conv_weight = torch.randn(conv_dim, 4, dtype=torch.float16, device=device) * 0.1
    conv_bias = torch.randn(conv_dim, dtype=torch.float16, device=device) * 0.01
    a_log = torch.randn(v_heads, dtype=torch.float16, device=device) * 0.1
    dt_bias = torch.randn(v_heads, dtype=torch.float16, device=device) * 0.1
    state_indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device)
    accepted = torch.tensor([num_accepted], dtype=torch.int32, device=device)

    initial_conv = torch.randn(
        slots, conv_state_len, conv_dim, dtype=torch.float16, device=device
    ) * 0.01
    initial_ssm = torch.randn(
        slots, v_heads, head_dim, head_dim, dtype=torch.float16, device=device
    ) * 0.01
    conv_ref, ssm_ref = initial_conv.clone(), initial_ssm.clone()
    conv_esimd, ssm_esimd = initial_conv.clone(), initial_ssm.clone()
    out_ref = torch.empty(tokens, v_heads, head_dim, dtype=torch.float16, device=device)
    z_ref = torch.empty_like(out_ref)
    out_esimd = torch.empty_like(out_ref)
    z_esimd = torch.empty_like(out_ref)

    torch.ops._xpu_C.gdn_attention(
        out_ref,
        z_ref,
        _seq_to_interleaved_qkvz(qkvz, k_heads, v_heads, head_dim),
        _seq_to_interleaved_ba(ba, k_heads, v_heads),
        16,
        48,
        head_dim,
        head_dim,
        conv_state=conv_ref,
        ssm_state=ssm_ref,
        conv_weights=conv_weight,
        conv_bias=conv_bias,
        activation="silu",
        # The production adapter casts A_log to FP16 for ESIMD. Use that
        # identical value (promoted for the native op's FP32 interface).
        A_log=a_log.float(),
        dt_bias=dt_bias,
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        spec_token_indx=token_indices,
        spec_state_indices_tensor=state_indices,
        num_accepted_tokens=accepted,
        num_actual_tokens=tokens,
        tp_size=2,
        reorder_input=False,
    )
    esimd.esimd_gdn_conv_fused_seq_spec(
        qkvz,
        conv_esimd,
        conv_weight,
        conv_bias,
        state_indices,
        a_log,
        dt_bias,
        ba,
        ssm_esimd,
        out_esimd,
        z_esimd,
        token_indices,
        accepted,
        1,
        tokens,
        k_heads,
        v_heads,
        head_dim,
        head_dim,
        head_dim**-0.5,
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(out_esimd, out_ref, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(z_esimd, z_ref, rtol=0, atol=0)
    # v0.26 stores speculative convolution rollback checkpoints packed in the
    # initial request block: the carried history at init_col followed by all
    # draft inputs. init_col is the same accepted-token rollback selection
    # used by the native SSM recurrence.
    init_col = num_accepted - 1
    torch.testing.assert_close(
        conv_esimd[1, :2], initial_conv[1, init_col + 1:init_col + 3], rtol=0, atol=0
    )
    torch.testing.assert_close(conv_esimd[1, 2:], qkvz[:, :conv_dim], rtol=0, atol=0)
    # The reference and ESIMD implementations use different vector math
    # intrinsics, while both checkpoint the recurrence in FP16.
    torch.testing.assert_close(ssm_esimd, ssm_ref, rtol=1e-1, atol=2e-2)
