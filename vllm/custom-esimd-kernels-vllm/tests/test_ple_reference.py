"""CPU-only contract tests for the independent Qwen3.8 PLE oracle."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ple_reference import (  # noqa: E402
    embedding_local,
    grouped_norm,
    ngram_ids,
    projection_int4,
    score_gate,
    short_conv_decode,
    short_conv_prefill,
    short_conv_spec,
    staged_ple,
)


QWEN_EOS = 99


def test_ngram_ids_are_int64_and_do_not_cross_eos() -> None:
    input_ids = torch.tensor([7, 8, QWEN_EOS, 11, 13], dtype=torch.int64)
    # Two requests: the second request starts immediately after the first EOS.
    starts = torch.tensor([0, 3, 5], dtype=torch.int64)
    context = torch.tensor([[5, 6], [QWEN_EOS, QWEN_EOS]], dtype=torch.int64)
    multipliers = torch.tensor([3, 5, 7], dtype=torch.int64)
    vocab = torch.tensor([17, 19, 23, 29], dtype=torch.int64)
    offsets = torch.tensor([0, 17, 36, 59], dtype=torch.int64)

    result = ngram_ids(
        input_ids,
        starts,
        context,
        multipliers,
        vocab,
        offsets,
        eos_token_id=QWEN_EOS,
        heads_per_ngram=2,
    )

    expected = torch.empty((5, 4), dtype=torch.int64)
    # Request 0, token 0 uses the two context values as its predecessors.
    expected[0] = torch.tensor(
        [(7 * 3 ^ 6 * 5) % 17, (7 * 3 ^ 6 * 5) % 19 + 17,
         (7 * 3 ^ 6 * 5 ^ 5 * 7) % 23 + 36,
         (7 * 3 ^ 6 * 5 ^ 5 * 7) % 29 + 59]
    )
    # Request 0, token 1 has token 7 as its same-segment predecessor.
    expected[1] = torch.tensor(
        [(8 * 3 ^ 7 * 5) % 17, (8 * 3 ^ 7 * 5) % 19 + 17,
         (8 * 3 ^ 7 * 5 ^ 6 * 7) % 23 + 36,
         (8 * 3 ^ 7 * 5 ^ 6 * 7) % 29 + 59]
    )
    # The EOS token itself starts a new segment for its predecessor lookup.
    expected[2] = torch.tensor(
        [(99 * 3 ^ 99 * 5) % 17, (99 * 3 ^ 99 * 5) % 19 + 17,
         (99 * 3 ^ 99 * 5 ^ 99 * 7) % 23 + 36,
         (99 * 3 ^ 99 * 5 ^ 99 * 7) % 29 + 59]
    )
    # Request 1 has EOS context, so token 11 cannot use token 99 as a real prior
    # value; it uses EOS for both shifted positions.
    expected[3] = torch.tensor(
        [(11 * 3 ^ 99 * 5) % 17, (11 * 3 ^ 99 * 5) % 19 + 17,
         (11 * 3 ^ 99 * 5 ^ 99 * 7) % 23 + 36,
         (11 * 3 ^ 99 * 5 ^ 99 * 7) % 29 + 59]
    )
    expected[4] = torch.tensor(
        [(13 * 3 ^ 11 * 5) % 17, (13 * 3 ^ 11 * 5) % 19 + 17,
         (13 * 3 ^ 11 * 5 ^ 99 * 7) % 23 + 36,
         (13 * 3 ^ 11 * 5 ^ 99 * 7) % 29 + 59]
    )
    assert result.dtype == torch.int64
    assert torch.equal(result, expected)


def test_embedding_local_empty_shard_is_zero() -> None:
    ids = torch.tensor([[3, 4]], dtype=torch.int64)
    weight = torch.empty((0, 8), dtype=torch.float16)
    result = embedding_local(ids, weight, 10, 0)
    assert result.shape == (1, 16)
    assert torch.count_nonzero(result) == 0


def test_projection_int4_uses_low_high_nibble_and_group_scales() -> None:
    values = torch.arange(128, dtype=torch.float16).reshape(1, 128)
    packed = torch.zeros((2, 64), dtype=torch.uint8)
    packed[0, 0] = 0x8F  # even=-8, odd=+7
    packed[1, 0] = 0x12  # even=-6, odd=-7
    scales = torch.tensor([[2.0], [-1.5]], dtype=torch.float16)
    result = projection_int4(values, packed, scales)
    # All other packed nibbles are 0 => dequantized -8; include them in the
    # explicit FP32 oracle rather than relying on a special-case first byte.
    low = (packed.to(torch.int16) & 0xF).float() - 8
    high = ((packed.to(torch.int16) >> 4) & 0xF).float() - 8
    unpacked = torch.empty((2, 128), dtype=torch.float32)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    expected = (values.float() @ (unpacked * scales.float()).transpose(0, 1)).to(
        torch.float16
    )
    assert torch.equal(result, expected)


def test_grouped_norm_uses_fp32_variance_and_one_plus_weight() -> None:
    values = torch.tensor(
        [[1.0e4, -1.0e4, 1.0, -1.0]], dtype=torch.float16
    )
    weight = torch.tensor([0.25, -0.5, 0.0, 1.0], dtype=torch.float16)
    result = grouped_norm(values, weight, 1.0e-5, 2)
    values_f = values.float().reshape(1, 2, 2)
    variance = values_f.square().mean(-1, keepdim=True)
    expected = (
        values_f * torch.rsqrt(variance + 1.0e-5)
    ).reshape(1, 4) * (1.0 + weight.float())
    assert result.dtype == torch.float16
    assert torch.equal(result, expected.to(torch.float16))


def _conv_fixture(
    *, state_dim_first: bool, state_dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    width = 3
    state_len = 9
    slots = 3
    x = torch.tensor(
        [[0.5, -1.0, 2.0], [1.0, 2.0, -0.5], [-2.0, 0.25, 1.5]],
        dtype=torch.float16,
    )
    state_sd = torch.arange(slots * width * state_len, dtype=state_dtype)
    state_sd = state_sd.reshape(slots, width, state_len)
    state = state_sd if state_dim_first else state_sd.transpose(-1, -2).contiguous()
    weights = torch.tensor(
        [[0.25, -0.5, 0.75, -1.0]] * width, dtype=torch.float16
    )
    indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    initial = torch.tensor([True, True, False])
    return x, state, weights, indices, initial


@pytest.mark.parametrize("state_dim_first", [True, False])
def test_decode_preserves_null_row_and_supports_state_dtype(
    state_dim_first: bool,
) -> None:
    x, state, weights, indices, initial = _conv_fixture(
        state_dim_first=state_dim_first
    )
    before = state.clone()
    output, updated = short_conv_decode(
        x,
        state,
        weights,
        indices,
        initial,
        dilation=3,
        state_dim_first=state_dim_first,
    )
    assert output.shape == x.shape
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert torch.equal(updated[1], before[1])
    assert torch.equal(state, before)
    assert not torch.equal(updated[0], before[0])
    assert not torch.equal(updated[2], before[2])


def test_decode_false_initial_state_does_not_read_old_state() -> None:
    x, state, weights, indices, _ = _conv_fixture(state_dim_first=True)
    initial = torch.zeros(3, dtype=torch.bool)
    zero_state = torch.zeros_like(state)
    output_a, updated_a = short_conv_decode(
        x, state, weights, indices, initial, dilation=3
    )
    output_b, updated_b = short_conv_decode(
        x, zero_state, weights, indices, initial, dilation=3
    )
    assert torch.equal(output_a, output_b)
    # Both valid requests write the same result when initial history is disabled.
    assert torch.equal(updated_a[0], updated_b[0])
    assert torch.equal(updated_a[2], updated_b[2])


def test_prefill_ragged_order_and_null_request() -> None:
    width, state_len = 2, 9
    x = torch.tensor(
        [[1.0, 2.0], [2.0, 3.0], [-1.0, 0.5], [4.0, -2.0]],
        dtype=torch.float16,
    )
    state = torch.arange(3 * width * state_len, dtype=torch.float32).reshape(
        3, width, state_len
    )
    weights = torch.ones((width, 4), dtype=torch.float16)
    starts = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    indices = torch.tensor([0, -1, 2], dtype=torch.int32)
    initial = torch.tensor([True, True, False])
    before = state.clone()
    output, updated = short_conv_prefill(
        x, starts, state, weights, indices, initial, dilation=3
    )
    assert output.shape == x.shape
    assert torch.equal(updated[1], before[1])
    assert torch.equal(state, before)
    assert torch.equal(output[2:], output[2:])  # packed order remains stable
    assert torch.count_nonzero(output[:2]) > 0
    assert torch.count_nonzero(output[2:]) > 0


def test_spec_rolls_back_and_keeps_candidate_extension() -> None:
    width, state_len, spec_tokens = 2, 9, 3
    x = torch.tensor(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        dtype=torch.float16,
    )
    state = torch.arange(2 * width * (state_len + spec_tokens), dtype=torch.float32)
    state = state.reshape(2, width, state_len + spec_tokens)
    weights = torch.ones((width, 4), dtype=torch.float16)
    starts = torch.tensor([0, 2, 4], dtype=torch.int64)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    accepted = torch.tensor([2, 4], dtype=torch.int32)
    before = state.clone()
    output, updated = short_conv_spec(
        x,
        starts,
        state,
        weights,
        indices,
        accepted,
        num_spec_tokens=spec_tokens,
        dilation=3,
    )
    assert output.shape == x.shape
    assert torch.equal(state, before)
    # The state starts at accepted-1, then retains each request's candidate rows.
    assert not torch.equal(updated[0], before[0])
    assert not torch.equal(updated[1], before[1])
    assert torch.count_nonzero(output) > 0


def test_score_gate_zero_is_exact_half() -> None:
    key = torch.zeros((2, 4, 8), dtype=torch.float16)
    query = torch.zeros_like(key)
    result = score_gate(key, query, hidden_size=8)
    assert torch.equal(result, torch.full((2, 4, 1), 0.5, dtype=torch.float16))


def test_staged_ple_returns_all_intermediates() -> None:
    torch.manual_seed(7)
    tokens, hidden, groups = 2, 4, 2
    embedding = torch.randn(tokens, hidden, dtype=torch.float16)
    query = torch.randn(tokens, groups * hidden, dtype=torch.float16)
    key_weight = torch.randn(groups * hidden, hidden, dtype=torch.float16)
    value_weight = torch.randn(hidden, hidden, dtype=torch.float16)
    norm = torch.zeros(groups * hidden, dtype=torch.float16)
    state = torch.zeros(2, groups * hidden, 3, dtype=torch.float32)
    conv_weights = torch.ones(groups * hidden, 2, dtype=torch.float16)
    result = staged_ple(
        embedding=embedding,
        hidden_states=query,
        key_weight=key_weight,
        value_weight=value_weight,
        norm_key_weight=norm,
        norm_query_weight=norm,
        norm_conv_weight=torch.zeros(groups * hidden, dtype=torch.float16),
        conv_state=state,
        conv_weights=conv_weights,
        state_indices=torch.tensor([0, 1], dtype=torch.int64),
        has_initial_state=torch.tensor([False, False]),
        mode="decode",
        eps=1.0e-5,
        group_size=hidden,
        dilation=3,
    )
    assert set(result) == {
        "key", "value", "query", "key_norm", "query_norm", "gate",
        "gated_value", "normalized", "conv_output", "output", "final_state",
    }
    assert result["output"].shape == (tokens, groups * hidden)
    assert result["gate"].shape == (tokens, groups, 1)
    assert math.isfinite(float(result["output"].float().abs().max()))
    expected = result["gated_value"].flatten(-2) + result["conv_output"]
    assert torch.equal(result["output"], expected)


def test_prefill_valid_empty_request_preserves_existing_state() -> None:
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    state = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    before = state.clone()
    weights = torch.ones((2, 2), dtype=torch.float16)
    starts = torch.tensor([0, 0, 1], dtype=torch.int32)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    initial = torch.tensor([False, True])
    output, updated = short_conv_prefill(
        x, starts, state, weights, indices, initial, dilation=3
    )
    assert torch.equal(updated[0], before[0])
    assert not torch.equal(updated[1], before[1])
    assert torch.count_nonzero(output) > 0


def test_spec_valid_empty_request_is_a_state_noop() -> None:
    x = torch.tensor([[1.0]], dtype=torch.float16)
    state = torch.arange(2 * 1 * 5, dtype=torch.float32).reshape(2, 1, 5)
    before = state.clone()
    weights = torch.ones((1, 2), dtype=torch.float16)
    starts = torch.tensor([0, 0, 1], dtype=torch.int32)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    accepted = torch.tensor([1, 1], dtype=torch.int32)

    output, updated = short_conv_spec(
        x, starts, state, weights, indices, accepted,
        dilation=3, num_spec_tokens=2,
    )

    assert torch.equal(updated[0], before[0])
    assert not torch.equal(updated[1], before[1])
    # The only packed token belongs to the non-empty second request.
    assert torch.count_nonzero(output) > 0


def test_spec_rejects_zero_accepted_count() -> None:
    x = torch.ones((1, 1), dtype=torch.float16)
    state = torch.zeros((1, 1, 5), dtype=torch.float32)
    weights = torch.ones((1, 2), dtype=torch.float16)
    with pytest.raises(ValueError, match="supported range"):
        short_conv_spec(
            x,
            torch.tensor([0, 1], dtype=torch.int32),
            state,
            weights,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            dilation=3,
            num_spec_tokens=2,
        )


@pytest.mark.parametrize("state_dim_first", [True, False])
def test_spec_writes_post_consumption_window(
    state_dim_first: bool,
) -> None:
    width, length, spec_tokens = 1, 3, 2
    state_sd = torch.tensor([[[10.0, 11.0, 12.0, 13.0, 14.0]]])
    state_sd = torch.cat(
        (state_sd, torch.tensor([[[20.0, 21.0, 22.0, 23.0, 24.0]]])), dim=0
    )
    state = state_sd if state_dim_first else state_sd.transpose(-1, -2).contiguous()
    x = torch.tensor([[30.0], [31.0]], dtype=torch.float16)
    weights = torch.ones((width, 2), dtype=torch.float16)
    starts = torch.tensor([0, 2], dtype=torch.int64)
    indices = torch.tensor([0], dtype=torch.int64)
    accepted = torch.tensor([2], dtype=torch.int32)
    _, updated = short_conv_spec(
        x, starts, state, weights, indices, accepted,
        dilation=3, num_spec_tokens=spec_tokens,
        state_dim_first=state_dim_first,
    )
    updated_sd = updated if state_dim_first else updated.transpose(-1, -2)
    # rollback=accepted-1=1: [11,12,13] + [30,31], then consume one entry.
    assert torch.equal(
        updated_sd[0, 0, :4], torch.tensor([12.0, 13.0, 30.0, 31.0])
    )
    assert torch.equal(updated_sd[0, 0, 4:], torch.tensor([14.0]))


@pytest.mark.parametrize("bad_index", [-2, 3])
def test_reference_rejects_non_null_out_of_range_state_index(bad_index: int) -> None:
    x, state, weights, _, initial = _conv_fixture(state_dim_first=True)
    indices = torch.tensor([0, bad_index, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="out-of-range"):
        short_conv_decode(
            x, state, weights, indices, initial, dilation=3
        )


def test_reference_rejects_spec_query_longer_than_capacity() -> None:
    x = torch.ones((4, 1), dtype=torch.float16)
    state = torch.zeros((1, 1, 6), dtype=torch.float32)
    weights = torch.ones((1, 2), dtype=torch.float16)
    with pytest.raises(ValueError, match="exceeds"):
        short_conv_spec(
            x,
            torch.tensor([0, 4], dtype=torch.int32),
            state,
            weights,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            dilation=3,
            num_spec_tokens=2,
        )
