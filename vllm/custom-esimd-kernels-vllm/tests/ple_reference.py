"""Independent Torch oracle for the Qwen3.8 PLE kernel pipeline.

This module intentionally imports neither vLLM nor custom ESIMD operators.  The
reference is used to generate frozen intermediate/state outputs for standalone
PLE tests, so changes to a candidate kernel cannot silently change its oracle.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _segment_shift(
    sequence: torch.Tensor,
    index: int,
    shift: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Return a causal same-segment token, or EOS across a segment boundary."""
    source = index - shift
    if source < 0:
        return sequence.new_tensor(eos_token_id)
    current_segment = -1
    for position in range(index + 1):
        if int(sequence[position]) == eos_token_id:
            current_segment = position
    if source <= current_segment:
        return sequence.new_tensor(eos_token_id)
    return sequence[source]


def ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    *,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Generate request-relative N-gram IDs with exact int64 hash semantics."""
    input_ids = input_ids.reshape(-1).to(dtype=torch.int64)
    query_start_loc = query_start_loc.reshape(-1).to(dtype=torch.int64)
    ngram_context = ngram_context.to(dtype=torch.int64)
    layer_multipliers = layer_multipliers.reshape(-1).to(dtype=torch.int64)
    vocab_sizes = ngram_heads_vocab_sizes.reshape(-1).to(dtype=torch.int64)
    offsets = ngram_heads_offsets.reshape(-1).to(dtype=torch.int64)

    if query_start_loc.numel() < 1:
        raise ValueError("query_start_loc must not be empty")
    num_reqs = query_start_loc.numel() - 1
    ngram_size = layer_multipliers.numel()
    expected_heads = (ngram_size - 1) * heads_per_ngram
    if vocab_sizes.numel() != expected_heads or offsets.numel() != expected_heads:
        raise ValueError("N-gram metadata head count does not match multipliers")
    if ngram_context.shape != (num_reqs, ngram_size - 1):
        raise ValueError("ngram_context has an invalid request/history shape")

    blocks: list[torch.Tensor] = []
    for request in range(num_reqs):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        request_tokens = input_ids[start:end]
        sequence = torch.cat((ngram_context[request], request_tokens), dim=0)
        request_ids: list[torch.Tensor] = []
        for local_index in range(request_tokens.numel()):
            sequence_index = ngram_size - 1 + local_index
            mixed_values: list[torch.Tensor] = []
            for ngram in range(2, ngram_size + 1):
                mixed = _segment_shift(sequence, sequence_index, 0, eos_token_id)
                mixed = mixed * layer_multipliers[0]
                for shift in range(1, ngram):
                    mixed = torch.bitwise_xor(
                        mixed,
                        _segment_shift(
                            sequence, sequence_index, shift, eos_token_id
                        )
                        * layer_multipliers[shift],
                    )
                mixed_values.append(mixed.expand(heads_per_ngram))
            mixed = torch.cat(mixed_values)
            request_ids.append(torch.remainder(mixed, vocab_sizes) + offsets)
        if request_ids:
            blocks.append(torch.stack(request_ids, dim=0))
    if not blocks:
        return torch.empty(
            (0, expected_heads), dtype=torch.int64, device=input_ids.device
        )
    return torch.cat(blocks, dim=0)


def embedding_local(
    ngram_ids_tensor: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor | int,
    local_num_rows: torch.Tensor | int,
) -> torch.Tensor:
    """Gather a local shard, returning zero for IDs outside this shard."""
    ids = ngram_ids_tensor.to(dtype=torch.int64)
    start = torch.as_tensor(
        local_vocab_start, dtype=torch.int64, device=ids.device
    ).reshape(1, 1)
    rows = torch.as_tensor(
        local_num_rows, dtype=torch.int64, device=ids.device
    ).reshape(1, 1)
    valid = (ids >= start) & (ids < start + rows)
    if local_weight.size(0) == 0:
        return torch.zeros(
            (*ids.shape, local_weight.size(1)),
            dtype=local_weight.dtype,
            device=ids.device,
        ).flatten(-2)
    local_ids = torch.where(valid, ids - start, torch.zeros_like(ids))
    gathered = F.embedding(local_ids, local_weight)
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return gathered.flatten(-2)


def projection_int4(
    input: torch.Tensor,
    weight_esimd: torch.Tensor,
    scale_esimd: torch.Tensor,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    """Decode the confirmed offset-binary Q4_0 layout with FP32 accumulation."""
    if input.dim() != 2 or weight_esimd.dim() != 2 or scale_esimd.dim() != 2:
        raise ValueError("INT4 projection inputs must be rank 2")
    rows, k_packed = weight_esimd.shape
    k = k_packed * 2
    if input.shape[1] != k or k % group_size or scale_esimd.shape != (rows, k // group_size):
        raise ValueError("INT4 projection layout is invalid")
    if weight_esimd.dtype != torch.uint8 or scale_esimd.dtype != torch.float16:
        raise ValueError("INT4 projection dtypes are invalid")
    low = (weight_esimd.to(torch.int16) & 0x0F) - 8
    high = ((weight_esimd.to(torch.int16) >> 4) & 0x0F) - 8
    unpacked = torch.empty(
        (rows, k), dtype=torch.float32, device=input.device
    )
    unpacked[:, 0::2] = low.float()
    unpacked[:, 1::2] = high.float()
    scales = scale_esimd.float().repeat_interleave(group_size, dim=1)
    dequant = unpacked * scales
    return (input.float() @ dequant.transpose(0, 1)).to(input.dtype)


def embedding_assemble(local_partials: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Reference TP assembly; callers may provide [R,T,E] or a list of [T,E]."""
    if isinstance(local_partials, (list, tuple)):
        return torch.stack(list(local_partials), dim=0).sum(dim=0)
    if local_partials.dim() != 3:
        raise ValueError("stacked local partials must have shape [R,T,E]")
    return local_partials.sum(dim=0)


def grouped_norm(
    values: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """PLE grouped norm: FP32 variance and ``1 + weight`` scale."""
    input_dtype = values.dtype
    values_f = values.float()
    width = values_f.shape[-1]
    if width % group_size:
        raise ValueError("grouped norm width is not divisible by group_size")
    grouped = values_f.reshape(*values_f.shape[:-1], width // group_size, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return (normalized * (1.0 + weight.float())).to(input_dtype)


def score_gate(
    key: torch.Tensor,
    query: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """Compute the Qwen3.8 signed-square-root gate in FP32."""
    score = (key.float() * query.float()).sum(dim=-1, keepdim=True)
    score = score / math.sqrt(hidden_size)
    return torch.sigmoid(
        score.sign() * score.abs().clamp_min(1e-6).sqrt()
    ).to(key.dtype)


def gated_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply gate to value with the model's explicit C-way broadcast."""
    return gate * value.unsqueeze(-2)


def _conv_step(
    history: torch.Tensor,
    conv_weights: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Depthwise causal convolution for one ``[D,L+1]`` history."""
    kernel_size = conv_weights.shape[-1]
    expected_len = (kernel_size - 1) * dilation
    if history.shape[-1] != expected_len + 1:
        raise ValueError("history width does not match dilated convolution")
    result = F.conv1d(
        history.unsqueeze(0),
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.shape[0],
        dilation=dilation,
    )
    return F.silu(result.squeeze(0).squeeze(-1))


_MAX_CONV_STATE = 128
_MAX_SPEC_TOKENS = 128


def _checked_state_length(conv_weights: torch.Tensor, dilation: int) -> int:
    if conv_weights.dim() != 2 or conv_weights.size(1) < 1:
        raise ValueError("conv_weights must have shape [channels, kernel_size]")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    kernel_span = int(conv_weights.size(1)) - 1
    if kernel_span > _MAX_CONV_STATE // dilation:
        raise ValueError("dilated convolution state length exceeds limit")
    return kernel_span * dilation


def _validate_state_contract(
    x: torch.Tensor,
    state: torch.Tensor,
    conv_weights: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    *,
    required_width: int,
    state_dim_first: bool,
    null_block_id: int,
) -> torch.Tensor:
    if x.dim() != 2 or x.size(1) <= 0:
        raise ValueError("input must have shape [tokens, channels]")
    if conv_weights.dim() != 2 or conv_weights.size(0) != x.size(1):
        raise ValueError("conv_weights channel shape does not match input")
    if state.dim() != 3 or state.size(0) <= 0:
        raise ValueError("conv_state must have rank 3 and positive slots")
    if state.dtype not in (torch.float16, torch.float32):
        raise ValueError("conv_state must have dtype float16 or float32")
    if state_dim_first:
        channels, capacity = state.size(1), state.size(2)
    else:
        capacity, channels = state.size(1), state.size(2)
    if channels != x.size(1) or capacity < required_width:
        raise ValueError("conv_state shape/layout is incompatible with input")
    indices = indices.reshape(-1).to(dtype=torch.int64)
    if has_initial_state is not None and has_initial_state.numel() != indices.numel():
        raise ValueError("initial-state metadata length does not match indices")
    if null_block_id >= 0 and null_block_id < state.size(0):
        raise ValueError("null_block_id must not identify a real state slot")
    seen: set[int] = set()
    for index in indices.tolist():
        index = int(index)
        if index == null_block_id:
            continue
        if index < 0 or index >= state.size(0):
            raise ValueError("state_indices contains an out-of-range non-null slot")
        if index in seen:
            raise ValueError("state_indices contains duplicate valid state slot")
        seen.add(index)
    return indices


def _validate_offsets(
    query_start_loc: torch.Tensor,
    requests: int,
    tokens: int,
) -> torch.Tensor:
    starts = query_start_loc.reshape(-1).to(dtype=torch.int64)
    if starts.numel() != requests + 1:
        raise ValueError("query_start_loc must have one entry per request plus one")
    values = starts.tolist()
    if not values or values[0] != 0 or values[-1] != tokens:
        raise ValueError("query_start_loc must start at zero and end at token count")
    previous = 0
    for value in values:
        value = int(value)
        if value < previous or value < 0 or value > tokens:
            raise ValueError("query_start_loc must be monotonic and in range")
        previous = value
    return starts


def _state_rows(
    state: torch.Tensor,
    indices: torch.Tensor,
    *,
    state_dim_first: bool,
) -> torch.Tensor:
    rows = state.index_select(0, indices.to(dtype=torch.int64))
    if state_dim_first:
        return rows
    return rows.transpose(-1, -2)


def short_conv_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    *,
    dilation: int,
    null_block_id: int = -1,
    state_dim_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ordinary decode, returning output and mutated state clone."""
    length = _checked_state_length(conv_weights, dilation)
    indices = _validate_state_contract(
        x, conv_state, conv_weights, state_indices, has_initial_state,
        required_width=length, state_dim_first=state_dim_first,
        null_block_id=null_block_id,
    )
    output = torch.zeros_like(x)
    updated = conv_state.clone()
    if has_initial_state is None:
        initial = indices != null_block_id
    else:
        initial = has_initial_state.reshape(-1).to(torch.bool)
    if initial.numel() != x.size(0):
        raise ValueError("initial-state metadata must have one entry per token")
    if indices.numel() != x.size(0):
        raise ValueError("state_indices must have one entry per token")
    for row in range(x.shape[0]):
        index = int(indices[row])
        valid = index != null_block_id and 0 <= index < conv_state.shape[0]
        if not valid:
            continue
        state_row = _state_rows(
            updated, torch.tensor([index], device=x.device),
            state_dim_first=state_dim_first,
        )[0]
        history_state = state_row[..., :length].to(x.dtype)
        if not bool(initial[row]):
            history_state = torch.zeros_like(history_state)
        history = torch.cat((history_state, x[row].unsqueeze(-1)), dim=-1)
        output[row] = _conv_step(history, conv_weights, dilation)
        next_state = history[..., -length:] if length else history[..., :0]
        if state_dim_first:
            updated[index, ..., :length] = next_state.to(updated.dtype)
        else:
            updated[index, :length, ...] = next_state.transpose(-1, -2).to(
                updated.dtype
            )
    return output, updated


def short_conv_prefill(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    dilation: int,
    null_block_id: int = -1,
    state_dim_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ragged prefill without device-side max/host synchronization."""
    length = _checked_state_length(conv_weights, dilation)
    indices = _validate_state_contract(
        x, conv_state, conv_weights, state_indices, has_initial_state,
        required_width=length, state_dim_first=state_dim_first,
        null_block_id=null_block_id,
    )
    starts = _validate_offsets(
        query_start_loc, indices.numel(), x.size(0)
    )
    if has_initial_state.numel() != indices.numel():
        raise ValueError("initial-state metadata length does not match requests")
    output = torch.zeros_like(x)
    updated = conv_state.clone()
    for request in range(starts.numel() - 1):
        begin, end = int(starts[request]), int(starts[request + 1])
        index = int(indices[request])
        valid = index != null_block_id
        if not valid:
            continue
        # A valid empty request is a no-op: it must not clear an existing state.
        if begin == end:
            continue
        state_row = _state_rows(
            updated, torch.tensor([index], device=x.device),
            state_dim_first=state_dim_first,
        )[0]
        history_state = state_row[..., :length].to(x.dtype)
        if not bool(has_initial_state[request]):
            history_state = torch.zeros_like(history_state)
        history = history_state
        for token in range(begin, end):
            history = torch.cat((history, x[token].unsqueeze(-1)), dim=-1)
            output[token] = _conv_step(history[..., -length - 1 :], conv_weights, dilation)
            if length:
                history = history[..., -length:]
        if length:
            if state_dim_first:
                updated[index, ..., :length] = history.to(updated.dtype)
            else:
                updated[index, :length, ...] = history.transpose(-1, -2).to(
                    updated.dtype
                )
    return output, updated


def short_conv_spec(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    dilation: int,
    num_spec_tokens: int,
    null_block_id: int = -1,
    state_dim_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference speculative conv with accepted-token rollback and extension."""
    length = _checked_state_length(conv_weights, dilation)
    if num_spec_tokens < 0 or num_spec_tokens > _MAX_SPEC_TOKENS:
        raise ValueError("num_spec_tokens is outside the supported range")
    capacity = length + int(num_spec_tokens)
    indices = _validate_state_contract(
        x, conv_state, conv_weights, state_indices, None,
        required_width=capacity, state_dim_first=state_dim_first,
        null_block_id=null_block_id,
    )
    starts = _validate_offsets(
        query_start_loc, indices.numel(), x.size(0)
    )
    accepted = num_accepted_tokens.reshape(-1).to(dtype=torch.int64)
    if accepted.numel() != indices.numel():
        raise ValueError("num_accepted_tokens length does not match requests")
    if bool(torch.any(accepted < 1)) or bool(torch.any(accepted > num_spec_tokens + 1)):
        raise ValueError("num_accepted_tokens is outside the supported range [1, num_spec_tokens + 1]")
    output = torch.zeros_like(x)
    updated = conv_state.clone()
    for request in range(starts.numel() - 1):
        begin, end = int(starts[request]), int(starts[request + 1])
        if end - begin > num_spec_tokens + 1:
            raise ValueError("spec query length exceeds num_spec_tokens + 1")
        index = int(indices[request])
        valid = index != null_block_id
        if not valid:
            continue
        # A valid empty request has no candidate to evaluate or consume.
        if begin == end:
            continue
        state_row = _state_rows(
            updated, torch.tensor([index], device=x.device),
            state_dim_first=state_dim_first,
        )[0]
        rollback = max(0, min(int(accepted[request]) - 1, num_spec_tokens))
        history = state_row[..., rollback : rollback + length].to(x.dtype)
        candidates: list[torch.Tensor] = []
        for token in range(begin, end):
            history = torch.cat((history, x[token].unsqueeze(-1)), dim=-1)
            output[token] = _conv_step(history[..., -length - 1 :], conv_weights, dilation)
            candidates.append(x[token].unsqueeze(-1))
            if length:
                history = history[..., -length:]
        extended = torch.cat(
            (state_row[..., rollback : rollback + length], *candidates), dim=-1
        )
        # The target forward consumes one token from the rollback window before
        # writing the candidate extension back.  Preserve the post-consumption
        # window rather than reintroducing the oldest history element.
        keep = min(capacity, max(0, length + len(candidates) - 1))
        if length and keep:
            shifted = extended[..., 1 : keep + 1]
            if state_dim_first:
                updated[index, ..., :keep] = shifted.to(updated.dtype)
            else:
                updated[index, :keep, ...] = shifted.transpose(-1, -2).to(
                    updated.dtype
                )
    return output, updated


def short_conv_mixed_three_way(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_token_indices: torch.Tensor,
    decode_token_indices: torch.Tensor,
    prefill_token_indices: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    decode_state_indices: torch.Tensor,
    decode_has_initial_state: torch.Tensor,
    prefill_query_start_loc: torch.Tensor,
    prefill_state_indices: torch.Tensor,
    prefill_has_initial_state: torch.Tensor,
    *,
    dilation: int,
    num_spec_tokens: int,
    null_block_id: int = -1,
    state_dim_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent three-way reference for spec/decode/prefill composition.

    The branch helpers above are the mathematical oracle; this function only
    adds the explicit packed-token permutation and cross-branch state ownership
    contract.  It intentionally never imports or calls the candidate wrapper.
    """
    if x.dim() != 2:
        raise ValueError("input must have shape [tokens, channels]")
    branches = (
        ("spec", spec_token_indices),
        ("decode", decode_token_indices),
        ("prefill", prefill_token_indices),
    )
    branch_values: dict[str, list[int]] = {}
    for name, tensor in branches:
        if tensor.dim() != 1:
            raise ValueError(f"{name}_token_indices must be one-dimensional")
        branch_values[name] = [int(value) for value in tensor.tolist()]
    all_tokens = (
        branch_values["spec"]
        + branch_values["decode"]
        + branch_values["prefill"]
    )
    if sorted(all_tokens) != list(range(x.size(0))):
        raise ValueError("mixed token indices must be a disjoint permutation")

    state_tensors = (
        ("spec", spec_state_indices),
        ("decode", decode_state_indices),
        ("prefill", prefill_state_indices),
    )
    state_values: dict[str, list[int]] = {}
    seen: set[int] = set()
    if null_block_id >= 0 and null_block_id < conv_state.size(0):
        raise ValueError("null_block_id must not identify a real state slot")
    for name, tensor in state_tensors:
        if tensor.dim() != 1:
            raise ValueError(f"{name}_state_indices must be one-dimensional")
        values = [int(value) for value in tensor.tolist()]
        state_values[name] = values
        for index in values:
            if index == null_block_id:
                continue
            if index < 0 or index >= conv_state.size(0):
                raise ValueError("mixed state metadata contains an invalid slot")
            if index in seen:
                raise ValueError("mixed state metadata contains duplicate slots")
            seen.add(index)

    spec_x = x.index_select(
        0, spec_token_indices.to(dtype=torch.int64, device=x.device)
    )
    decode_x = x.index_select(
        0, decode_token_indices.to(dtype=torch.int64, device=x.device)
    )
    prefill_x = x.index_select(
        0, prefill_token_indices.to(dtype=torch.int64, device=x.device)
    )

    # The reference executes exactly the production order on a new state after
    # every branch, so no branch can observe a partially committed caller state.
    spec_output, state_after_spec = short_conv_spec(
        spec_x,
        spec_query_start_loc,
        conv_state,
        conv_weights,
        spec_state_indices,
        num_accepted_tokens,
        dilation=dilation,
        num_spec_tokens=num_spec_tokens,
        null_block_id=null_block_id,
        state_dim_first=state_dim_first,
    )
    decode_output, state_after_decode = short_conv_decode(
        decode_x,
        state_after_spec,
        conv_weights,
        decode_state_indices,
        decode_has_initial_state,
        dilation=dilation,
        null_block_id=null_block_id,
        state_dim_first=state_dim_first,
    )
    prefill_output, final_state = short_conv_prefill(
        prefill_x,
        prefill_query_start_loc,
        state_after_decode,
        conv_weights,
        prefill_state_indices,
        prefill_has_initial_state,
        dilation=dilation,
        null_block_id=null_block_id,
        state_dim_first=state_dim_first,
    )

    output = torch.empty_like(x)
    output.index_copy_(
        0,
        spec_token_indices.to(dtype=torch.int64, device=x.device),
        spec_output,
    )
    output.index_copy_(
        0,
        decode_token_indices.to(dtype=torch.int64, device=x.device),
        decode_output,
    )
    output.index_copy_(
        0,
        prefill_token_indices.to(dtype=torch.int64, device=x.device),
        prefill_output,
    )
    return output, final_state


def staged_ple(
    *,
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    norm_key_weight: torch.Tensor,
    norm_query_weight: torch.Tensor,
    norm_conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    mode: str,
    eps: float,
    group_size: int,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    num_spec_tokens: int = 0,
    null_block_id: int = -1,
    state_dim_first: bool = True,
) -> dict[str, torch.Tensor]:
    """Run the complete mathematical PLE pipeline and return all stages."""
    key = embedding @ key_weight.transpose(-1, -2)
    value = embedding @ value_weight.transpose(-1, -2)
    token_count = hidden_states.shape[0]
    group_count = key.shape[1] // group_size
    key = key.reshape(token_count, group_count, group_size)
    query = hidden_states.reshape(token_count, group_count, group_size)
    key_norm = grouped_norm(
        key.flatten(-2), norm_key_weight, eps, group_size
    ).reshape_as(key)
    query_norm = grouped_norm(
        query.flatten(-2), norm_query_weight, eps, group_size
    ).reshape_as(query)
    gate = score_gate(key_norm, query_norm, group_size)
    gated = gated_value(gate, value)
    normalized = grouped_norm(
        gated.flatten(-2), norm_conv_weight, eps, group_size
    )

    if mode == "decode":
        conv, final_state = short_conv_decode(
            normalized, conv_state, conv_weights, state_indices,
            has_initial_state, dilation=dilation,
            null_block_id=null_block_id, state_dim_first=state_dim_first,
        )
    elif mode == "prefill":
        if query_start_loc is None or has_initial_state is None:
            raise ValueError("prefill requires query_start_loc and initial mask")
        conv, final_state = short_conv_prefill(
            normalized, query_start_loc, conv_state, conv_weights, state_indices,
            has_initial_state, dilation=dilation,
            null_block_id=null_block_id, state_dim_first=state_dim_first,
        )
    elif mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError("spec requires query offsets and accepted counts")
        conv, final_state = short_conv_spec(
            normalized, query_start_loc, conv_state, conv_weights, state_indices,
            num_accepted_tokens, dilation=dilation, num_spec_tokens=num_spec_tokens,
            null_block_id=null_block_id, state_dim_first=state_dim_first,
        )
    else:
        raise ValueError(f"unsupported staged mode: {mode}")

    result = gated.flatten(-2) + conv
    return {
        "key": key,
        "value": value,
        "query": query,
        "key_norm": key_norm,
        "query_norm": query_norm,
        "gate": gate,
        "gated_value": gated,
        "normalized": normalized,
        "conv_output": conv,
        "output": result,
        "final_state": final_state,
    }


def staged_ple_full(
    *,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor | int,
    local_num_rows: torch.Tensor | int,
    rank_local_partials: torch.Tensor,
    hidden_states: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    norm_key_weight: torch.Tensor,
    norm_query_weight: torch.Tensor,
    norm_conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    mode: str,
    eps: float,
    group_size: int,
    dilation: int,
    eos_token_id: int,
    heads_per_ngram: int,
    query_start_loc_conv: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    num_spec_tokens: int = 0,
    null_block_id: int = -1,
    state_dim_first: bool = True,
    projection_kind: str = "fp16",
) -> dict[str, torch.Tensor]:
    """Reference the complete K0-K10 pipeline, including the K2 boundary."""
    ids = ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        eos_token_id=eos_token_id,
        heads_per_ngram=heads_per_ngram,
    )
    local_partial = embedding_local(
        ids, local_weight, local_vocab_start, local_num_rows
    )
    if rank_local_partials.dim() != 3 or rank_local_partials.size(0) < 1:
        raise ValueError("rank_local_partials must have shape [tp, tokens, E]")
    if rank_local_partials.shape[1:] != local_partial.shape:
        raise ValueError("rank_local_partials shape does not match K1 output")
    partials_for_assembly = rank_local_partials.clone()
    partials_for_assembly[0].copy_(local_partial)
    assembled = embedding_assemble(partials_for_assembly)
    staged = staged_ple(
        embedding=assembled,
        hidden_states=hidden_states,
        key_weight=key_weight,
        value_weight=value_weight,
        norm_key_weight=norm_key_weight,
        norm_query_weight=norm_query_weight,
        norm_conv_weight=norm_conv_weight,
        conv_state=conv_state,
        conv_weights=conv_weights,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        mode=mode,
        eps=eps,
        group_size=group_size,
        dilation=dilation,
        query_start_loc=query_start_loc_conv,
        num_accepted_tokens=num_accepted_tokens,
        num_spec_tokens=num_spec_tokens,
        null_block_id=null_block_id,
        state_dim_first=state_dim_first,
    )
    return {
        "ngram_ids": ids,
        "local_partial": local_partial,
        "assembled_embedding": assembled,
        **staged,
    }


# A small alias used by fixture generators to make the reference API explicit.
reference_ple = staged_ple

__all__ = [
    "embedding_assemble",
    "embedding_local",
    "grouped_norm",
    "gated_value",
    "ngram_ids",
    "projection_int4",
    "reference_ple",
    "score_gate",
    "short_conv_decode",
    "short_conv_mixed_three_way",
    "short_conv_prefill",
    "short_conv_spec",
    "staged_ple",
    "staged_ple_full",
]
