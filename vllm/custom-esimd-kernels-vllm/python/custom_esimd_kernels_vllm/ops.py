"""Python wrappers for custom ESIMD kernels."""
import torch
import torch.nn.functional as F

_ops = torch.ops.custom_esimd_kernels_vllm


def _tensors_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Use the alias API available across supported PyTorch XPU builds."""
    method = getattr(left, "is_alias_of", None)
    if method is not None:
        return bool(method(right))
    return bool(torch._C._is_alias_of(left, right))


QWEN38_NGRAM_VOCAB_SIZES = (
    20000003, 20000023, 20000033, 20000047,
    20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147,
    20000153, 20000159, 20000161, 20000171,
)
QWEN38_NGRAM_OFFSETS = (
    0, 20000003, 40000026, 60000059,
    80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655,
    240000802, 260000955, 280001114, 300001275,
)

# The current BMG INT4 GEMM kernel has a fixed maximum of eight 8-row tiles.
# Keep larger packed batches correct by submitting bounded chunks rather than
# allowing the native kernel to silently leave rows unwritten.
_PLE_PROJECTION_MAX_GEMM_ROWS = 64


def esimd_qwen38_ngram_ids_decode(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
) -> torch.Tensor:
    """Generate the 16 Qwen3.8 decode N-gram IDs in one XPU launch.

    This fast path specializes ``QWEN38_NGRAM_VOCAB_SIZES`` and
    ``QWEN38_NGRAM_OFFSETS``. The model call site must verify those immutable
    metadata values once before dispatch and retain the Torch fallback when
    they do not match. Native checks cover dtype, device, contiguity and the
    frozen decode shapes without adding a device-to-host synchronization.
    """
    return _ops.esimd_qwen38_ngram_ids_decode(
        input_ids, ngram_context, layer_multipliers)


def esimd_qwen38_ngram_ids_decode_out(
    input_ids: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Preallocated-output variant for a guarded model decode hot path."""
    _ops.esimd_qwen38_ngram_ids_decode_out(
        input_ids, ngram_context, layer_multipliers, output)
    return output


def esimd_qwen38_ngram_embedding_gather(
    ngram_ids: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor,
    local_num_rows: torch.Tensor,
) -> torch.Tensor:
    """Gather 16 global IDs from one runtime FP16 local shard.

    The result is the local partial ``[1, 2560]`` before TP all-reduce.
    ``local_weight`` and both shard metadata tensors are runtime inputs; this
    op is not tied to the small correctness fixture or a particular TP rank.
    """
    return _ops.esimd_qwen38_ngram_embedding_gather(
        ngram_ids, local_weight, local_vocab_start, local_num_rows)


def esimd_qwen38_ngram_embedding_gather_out(
    ngram_ids: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor,
    local_num_rows: torch.Tensor,
    local_partial: torch.Tensor,
) -> torch.Tensor:
    """Preallocated-output variant for a model hot path."""
    _ops.esimd_qwen38_ngram_embedding_gather_out(
        ngram_ids, local_weight, local_vocab_start, local_num_rows,
        local_partial)
    return local_partial


# ---- Qwen3.8 PLE standalone primitives ----


def ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    output: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Generate general EOS-aware N-gram IDs into a caller-owned buffer."""
    _ops.ple_ngram_ids(
        input_ids, query_start_loc, ngram_context, layer_multipliers,
        ngram_heads_vocab_sizes, ngram_heads_offsets, output,
        eos_token_id, heads_per_ngram)
    return output


def ple_embedding_gather(
    ngram_ids: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor,
    local_num_rows: torch.Tensor,
    local_partial: torch.Tensor,
) -> torch.Tensor:
    """Gather local embedding contributions before the TP assembly boundary."""
    _ops.ple_embedding_gather(
        ngram_ids, local_weight, local_vocab_start, local_num_rows, local_partial)
    return local_partial


def ple_grouped_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """PLE grouped norm with FP32 variance and a caller-owned output."""
    _ops.ple_grouped_norm(input, weight, output, eps, group_size)
    return output


def ple_score_gate(
    key: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """Compute the signed-square-root gate into ``[tokens, hc_count]``."""
    _ops.ple_score_gate(key, query, output, hidden_size)
    return output


def ple_gated_value(
    gate: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Broadcast gate over value and write ``[tokens, hc_count, hidden]``."""
    _ops.ple_gated_value(gate, value, output, hc_count)
    return output


def ple_gated_value_grouped_norm(
    gate: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    raw_output: torch.Tensor,
    normalized_output: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write rounded gated values and their grouped norm in one launch."""
    _ops.ple_gated_value_grouped_norm(
        gate, value, weight, raw_output, normalized_output, eps
    )
    return raw_output, normalized_output


def ple_residual_add(
    gated_value_flat: torch.Tensor,
    conv_output: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Add the flattened gated value and short-conv output."""
    _ops.ple_residual_add(gated_value_flat, conv_output, output)
    return output


def ple_short_conv_decode(
    input: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    output: torch.Tensor,
    dilation: int,
    state_dim_first: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    """Run ordinary decode short-conv and mutate the caller-owned state."""
    _ops.ple_short_conv_decode(
        input, conv_state, conv_weights, state_indices, has_initial_state,
        output, dilation, state_dim_first, null_block_id)
    return output


def ple_short_conv_prefill(
    input: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    output: torch.Tensor,
    dilation: int,
    state_dim_first: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    """Run ragged prefill short-conv and mutate the per-request state."""
    _ops.ple_short_conv_prefill(
        input, query_start_loc, conv_state, conv_weights, state_indices,
        has_initial_state, output, dilation, state_dim_first, null_block_id)
    return output


def ple_short_conv_spec(
    input: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    output: torch.Tensor,
    num_spec_tokens: int,
    dilation: int,
    state_dim_first: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    """Run speculative rollback/extension short-conv and mutate state."""
    _ops.ple_short_conv_spec(
        input, query_start_loc, conv_state, conv_weights, state_indices,
        num_accepted_tokens, output, num_spec_tokens, dilation,
        state_dim_first, null_block_id)
    return output


def ple_gated_value_norm(
    gated_value: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """K8 grouped norm for ``[tokens, hc_count, hidden]`` gated values."""
    if gated_value.dim() != 3:
        raise ValueError("gated_value must have shape [tokens, hc_count, hidden]")
    flattened = gated_value.flatten(-2)
    if output.shape == gated_value.shape:
        if not output.is_contiguous():
            raise ValueError("three-dimensional output must be contiguous")
        output_flat = output.view_as(flattened)
    elif output.shape == flattened.shape:
        output_flat = output
    else:
        raise ValueError("output must be flattened or match gated_value shape")
    return ple_grouped_norm(
        flattened, weight, output_flat, eps, group_size
    )


def ple_embedding_assemble(
    local_partials: torch.Tensor,
    assembled: torch.Tensor,
) -> torch.Tensor:
    """Assemble TP-local embedding partials without owning a communicator.

    ``local_partials`` is ``[tp, tokens, embed_dim]``.  The caller controls
    the all-reduce boundary; this helper only performs the deterministic
    mathematical sum into a caller-owned output buffer.
    """
    if local_partials.dim() != 3:
        raise ValueError("local_partials must have shape [tp, tokens, embed_dim]")
    if assembled.shape != local_partials.shape[1:]:
        raise ValueError("assembled shape must match one local partial")
    if assembled.device != local_partials.device or assembled.dtype != local_partials.dtype:
        raise ValueError("assembled must share device and dtype with local_partials")
    if assembled.numel() == 0:
        return assembled
    assembled.copy_(local_partials.sum(dim=0))
    return assembled


def _validate_ple_projection_int4(
    input: torch.Tensor,
    weight_esimd: torch.Tensor,
    scale_esimd: torch.Tensor,
    output: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = {
        "input": input,
        "weight_esimd": weight_esimd,
        "scale_esimd": scale_esimd,
        "output": output,
    }
    if any(t.device.type != "xpu" for t in tensors.values()):
        raise ValueError("PLE INT4 projection tensors must be on XPU")
    if any(t.device != input.device for t in tensors.values()):
        raise ValueError("PLE INT4 projection tensors must share one XPU device")
    if any(not t.is_contiguous() for t in tensors.values()):
        raise ValueError("PLE INT4 projection tensors must be contiguous")
    if input.dtype != torch.float16 or output.dtype != torch.float16:
        raise ValueError("PLE INT4 projection input/output must be float16")
    if weight_esimd.dtype != torch.uint8 or scale_esimd.dtype != torch.float16:
        raise ValueError("PLE INT4 weight/scale dtypes must be uint8/float16")
    if input.ndim != 2 or weight_esimd.ndim != 2 or scale_esimd.ndim != 2:
        raise ValueError("PLE INT4 projection input/weight/scale must be rank 2")
    if output.ndim != 2:
        raise ValueError("PLE INT4 projection output must be rank 2")
    m, k = input.shape
    n = weight_esimd.size(0)
    # Empty token batches are valid no-ops; channel dimensions remain strict.
    if n <= 0 or k <= 0 or k % 128:
        raise ValueError("PLE INT4 projection requires positive K divisible by 128")
    if weight_esimd.size(1) != k // 2:
        raise ValueError("projection K does not match packed weight")
    if n % 16:
        raise ValueError("PLE INT4 projection N must be divisible by 16")
    if scale_esimd.shape != (n, k // 128):
        raise ValueError("projection scale shape must be [N, K/128]")
    if output.shape != (m, n):
        raise ValueError("projection output shape is invalid")
    if any(_tensors_alias(output, t) for t in (input, weight_esimd, scale_esimd)):
        raise ValueError("projection output must not alias an input")
    return m, n, k


def ple_projection_int4(
    input: torch.Tensor,
    weight_esimd: torch.Tensor,
    scale_esimd: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run a PLE INT4 projection using the confirmed packed-weight ABI.

    ``weight_esimd`` is ``[N, K/2]`` uint8 and ``scale_esimd`` is
    ``[N, K/128]`` fp16.  M=1 uses the fused GEMV implementation; M>1 uses
    the batch INT4 GEMM implementation.  No logical-transpose view is passed
    to either kernel.  Batches larger than 64 rows are split into bounded
    GEMM/GEMV submissions because the native GEMM supports at most 64 rows.
    """
    m, _, _ = _validate_ple_projection_int4(
        input, weight_esimd, scale_esimd, output
    )
    if m == 0:
        return output
    if m == 1:
        return esimd_gemv_int4(input, weight_esimd, scale_esimd, output)

    for row_start in range(0, m, _PLE_PROJECTION_MAX_GEMM_ROWS):
        row_end = min(row_start + _PLE_PROJECTION_MAX_GEMM_ROWS, m)
        input_chunk = input[row_start:row_end]
        output_chunk = output[row_start:row_end]
        if row_end - row_start == 1:
            esimd_gemv_int4(
                input_chunk, weight_esimd, scale_esimd, output_chunk
            )
        else:
            esimd_gemm_int4_pgrp(
                input_chunk, weight_esimd, scale_esimd, output_chunk
            )
    return output


def ple_projection_fp16(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fallback/diagnostic FP16 PLE projection with row-major weights."""
    tensors = (input, weight, output)
    if any(t.device.type != "xpu" for t in tensors):
        raise ValueError("PLE FP16 projection tensors must be on XPU")
    if any(t.device != input.device for t in tensors):
        raise ValueError("PLE FP16 projection tensors must share one XPU device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("PLE FP16 projection tensors must be contiguous")
    if any(t.dtype != torch.float16 for t in tensors):
        raise ValueError("PLE FP16 projection tensors must be float16")
    if any(t.dim() != 2 for t in tensors):
        raise ValueError("PLE FP16 projection tensors must be rank 2")
    if input.size(1) <= 0 or weight.size(0) <= 0:
        raise ValueError("PLE FP16 projection dimensions must be positive")
    if input.size(1) != weight.size(1):
        raise ValueError("FP16 projection K mismatch")
    if output.shape != (input.size(0), weight.size(0)):
        raise ValueError("invalid FP16 projection shape")
    if any(_tensors_alias(output, t) for t in (input, weight)):
        raise ValueError("FP16 projection output must not alias an input")
    if input.size(0) == 0:
        return output
    return esimd_gemv_fp16(input, weight, output)


def ple_staged(
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    key_weight: torch.Tensor,
    key_scale: torch.Tensor | None,
    value_weight: torch.Tensor,
    value_scale: torch.Tensor | None,
    norm_key_weight: torch.Tensor,
    norm_query_weight: torch.Tensor,
    norm_conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    mode: str,
    eps: float,
    group_size: int,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    num_spec_tokens: int = 0,
    state_dim_first: bool = True,
    null_block_id: int = -1,
    projection_kind: str = "int4",
) -> dict[str, torch.Tensor]:
    """Run the caller-owned PLE primitive pipeline without vLLM integration.

    The returned dictionary intentionally exposes each intermediate so the
    standalone harness can identify the first divergent stage.  ``embedding``
    must already be the assembled TP result; K2's communicator boundary stays
    outside this function.
    """
    if embedding.dim() != 2 or hidden_states.dim() != 2:
        raise ValueError("embedding and hidden_states must be rank 2")
    if embedding.size(0) != hidden_states.size(0):
        raise ValueError("embedding and hidden_states token counts differ")
    tokens = hidden_states.size(0)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if hidden_states.size(1) <= 0 or hidden_states.size(1) % group_size:
        raise ValueError("hidden width must be positive and divisible by group_size")
    hc_count = hidden_states.size(1) // group_size
    key_linear = torch.empty(
        (tokens, hc_count * group_size),
        dtype=embedding.dtype,
        device=embedding.device,
    )
    value = torch.empty(
        (tokens, group_size), dtype=embedding.dtype, device=embedding.device
    )
    if projection_kind == "int4":
        if key_scale is None or value_scale is None:
            raise ValueError("INT4 projections require both scale tensors")
        ple_projection_int4(embedding, key_weight, key_scale, key_linear)
        ple_projection_int4(embedding, value_weight, value_scale, value)
    elif projection_kind == "fp16":
        ple_projection_fp16(embedding, key_weight, key_linear)
        ple_projection_fp16(embedding, value_weight, value)
    else:
        raise ValueError("projection_kind must be 'int4' or 'fp16'")

    key = key_linear.view(tokens, hc_count, group_size)
    query = hidden_states.view(tokens, hc_count, group_size)
    key_norm_flat = torch.empty_like(key_linear)
    query_norm_flat = torch.empty_like(hidden_states)
    ple_grouped_norm(
        key_linear, norm_key_weight, key_norm_flat, eps, group_size
    )
    ple_grouped_norm(
        hidden_states, norm_query_weight, query_norm_flat, eps, group_size
    )
    key_norm = key_norm_flat.view_as(key)
    query_norm = query_norm_flat.view_as(query)
    gate = torch.empty(
        (tokens, hc_count, 1), dtype=embedding.dtype, device=embedding.device
    )
    ple_score_gate(key_norm, query_norm, gate, group_size)
    gated = torch.empty(
        (tokens, hc_count, group_size),
        dtype=embedding.dtype,
        device=embedding.device,
    )
    ple_gated_value(gate, value, gated, hc_count)
    normalized = torch.empty_like(gated.reshape(tokens, -1))
    ple_gated_value_norm(
        gated, norm_conv_weight, normalized, eps, group_size
    )
    conv_output = torch.empty_like(normalized)
    if mode == "decode":
        ple_short_conv_decode(
            normalized, conv_state, conv_weights, state_indices,
            has_initial_state, conv_output, dilation, state_dim_first,
            null_block_id,
        )
    elif mode == "prefill":
        if query_start_loc is None:
            raise ValueError("prefill requires query_start_loc")
        ple_short_conv_prefill(
            normalized, query_start_loc, conv_state, conv_weights,
            state_indices, has_initial_state, conv_output, dilation,
            state_dim_first, null_block_id,
        )
    elif mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError("spec requires query_start_loc and accepted tokens")
        ple_short_conv_spec(
            normalized, query_start_loc, conv_state, conv_weights,
            state_indices, num_accepted_tokens, conv_output, num_spec_tokens,
            dilation, state_dim_first, null_block_id,
        )
    else:
        raise ValueError("mode must be decode, prefill, or spec")
    output = torch.empty_like(normalized)
    ple_residual_add(gated.flatten(-2), conv_output, output)
    return {
        "key": key,
        "value": value,
        "query": query,
        "key_norm": key_norm,
        "query_norm": query_norm,
        "gate": gate,
        "gated_value": gated,
        "normalized": normalized,
        "conv_output": conv_output,
        "output": output,
        # Return a snapshot: stateful primitives mutate the caller-owned state,
        # but staged diagnostics need a stable post-call artifact.
        "final_state": conv_state.clone(),
    }


def ple_staged_full(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    local_weight: torch.Tensor,
    local_vocab_start: torch.Tensor,
    local_num_rows: torch.Tensor,
    rank_local_partials: torch.Tensor,
    assembled_embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    key_weight: torch.Tensor,
    key_scale: torch.Tensor | None,
    value_weight: torch.Tensor,
    value_scale: torch.Tensor | None,
    norm_key_weight: torch.Tensor,
    norm_query_weight: torch.Tensor,
    norm_conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    mode: str,
    eps: float,
    group_size: int,
    dilation: int,
    eos_token_id: int,
    heads_per_ngram: int,
    query_start_loc_conv: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    num_spec_tokens: int = 0,
    state_dim_first: bool = True,
    null_block_id: int = -1,
    projection_kind: str = "int4",
) -> dict[str, torch.Tensor]:
    """Run standalone K0-K10 with an explicit K2 assembly boundary.

    ``rank_local_partials`` is caller-supplied mathematical input for K2.  Its
    rank-zero row is replaced in a private clone by the K1 gather result, then
    ``ple_embedding_assemble`` performs only the explicit sum.  No process group
    or all-reduce is created here; a caller integrating TP must own that boundary
    and pass the resulting partials/assembly contract explicitly.
    """
    if input_ids.dim() != 1 or input_ids.dtype != torch.int64:
        raise ValueError("input_ids must be a one-dimensional int64 tensor")
    if query_start_loc.dim() != 1 or ngram_context.dim() != 2:
        raise ValueError("N-gram metadata has an invalid rank")
    if hidden_states.dim() != 2 or hidden_states.size(0) != input_ids.size(0):
        raise ValueError("hidden_states must match packed input token count")
    if local_weight.dim() != 2 or local_weight.size(1) <= 0:
        raise ValueError("local_weight must be a rank-2 tensor with positive width")
    if local_weight.device != input_ids.device or local_weight.dtype != torch.float16:
        raise ValueError("local_weight must be FP16 on the input device")
    if heads_per_ngram <= 0 or layer_multipliers.numel() < 2:
        raise ValueError("N-gram metadata must define a positive head count")
    if ngram_context.size(1) != layer_multipliers.numel() - 1:
        raise ValueError("ngram_context width does not match multipliers")
    expected_embedding_width = (
        (layer_multipliers.numel() - 1)
        * heads_per_ngram
        * local_weight.size(1)
    )
    if rank_local_partials.dim() != 3 or rank_local_partials.size(0) <= 0:
        raise ValueError("rank_local_partials must have shape [tp, tokens, E]")
    if rank_local_partials.size(1) != input_ids.size(0):
        raise ValueError("rank_local_partials token count does not match input")
    if rank_local_partials.size(2) != expected_embedding_width:
        raise ValueError("rank_local_partials width does not match K1 output")
    if rank_local_partials.device != input_ids.device:
        raise ValueError("rank_local_partials must share the input device")
    if rank_local_partials.dtype != local_weight.dtype:
        raise ValueError("rank_local_partials must share local_weight dtype")
    if assembled_embedding.shape != rank_local_partials.shape[1:]:
        raise ValueError("assembled_embedding must match one local partial")
    if assembled_embedding.device != input_ids.device:
        raise ValueError("assembled_embedding must share the input device")
    if assembled_embedding.dtype != rank_local_partials.dtype:
        raise ValueError("assembled_embedding must match partial dtype")
    if not assembled_embedding.is_contiguous() or not rank_local_partials.is_contiguous():
        raise ValueError("K2 embedding buffers must be contiguous")
    if any(
        _tensors_alias(assembled_embedding, tensor)
        for tensor in (
            input_ids,
            query_start_loc,
            ngram_context,
            layer_multipliers,
            ngram_heads_vocab_sizes,
            ngram_heads_offsets,
            local_weight,
            local_vocab_start,
            local_num_rows,
            rank_local_partials,
            hidden_states,
        )
    ):
        raise ValueError("assembled_embedding must not alias K0-K2 inputs")
    if local_vocab_start.dim() != 1 or local_num_rows.dim() != 1:
        raise ValueError("local shard metadata must be one-dimensional")
    if local_vocab_start.dtype not in (torch.int32, torch.int64) or local_num_rows.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("local shard metadata must be int32 or int64")
    if local_vocab_start.device != input_ids.device or local_num_rows.device != input_ids.device:
        raise ValueError("local shard metadata must share the input device")

    tokens = input_ids.size(0)
    ngram_heads = (layer_multipliers.numel() - 1) * heads_per_ngram
    if ngram_heads <= 0:
        raise ValueError("N-gram metadata must define at least one head")
    ngram_output = torch.empty(
        (tokens, ngram_heads),
        dtype=torch.int64,
        device=input_ids.device,
    )
    ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        ngram_output,
        eos_token_id,
        heads_per_ngram,
    )
    local_partial = torch.empty_like(rank_local_partials[0])
    ple_embedding_gather(
        ngram_output,
        local_weight,
        local_vocab_start,
        local_num_rows,
        local_partial,
    )

    # Keep the caller's partials immutable while making the K1 -> K2 boundary
    # explicit.  The first row is the gathered local result; remaining rows are
    # deterministic caller-provided partials for the assembly contract.
    partials_for_assembly = rank_local_partials.clone()
    partials_for_assembly[0].copy_(local_partial)
    ple_embedding_assemble(partials_for_assembly, assembled_embedding)
    staged = ple_staged(
        assembled_embedding,
        hidden_states,
        key_weight,
        key_scale,
        value_weight,
        value_scale,
        norm_key_weight,
        norm_query_weight,
        norm_conv_weight,
        conv_state,
        conv_weights,
        state_indices,
        has_initial_state,
        mode,
        eps,
        group_size,
        dilation,
        query_start_loc_conv,
        num_accepted_tokens,
        num_spec_tokens,
        state_dim_first,
        null_block_id,
        projection_kind,
    )
    return {
        "ngram_ids": ngram_output,
        "local_partial": local_partial,
        "assembled_embedding": assembled_embedding,
        **staged,
    }


def _mixed_positive_non_overlapping(tensor: torch.Tensor) -> bool:
    dims = sorted(zip(tensor.stride(), tensor.shape))
    required_span = 1
    for stride, size in dims:
        if stride <= 0:
            return False
        if size <= 1:
            continue
        if stride < required_span:
            return False
        required_span += (size - 1) * stride
    return True


def _mixed_values(tensor: torch.Tensor, name: str) -> list[int]:
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32 or int64")
    return [int(value) for value in tensor.to(torch.int64).cpu().tolist()]


def _validate_mixed_short_conv_contract(
    input: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_token_indices: torch.Tensor,
    non_spec_token_indices: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    non_spec_mode: str,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices: torch.Tensor,
    non_spec_has_initial_state: torch.Tensor,
    output: torch.Tensor,
    num_spec_tokens: int,
    dilation: int,
    state_dim_first: bool,
    null_block_id: int,
) -> None:
    if input.dim() != 2 or input.size(1) <= 0 or not input.is_contiguous():
        raise ValueError("mixed input must be contiguous rank-2 with positive width")
    if output.device != input.device or output.dtype != input.dtype:
        raise ValueError("mixed output must share input device and dtype")
    if output.shape != input.shape or not output.is_contiguous():
        raise ValueError("mixed output must be contiguous and match input shape")
    if conv_weights.device != input.device or conv_weights.dtype != torch.float16:
        raise ValueError("mixed conv_weights must be FP16 on the input device")
    if conv_weights.dim() != 2 or not conv_weights.is_contiguous():
        raise ValueError("mixed conv_weights must be contiguous rank-2")
    if conv_weights.size(0) != input.size(1) or conv_weights.size(1) < 1:
        raise ValueError("mixed conv_weights shape does not match input width")
    if conv_state.device != input.device or conv_state.dim() != 3:
        raise ValueError("mixed conv_state must be rank-3 on the input device")
    if conv_state.size(0) <= 0 or conv_state.dtype not in (torch.float16, torch.float32):
        raise ValueError("mixed conv_state must have positive slots and FP16/FP32 dtype")
    if not _mixed_positive_non_overlapping(conv_state):
        raise ValueError("mixed conv_state must have positive non-overlapping strides")
    if dilation <= 0:
        raise ValueError("mixed dilation must be positive")
    kernel_span = int(conv_weights.size(1)) - 1
    if kernel_span > 128 // dilation:
        raise ValueError("mixed dilated convolution state length exceeds limit")
    state_length = kernel_span * dilation
    if num_spec_tokens < 0 or num_spec_tokens > 128:
        raise ValueError("mixed num_spec_tokens is outside the supported range")
    state_capacity = state_length + int(num_spec_tokens)
    channels = conv_state.size(1) if state_dim_first else conv_state.size(2)
    capacity = conv_state.size(2) if state_dim_first else conv_state.size(1)
    if channels != input.size(1) or capacity < state_capacity:
        raise ValueError("mixed conv_state shape/layout is incompatible")

    metadata = (
        spec_token_indices, non_spec_token_indices, spec_query_start_loc,
        spec_state_indices, num_accepted_tokens, non_spec_query_start_loc,
        non_spec_state_indices, non_spec_has_initial_state,
    )
    if any(t.device != input.device for t in metadata):
        raise ValueError("mixed metadata must share input device")
    if any(t.dim() != 1 or not t.is_contiguous() for t in metadata):
        raise ValueError("mixed metadata must be contiguous one-dimensional tensors")
    for name, tensor in (
        ("spec_token_indices", spec_token_indices),
        ("non_spec_token_indices", non_spec_token_indices),
        ("spec_query_start_loc", spec_query_start_loc),
        ("spec_state_indices", spec_state_indices),
        ("num_accepted_tokens", num_accepted_tokens),
        ("non_spec_query_start_loc", non_spec_query_start_loc),
        ("non_spec_state_indices", non_spec_state_indices),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64")
    if non_spec_has_initial_state.dtype != torch.bool:
        raise ValueError("non_spec_has_initial_state must be bool")

    readonly = (input, conv_weights, *metadata)
    for target, target_name in ((output, "output"), (conv_state, "conv_state")):
        for source in readonly:
            if _tensors_alias(target, source):
                raise ValueError(f"{target_name} must not alias mixed inputs or metadata")
    if _tensors_alias(output, conv_state):
        raise ValueError("mixed output and conv_state must not share storage")

    spec_tokens = _mixed_values(spec_token_indices, "spec_token_indices")
    non_spec_tokens = _mixed_values(non_spec_token_indices, "non_spec_token_indices")
    token_indices = spec_tokens + non_spec_tokens
    if len(token_indices) != input.size(0):
        raise ValueError("mixed token indices must cover every input token")
    if sorted(token_indices) != list(range(input.size(0))):
        raise ValueError("mixed token indices must be a disjoint permutation")

    spec_states = _mixed_values(spec_state_indices, "spec_state_indices")
    non_spec_states = _mixed_values(non_spec_state_indices, "non_spec_state_indices")
    state_indices = spec_states + non_spec_states
    if null_block_id >= 0 and null_block_id < conv_state.size(0):
        raise ValueError("null_block_id must not identify a real state slot")
    seen: set[int] = set()
    for index in state_indices:
        if index == null_block_id:
            continue
        if index < 0 or index >= conv_state.size(0):
            raise ValueError("mixed state metadata contains an out-of-range non-null slot")
        if index in seen:
            raise ValueError("mixed state metadata contains duplicate valid slots")
        seen.add(index)

    spec_starts = _mixed_values(spec_query_start_loc, "spec_query_start_loc")
    if len(spec_starts) != len(spec_states) + 1:
        raise ValueError("spec_query_start_loc must have one entry per spec request plus one")
    if not spec_starts or spec_starts[0] != 0 or spec_starts[-1] != len(spec_tokens):
        raise ValueError("spec_query_start_loc must cover spec tokens")
    previous = 0
    for begin, end in zip(spec_starts, spec_starts[1:]):
        if begin < previous or end < begin or end - begin > num_spec_tokens + 1:
            raise ValueError("spec query metadata is invalid")
        previous = end

    accepted = _mixed_values(num_accepted_tokens, "num_accepted_tokens")
    if len(accepted) != len(spec_states):
        raise ValueError("num_accepted_tokens must match spec requests")
    if any(value < 1 or value > num_spec_tokens + 1 for value in accepted):
        raise ValueError(
            "num_accepted_tokens is outside the supported range "
            "[1, num_spec_tokens + 1]"
        )

    if non_spec_mode not in ("decode", "prefill"):
        raise ValueError("non_spec_mode must be 'decode' or 'prefill'")
    non_initial = non_spec_has_initial_state.numel()
    if non_spec_mode == "decode":
        if len(non_spec_states) != len(non_spec_tokens) or non_initial != len(non_spec_tokens):
            raise ValueError("decode branch metadata must be token-indexed")
    else:
        non_starts = _mixed_values(
            non_spec_query_start_loc, "non_spec_query_start_loc"
        )
        if len(non_starts) != len(non_spec_states) + 1:
            raise ValueError("non-spec prefill offsets must match requests")
        if not non_starts or non_starts[0] != 0 or non_starts[-1] != len(non_spec_tokens):
            raise ValueError("non-spec prefill offsets must cover tokens")
        previous = 0
        for begin, end in zip(non_starts, non_starts[1:]):
            if begin < previous or end < begin:
                raise ValueError("non-spec prefill offsets must be monotonic")
            previous = end
        if non_initial != len(non_spec_states):
            raise ValueError("prefill branch initial mask must be request-indexed")


def ple_short_conv_mixed(
    input: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_token_indices: torch.Tensor,
    non_spec_token_indices: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    non_spec_mode: str,
    non_spec_query_start_loc: torch.Tensor,
    non_spec_state_indices: torch.Tensor,
    non_spec_has_initial_state: torch.Tensor,
    output: torch.Tensor,
    num_spec_tokens: int,
    dilation: int,
    state_dim_first: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    """Compose spec and non-spec branches and restore original token order.

    The two index tensors are an explicit stable permutation contract.  This
    wrapper deliberately does not infer request type from token position.
    ``non_spec_mode`` is either ``decode`` or ``prefill``; callers needing
    both regular modes invoke this wrapper twice with disjoint explicit
    permutations and a shared output/state only on the current stream.
    """
    _validate_mixed_short_conv_contract(
        input, conv_state, conv_weights, spec_token_indices,
        non_spec_token_indices, spec_query_start_loc, spec_state_indices,
        num_accepted_tokens, non_spec_mode, non_spec_query_start_loc,
        non_spec_state_indices, non_spec_has_initial_state, output,
        num_spec_tokens, dilation, state_dim_first, null_block_id,
    )

    # All validation happens before a stateful launch.  An entirely empty
    # packed batch is a validated no-op: do not clone or commit state.
    if input.size(0) == 0:
        return output

    # Work on a private copy so a later branch error cannot partially mutate
    # caller-owned state.  Empty branches are deliberately not dispatched.
    working_state = conv_state.clone()
    spec_input = input.index_select(0, spec_token_indices.to(torch.int64))
    non_spec_input = input.index_select(0, non_spec_token_indices.to(torch.int64))
    spec_output = torch.empty_like(spec_input)
    non_spec_output = torch.empty_like(non_spec_input)
    if spec_input.size(0) > 0:
        ple_short_conv_spec(
            spec_input, spec_query_start_loc, working_state, conv_weights,
            spec_state_indices, num_accepted_tokens, spec_output, num_spec_tokens,
            dilation, state_dim_first, null_block_id,
        )
    if non_spec_input.size(0) > 0:
        if non_spec_mode == "decode":
            ple_short_conv_decode(
                non_spec_input, working_state, conv_weights, non_spec_state_indices,
                non_spec_has_initial_state, non_spec_output, dilation,
                state_dim_first, null_block_id,
            )
        else:
            ple_short_conv_prefill(
                non_spec_input, non_spec_query_start_loc, working_state,
                conv_weights, non_spec_state_indices, non_spec_has_initial_state,
                non_spec_output, dilation, state_dim_first, null_block_id,
            )

    # Both non-empty branches have passed synchronous validation and are ordered
    # on the current stream. Commit state first, then restore token order.
    conv_state.copy_(working_state)
    if spec_input.size(0) > 0:
        output.index_copy_(0, spec_token_indices.to(torch.int64), spec_output)
    if non_spec_input.size(0) > 0:
        output.index_copy_(0, non_spec_token_indices.to(torch.int64), non_spec_output)
    return output


def _validate_mixed_three_way_contract(
    input: torch.Tensor,
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
    output: torch.Tensor,
    num_spec_tokens: int,
    dilation: int,
    state_dim_first: bool,
    null_block_id: int,
) -> None:
    """Validate the cold-path metadata for the three-way composition."""
    if input.dim() != 2 or input.size(1) <= 0 or not input.is_contiguous():
        raise ValueError(
            "mixed input must be contiguous rank-2 with positive width"
        )
    if output.device != input.device or output.dtype != input.dtype:
        raise ValueError("mixed output must share input device and dtype")
    if output.shape != input.shape or not output.is_contiguous():
        raise ValueError("mixed output must be contiguous and match input shape")
    if conv_weights.device != input.device or conv_weights.dtype != torch.float16:
        raise ValueError("mixed conv_weights must be FP16 on the input device")
    if conv_weights.dim() != 2 or not conv_weights.is_contiguous():
        raise ValueError("mixed conv_weights must be contiguous rank-2")
    if conv_weights.size(0) != input.size(1) or conv_weights.size(1) < 1:
        raise ValueError("mixed conv_weights shape does not match input width")
    if conv_state.device != input.device or conv_state.dim() != 3:
        raise ValueError("mixed conv_state must be rank-3 on the input device")
    if conv_state.size(0) <= 0 or conv_state.dtype not in (
        torch.float16,
        torch.float32,
    ):
        raise ValueError(
            "mixed conv_state must have positive slots and FP16/FP32 dtype"
        )
    if not _mixed_positive_non_overlapping(conv_state):
        raise ValueError(
            "mixed conv_state must have positive non-overlapping strides"
        )
    if dilation <= 0:
        raise ValueError("mixed dilation must be positive")
    kernel_span = int(conv_weights.size(1)) - 1
    if kernel_span > 128 // dilation:
        raise ValueError("mixed dilated convolution state length exceeds limit")
    if num_spec_tokens < 0 or num_spec_tokens > 128:
        raise ValueError("mixed num_spec_tokens is outside the supported range")
    state_length = kernel_span * dilation
    state_capacity = state_length + int(num_spec_tokens)
    channels = conv_state.size(1) if state_dim_first else conv_state.size(2)
    capacity = conv_state.size(2) if state_dim_first else conv_state.size(1)
    if channels != input.size(1) or capacity < state_capacity:
        raise ValueError("mixed conv_state shape/layout is incompatible")

    metadata = (
        spec_token_indices,
        decode_token_indices,
        prefill_token_indices,
        spec_query_start_loc,
        spec_state_indices,
        num_accepted_tokens,
        decode_state_indices,
        decode_has_initial_state,
        prefill_query_start_loc,
        prefill_state_indices,
        prefill_has_initial_state,
    )
    if any(t.device != input.device for t in metadata):
        raise ValueError("mixed metadata must share input device")
    if any(t.dim() != 1 or not t.is_contiguous() for t in metadata):
        raise ValueError(
            "mixed metadata must be contiguous one-dimensional tensors"
        )
    integer_metadata = (
        ("spec_token_indices", spec_token_indices),
        ("decode_token_indices", decode_token_indices),
        ("prefill_token_indices", prefill_token_indices),
        ("spec_query_start_loc", spec_query_start_loc),
        ("spec_state_indices", spec_state_indices),
        ("num_accepted_tokens", num_accepted_tokens),
        ("decode_state_indices", decode_state_indices),
        ("prefill_query_start_loc", prefill_query_start_loc),
        ("prefill_state_indices", prefill_state_indices),
    )
    for name, tensor in integer_metadata:
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64")
    for name, tensor in (
        ("decode_has_initial_state", decode_has_initial_state),
        ("prefill_has_initial_state", prefill_has_initial_state),
    ):
        if tensor.dtype != torch.bool:
            raise ValueError(f"{name} must be bool")

    readonly = (input, conv_weights, *metadata)
    for target, target_name in ((output, "output"), (conv_state, "conv_state")):
        for source in readonly:
            if _tensors_alias(target, source):
                raise ValueError(
                    f"{target_name} must not alias mixed inputs or metadata"
                )
    if _tensors_alias(output, conv_state):
        raise ValueError("mixed output and conv_state must not share storage")

    spec_tokens = _mixed_values(spec_token_indices, "spec_token_indices")
    decode_tokens = _mixed_values(decode_token_indices, "decode_token_indices")
    prefill_tokens = _mixed_values(
        prefill_token_indices, "prefill_token_indices"
    )
    token_indices = spec_tokens + decode_tokens + prefill_tokens
    if len(token_indices) != input.size(0):
        raise ValueError("mixed token indices must cover every input token")
    if sorted(token_indices) != list(range(input.size(0))):
        raise ValueError("mixed token indices must be a disjoint permutation")

    spec_states = _mixed_values(spec_state_indices, "spec_state_indices")
    decode_states = _mixed_values(decode_state_indices, "decode_state_indices")
    prefill_states = _mixed_values(
        prefill_state_indices, "prefill_state_indices"
    )
    if null_block_id >= 0 and null_block_id < conv_state.size(0):
        raise ValueError("null_block_id must not identify a real state slot")
    seen: set[int] = set()
    for index in spec_states + decode_states + prefill_states:
        if index == null_block_id:
            continue
        if index < 0 or index >= conv_state.size(0):
            raise ValueError(
                "mixed state metadata contains an out-of-range non-null slot"
            )
        if index in seen:
            raise ValueError(
                "mixed state metadata contains duplicate valid slots"
            )
        seen.add(index)

    spec_starts = _mixed_values(spec_query_start_loc, "spec_query_start_loc")
    if len(spec_starts) != len(spec_states) + 1:
        raise ValueError(
            "spec_query_start_loc must have one entry per spec request plus one"
        )
    if (
        not spec_starts
        or spec_starts[0] != 0
        or spec_starts[-1] != len(spec_tokens)
    ):
        raise ValueError("spec_query_start_loc must cover spec tokens")
    previous = 0
    for begin, end in zip(spec_starts, spec_starts[1:]):
        if (
            begin < previous
            or end < begin
            or end - begin > num_spec_tokens + 1
        ):
            raise ValueError("spec query metadata is invalid")
        previous = end

    accepted = _mixed_values(num_accepted_tokens, "num_accepted_tokens")
    if len(accepted) != len(spec_states):
        raise ValueError("num_accepted_tokens must match spec requests")
    if any(value < 1 or value > num_spec_tokens + 1 for value in accepted):
        raise ValueError(
            "num_accepted_tokens is outside the supported range "
            "[1, num_spec_tokens + 1]"
        )

    if len(decode_states) != len(decode_tokens):
        raise ValueError("decode state metadata must be token-indexed")
    if decode_has_initial_state.numel() != len(decode_tokens):
        raise ValueError(
            "decode initial-state metadata must be token-indexed"
        )

    prefill_starts = _mixed_values(
        prefill_query_start_loc, "prefill_query_start_loc"
    )
    if len(prefill_starts) != len(prefill_states) + 1:
        raise ValueError(
            "prefill_query_start_loc must have one entry per request plus one"
        )
    if (
        not prefill_starts
        or prefill_starts[0] != 0
        or prefill_starts[-1] != len(prefill_tokens)
    ):
        raise ValueError("prefill_query_start_loc must cover prefill tokens")
    previous = 0
    for begin, end in zip(prefill_starts, prefill_starts[1:]):
        if begin < previous or end < begin:
            raise ValueError("prefill query metadata is invalid")
        previous = end
    if prefill_has_initial_state.numel() != len(prefill_states):
        raise ValueError(
            "prefill initial-state metadata must be request-indexed"
        )


def ple_short_conv_mixed_three_way(
    input: torch.Tensor,
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
    output: torch.Tensor,
    num_spec_tokens: int,
    dilation: int,
    state_dim_first: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    """Run speculative, decode, and prefill branches as one transaction.

    Each token-index tensor selects a stable branch-local packing from the
    original input.  The three tensors must form a disjoint permutation, while
    each branch supplies its own request offsets and state metadata.  All
    synchronous validation happens before any native stateful launch.  Branches
    execute in the production order spec -> decode -> prefill on a private state
    clone; caller-owned state and output are committed only after all branches
    return successfully.  Device-side asynchronous faults are outside this
    synchronous transaction guarantee.
    """
    _validate_mixed_three_way_contract(
        input,
        conv_state,
        conv_weights,
        spec_token_indices,
        decode_token_indices,
        prefill_token_indices,
        spec_query_start_loc,
        spec_state_indices,
        num_accepted_tokens,
        decode_state_indices,
        decode_has_initial_state,
        prefill_query_start_loc,
        prefill_state_indices,
        prefill_has_initial_state,
        output,
        num_spec_tokens,
        dilation,
        state_dim_first,
        null_block_id,
    )

    if input.size(0) == 0:
        return output

    working_state = conv_state.clone()
    spec_input = input.index_select(0, spec_token_indices.to(torch.int64))
    decode_input = input.index_select(0, decode_token_indices.to(torch.int64))
    prefill_input = input.index_select(
        0, prefill_token_indices.to(torch.int64)
    )
    spec_output = torch.empty_like(spec_input)
    decode_output = torch.empty_like(decode_input)
    prefill_output = torch.empty_like(prefill_input)

    if spec_input.size(0) > 0:
        ple_short_conv_spec(
            spec_input,
            spec_query_start_loc,
            working_state,
            conv_weights,
            spec_state_indices,
            num_accepted_tokens,
            spec_output,
            num_spec_tokens,
            dilation,
            state_dim_first,
            null_block_id,
        )
    if decode_input.size(0) > 0:
        ple_short_conv_decode(
            decode_input,
            working_state,
            conv_weights,
            decode_state_indices,
            decode_has_initial_state,
            decode_output,
            dilation,
            state_dim_first,
            null_block_id,
        )
    if prefill_input.size(0) > 0:
        ple_short_conv_prefill(
            prefill_input,
            prefill_query_start_loc,
            working_state,
            conv_weights,
            prefill_state_indices,
            prefill_has_initial_state,
            prefill_output,
            dilation,
            state_dim_first,
            null_block_id,
        )

    # Commit only after every branch has completed; restore the original order
    # with the same explicit permutation used for branch-local inputs.
    conv_state.copy_(working_state)
    if spec_input.size(0) > 0:
        output.index_copy_(0, spec_token_indices.to(torch.int64), spec_output)
    if decode_input.size(0) > 0:
        output.index_copy_(0, decode_token_indices.to(torch.int64), decode_output)
    if prefill_input.size(0) > 0:
        output.index_copy_(
            0, prefill_token_indices.to(torch.int64), prefill_output
        )
    return output


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


def esimd_gemv_fp16(
    input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
) -> torch.Tensor:
    """FP16 weight GEMV (no quantization), including small decode batches.

    input:  [M, K] fp16
    weight: [N, K] fp16, contiguous (row-major). N inferred from weight.size(0),
            K from weight.size(1).
    output: [M, N] fp16.

    Used by gemma4's decode router projection (GateLinear is fp16 fp16-fp16).
    """
    return _ops.esimd_gemv_fp16(input, weight, output)


def esimd_gemv_fp16_gelu_mul(
    input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
) -> torch.Tensor:
    """Fused FP16 gate-up GEMV followed by GELU-tanh and elementwise multiply.

    ``weight`` is ``[2 * N, K]`` with gate rows first and up rows second.
    """
    return _ops.esimd_gemv_fp16_gelu_mul(input, weight, output)


def esimd_gemv_fp8_pert_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
) -> torch.Tensor:
    """Fused FP8 GEMV for 2 weight matrices with per-tensor scale.

    N0, N1 inferred from w0.size(0), w1.size(0). K from w0.size(1).
    """
    return _ops.esimd_gemv_fp8_pert_fused2(input, w0, s0, o0, w1, s1, o1)


def esimd_gemv_fp8_blockscale_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, s1: torch.Tensor, o1: torch.Tensor,
    block_n: int = 128, block_k: int = 128,
) -> torch.Tensor:
    """Decode dual GEMV for two E4M3 weights with 128x128 block scales.

    Results are written to ``o0`` and ``o1`` in place. Returns ``o0``.
    """
    return _ops.esimd_gemv_fp8_blockscale_fused2(
        input, w0, s0, o0, w1, s1, o1, block_n, block_k)


def esimd_gemv_fp8_blockscale_fp16_fused2(
    input: torch.Tensor,
    w0: torch.Tensor, s0: torch.Tensor, o0: torch.Tensor,
    w1: torch.Tensor, o1: torch.Tensor,
    block_n: int = 128, block_k: int = 128,
) -> torch.Tensor:
    """Decode dual GEMV for block-E4M3 qkvz plus an FP16 ba weight.

    Results are written to ``o0`` and ``o1`` in place. Returns ``o0``.
    """
    return _ops.esimd_gemv_fp8_blockscale_fp16_fused2(
        input, w0, s0, o0, w1, o1, block_n, block_k)


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


def esimd_gemm_int4_pgrp(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """INT4 GEMM via DPAS with per-group scale (group_size=128), for M>=2.

    Complements esimd_gemv_int4 (M=1).  Built on BMG XMX matrix engine;
    each byte of the packed INT4 weight already pairs (K_even, K_odd) in
    the layout DPAS's VNNI K-pair expects, so building the B tile is a
    fully vectorized nibble-extract + fused FMA dequant on simd<uint32,16>.

    input:        [M, K]       fp16
    weight:       [N, K/2]     uint8 — packed INT4 (2 per byte, low
                                       nibble = even K index)
    weight_scale: [N, K/128]   fp16  — per-group scale (group_size=128,
                                       may be negative per GGML q4_0)
    output:       [M, N]       fp16 — pre-allocated

    Requirements: N % 16 == 0, K % 128 == 0.  M, N, K inferred from
    tensor shapes.
    """
    return _ops.esimd_gemm_int4_pgrp(input, weight, weight_scale, output)


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


def esimd_qkv_split_norm_rope_v(
    qkv_state: torch.Tensor,
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    norm_wq: torch.Tensor,
    norm_wk: torch.Tensor,
    norm_wv: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    attn_output_gate: bool,
    rotary_dim: int = 256,
    cos_sin_cache: torch.Tensor = None,
) -> torch.Tensor:
    """Like esimd_qkv_split_norm_rope, but also RMSNorms V heads (no RoPE).

    All norm weights still follow the Qwen w+1.0 convention; gemma4 callers
    must pass (gemma_weight - 1.0) so the kernel's `+1.0` reproduces the
    desired RMSNorm scale. For gemma4 V-Norm (has_weight=False), pass a
    zeros([head_dim]) tensor so the kernel multiplies by ones.
    """
    return _ops.esimd_qkv_split_norm_rope_v(
        qkv_state, q_out, gate_out, k_out, v_out,
        norm_wq, norm_wk, norm_wv, positions,
        q_heads, kv_heads, attn_output_gate, rotary_dim, cos_sin_cache)


def esimd_qkv_split_norm_rope_muse_glimmer(
    qkv_state: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    q_scale: float,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """MuseGlimmer fused Q/K split + parameterless RMSNorm + q_scale + interleaved-pair RoPE.

    head_dim is fixed at 128. Q/K get RMSNorm (parameterless: no weight tensor)
    then interleaved-pair RoPE (is_neox_style=False). Q is additionally scaled by
    `q_scale` (MuseGlimmer: qk_scale_factor / sqrt(head_dim)) before RoPE.

    qkv_state:     [nTokens, (q_heads + 2*kv_heads)*128] fp16 contiguous
    q_out:         [nTokens, q_heads*128] fp16
    k_out/v_out:   [nTokens, kv_heads*128] fp16
    positions:     [nTokens] int32 or int64
    cos_sin_cache: [max_pos, 128] fp16, per row = concat(cos(64), sin(64))
    """
    # Keep the legacy compiled operator name until the next kernel rebuild.
    return _ops.esimd_qkv_split_norm_rope_onyx(
        qkv_state, q_out, k_out, v_out, positions,
        q_heads, kv_heads, float(q_scale), cos_sin_cache)


def esimd_qkv_split_norm_rope_muse_glimmer_neox(
    qkv_state: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    q_scale: float,
    eps: float,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """MuseGlimmer fused Q/K split + norm + half-split (NEOX) RoPE.

    ``positions`` accepts contiguous int32 or int64 tensors.
    """
    return _ops.esimd_qkv_split_norm_rope_onyx_neox(
        qkv_state, q_out, k_out, v_out, positions,
        q_heads, kv_heads, float(q_scale), float(eps), cos_sin_cache)


def esimd_qkv_split_norm_rope_mrope_v1(
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
    positions_bounds_proven: bool,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Qwen3.8 exact 3-axis interleaved MRoPE QKV postprocess.

    Fixed ABI: head_dim=256, rotary_dim=64, and mrope_section=[11, 11, 10].
    ``positions`` is [3, nTokens] and is consumed without collapsing axes.
    ``positions_bounds_proven`` is a scheduler-owned CPU proof bit; the native
    path intentionally does not perform a synchronizing XPU bounds read.
    """
    return _ops.esimd_qkv_split_norm_rope_mrope_v1(
        qkv_state, q_out, gate_out, k_out, v_out, norm_wq, norm_wk, positions,
        q_heads, kv_heads, attn_output_gate, positions_bounds_proven,
        cos_sin_cache
    )


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


def esimd_rms_norm(
    input: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Standalone RMSNorm (no residual add).

    output = rmsnorm(input) * weight

    For decode (M==1) one-token RMSNorm spots that don't share their input
    with the accumulating residual stream (e.g. gemma4 post_attn_norm,
    post_feedforward_layernorm_1, pre_feedforward_layernorm_2,
    post_feedforward_layernorm_2).

    Caller's responsibility: pass the right weight, including any per-model
    convention adjustment (e.g. (w-1) if calling from a Qwen-style stack
    where the kernel adds 1.0 — this kernel does NOT add 1.0; the multiply
    is done verbatim).
    """
    return _ops.esimd_rms_norm(input, output, weight, eps)


def esimd_fused_scaled_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: float,
) -> torch.Tensor:
    """Scaled fused add + RMSNorm.

        residual = (hidden_states + residual) * scalar  (in-place)
        hidden_states = rmsnorm(residual) * weight       (output)

    Used by gemma4 cross-layer fuse: layer N's `final_add + scalar_mul`
    plus layer N+1's `input_norm` collapse into one kernel call.
    `weight` must be pre-adjusted if the model uses a non-vanilla RMSNorm
    convention (caller's responsibility, same as esimd_fused_add_rms_norm).
    """
    return _ops.esimd_fused_scaled_add_rms_norm(
        hidden_states, residual, weight, eps, scalar)


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


def esimd_norm_gemv_fp8_blockscale(
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
    """Fused RMSNormGated + E4M3 GEMV with 128x128 weight scales.

    The result is written to ``output`` in place and the same tensor is
    returned.
    """
    return _ops.esimd_norm_gemv_fp8_blockscale(
        x, z, norm_weight, gemv_weight, gemv_scale, output, HV, V, eps)


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


def esimd_gdn_conv_fused_seq_spec(
    qkvz: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    spec_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ba: torch.Tensor,
    ssm_state: torch.Tensor,
    output: torch.Tensor,
    z_out: torch.Tensor,
    token_indx: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    num_spec_tokens: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    scale: float,
) -> torch.Tensor:
    """Fused sequential GDN for speculative tokens with rollback states."""
    return _ops.esimd_gdn_conv_fused_seq_spec(
        qkvz, conv_state, conv_weight, conv_bias, spec_state_indices,
        A_log, dt_bias, ba, ssm_state, output, z_out, token_indx,
        num_accepted_tokens, num_spec_decodes, num_spec_tokens,
        H, HV, K, V, scale)


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


def esimd_moe_gelu_tanh_mul(
    input: torch.Tensor,
    output: torch.Tensor,
    N_gate_up: int,
    N_half: int,
    total_rows: int,
) -> torch.Tensor:
    """GELU_tanh(gate) * up activation (gemma4 MoE)."""
    return _ops.esimd_moe_gelu_tanh_mul(input, output, N_gate_up, N_half, total_rows)


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


def esimd_moe_gemm_fp8_blockscale(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    expert_idx: torch.Tensor,
    N: int,
    K: int,
    num_experts: int,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """MoE grouped block-scaled FP8 GEMM (DeepSeek 128x128 weight block, w8a16).

    input:        [total_tokens, K] fp16   (expert-grouped/scattered rows)
    weight:       [num_experts, N, K] fp8_e4m3 (or uint8 bits)
    weight_scale: [num_experts, ceil(N/128), ceil(K/128)] float32 (weight_scale_inv)
    output:       [total_tokens, N] fp16
    expert_idx:   [num_experts+1] uint32/int32 — token start offsets per expert
    Activation stays fp16 (no per-token-group act quant).
    """
    return _ops.esimd_moe_gemm_fp8_blockscale(
        input, weight, weight_scale, output, expert_idx,
        N, K, num_experts, block_n, block_k)


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


def esimd_gemm_fp8_blockscale(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor, block_n: int = 128, block_k: int = 128,
) -> torch.Tensor:
    """FP8 block-scaled GEMM (DeepSeek-style), w8a16 (fp16 activation).

    Computes: output[M, N] = input[M, K] @ dequant(weight[N, K])^T
    where the fp8_e4m3 weight is dequantized with a 2D 128x128 block scale
    (weight_scale[nb, kb] scales the 128x128 weight block). The activation is
    NOT quantized — it is consumed in fp16 directly.

    input:        [M, K]                         fp16
    weight:       [N, K]                         fp8_e4m3 (or uint8 bits)
    weight_scale: [ceil(N/128), ceil(K/128)]     float32  (== weight_scale_inv)
    output:       [M, N]                         fp16 — pre-allocated

    M, N, K inferred from tensor shapes. K must be a multiple of block_k (128).
    Only block_n == block_k == 128 is currently supported.
    """
    return _ops.esimd_gemm_fp8_blockscale(
        input, weight, weight_scale, output, block_n, block_k)


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
    k_scale: float = 1.0,
    v_scale: float = 1.0,
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
        max_query_len, max_seq_len, k_scale, v_scale)


def eagle_page_attn_decode_separate(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Eagle paged attention for v0.26's separate K/V cache views.

    key_cache and value_cache are [num_blocks, page_size, num_kv_heads,
    head_dim] views into the packed vLLM cache.  The kernel consumes their
    strides directly, so this path does not materialize a reordered cache.
    """
    return _eagle_ops.page_attn_decode_separate(
        query, key_cache, value_cache, block_table, seq_lens, out,
        max_query_len, max_seq_len, k_scale, v_scale)


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


def moe_forward_full_fp8_block(
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
    """Small-batch MoE decode for 128x128 offline FP8 block weights.

    Supports 1 to 4 tokens and exactly one shared expert. Activations and the
    caller-owned output are contiguous fp16 tensors. Weights are contiguous
    ``float8_e4m3fn`` tensors with contiguous fp32 128x128 block scales;
    hidden and intermediate dimensions must both be divisible by 128. All
    tensors must be on the same XPU device. Routed scales have layout
    ``[E, N/128, K/128]``; shared-expert scales omit the leading expert
    dimension.
    """
    output = torch.empty_like(x)
    return _moe_batch.moe_forward_full_fp8_block(
        x, logits, output, gate_up_weight, gate_up_scale,
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


def moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    num_experts: int,
) -> torch.Tensor:
    return _moe_int4.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
        hidden_states,
        logits,
        w13_qweight_s4,
        w13_scales,
        w2_qweight_s4,
        w2_scales,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        output,
        top_k,
        num_shared_experts,
        num_experts,
    )


def moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    w13_qweight_s4: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_qweight_s4: torch.Tensor,
    w2_scales: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    num_shared_experts: int,
    num_experts: int,
) -> torch.Tensor:
    return _moe_int4.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
        hidden_states,
        logits,
        w13_qweight_s4,
        w13_scales,
        w2_qweight_s4,
        w2_scales,
        shared_gate_up_weight,
        shared_down_weight,
        shared_expert_gate_weight,
        output,
        top_k,
        num_shared_experts,
        num_experts,
    )


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


def moe_forward_full_gelu_tanh(
    x: torch.Tensor,
    logits: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Full MoE forward with gelu_tanh activation (gemma4, no shared expert)."""
    return _moe_batch.moe_forward_full_gelu_tanh(
        x, logits, gate_up_weight, gate_up_scale,
        down_weight, down_scale, top_k, n_routed_experts)


def moe_forward_full_fp8_grouped(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_tokens: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Full Gemma grouped FP8 forward with routing supplied externally."""
    return _moe_batch.moe_forward_full_fp8_grouped(
        x, gate_up_weight, gate_up_scale, down_weight, down_scale,
        routing_weights, expert_offsets, expert_tokens, top_k,
        n_routed_experts)


def moe_forward_full_gelu_tanh_routed(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """gelu_tanh MoE with caller-supplied routing.

    Use when the model needs routing logic the kernel's built-in
    softmax/topk does not cover (e.g. gemma4 folds per_expert_scale into
    the routing weights). topk_weights must be fp16 [T, top_k];
    topk_indices int32 [T, top_k]. Weight layout is the unmodified vllm
    FusedMoE format: w13 [E, 2*inter, hidden], w2 [E, hidden, inter].
    """
    return _moe_batch.moe_forward_full_gelu_tanh_routed(
        x, topk_weights, topk_indices,
        gate_up_weight, gate_up_scale,
        down_weight, down_scale,
        top_k, n_routed_experts)


def moe_forward_full_gelu_tanh_routed_decode(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Decode-only (M==1) variant of moe_forward_full_gelu_tanh_routed.

    Uses 1D block_load expert GEMV instead of the 16-wide 2D DPAS load,
    restoring full HBM bandwidth (~528 vs ~315 GB/s) for the single-token
    decode case. Requires x.size(0) == 1. Bit-identical to the DPAS path.
    """
    return _moe_batch.moe_forward_full_gelu_tanh_routed_decode(
        x, topk_weights, topk_indices,
        gate_up_weight, gate_up_scale,
        down_weight, down_scale,
        top_k, n_routed_experts)


def moe_forward_full_gelu_tanh_decode(
    x: torch.Tensor,
    logits: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Fully-fused gemma4 MoE decode (M==1): router logits in, output out.

    Internal topk (fp32 production kernel) + per_expert_scale fold + 1D-load
    gelu_tanh up/down GEMV + accumulate, all in one op. Removes the Python-side
    moe_topk call, torch scale-fold, and separate expert dispatch.
    """
    return _moe_batch.moe_forward_full_gelu_tanh_decode(
        x, logits, gate_up_weight, gate_up_scale,
        down_weight, down_scale, per_expert_scale,
        top_k, n_routed_experts)


def esimd_norm_gemv_norm_fp16(
    residual: torch.Tensor,
    scale_with_root: torch.Tensor,
    proj_w: torch.Tensor,
    pre_ff_w: torch.Tensor,
    router_logits: torch.Tensor,
    moe_input: torch.Tensor,
    eps: float,
) -> None:
    """Fused (rms_norm | * scale_with_root | fp16 GEMV) + (rms_norm | * pre_ff_w).

    Designed for gemma4 MoE branch where router(residual) and
    pre_feedforward_layernorm_2(residual) both compute rms(residual) and then
    apply different post-norm scales -- this kernel shares the rms computation
    and emits both outputs in one launch.

    Layout:
        residual:        [1, K] fp16
        scale_with_root: [K] fp16   (Gemma4 router scale * root_size, pre-folded)
        proj_w:          [N, K] fp16  (router projection)
        pre_ff_w:        [K] fp16   (pre_feedforward_layernorm_2 weight)
        router_logits:   [1, N] fp16
        moe_input:       [1, K] fp16

    Replaces 3 launches (esimd_rms_norm + esimd_gemv_fp16 + esimd_rms_norm).
    """
    return _ops.esimd_norm_gemv_norm_fp16(
        residual, scale_with_root, proj_w, pre_ff_w,
        router_logits, moe_input, eps)


def esimd_scaled_resadd_norm_gemv_fp8_pert(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_scale: torch.Tensor,
    qkv_out: torch.Tensor,
    eps: float,
    scalar: float,
) -> None:
    """Fused (h+r)*scalar + RMSNorm + FP8 GEMV (qkv_proj decode entry).

    Replaces 2 launches (esimd_fused_scaled_add_rms_norm + the FP8 GEMV inside
    vllm linear) with one. Updates residual in-place to (h+r)*scalar.
    """
    return _ops.esimd_scaled_resadd_norm_gemv_fp8_pert(
        hidden_states, residual, norm_weight, qkv_weight, qkv_scale, qkv_out,
        eps, scalar)


def esimd_norm_add_norm(
    h2_raw: torch.Tensor,
    h1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
    eps1: float,
    eps2: float,
) -> None:
    """Fused (rms_norm(h2_raw) × w1) + add to h1 + (rms_norm(h1_new) × w2).

    Layout:
        h2_raw: [1, K] fp16 (read)
        h1:     [1, K] fp16 (in-place: ← h2_normed_w1 + h1)
        w1, w2: [K] fp16
        out:    [1, K] fp16 (= rms_norm(h1) × w2)

    Replaces 2 launches (esimd_rms_norm + esimd_fused_add_rms_norm).
    """
    return _ops.esimd_norm_add_norm(h2_raw, h1, w1, w2, out, eps1, eps2)


def esimd_accum_norm_add_norm(
    routed_output: torch.Tensor,
    h1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    eps1: float,
    eps2: float,
) -> None:
    """Fused MoE-output (top_k sum) + RMSNorm × w1 + Add to h1 + RMSNorm × w2.

    Replaces 3 kernels (moe_accumulate + esimd_rms_norm + esimd_fused_add_rms_norm)
    or 2 kernels (moe_accumulate + esimd_norm_add_norm) with 1.
    """
    return _ops.esimd_accum_norm_add_norm(
        routed_output, h1, w1, w2, out, top_k, eps1, eps2)


def moe_forward_full_gelu_tanh_routed_no_accum(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    top_k: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Same as moe_forward_full_gelu_tanh_routed but without the final
    moe_accumulate kernel — returns [T*top_k, hidden] partial outputs so
    the caller can fuse the accumulate into a downstream kernel
    (e.g. esimd_accum_norm_add_norm)."""
    return _moe_batch.moe_forward_full_gelu_tanh_routed_no_accum(
        x, topk_weights, topk_indices,
        gate_up_weight, gate_up_scale,
        down_weight, down_scale,
        top_k, n_routed_experts)


def esimd_gemv_fp8_pert_bmg(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """BMG-tuned FP8 per-tensor GEMV with K_SPLIT and tail handling."""
    return _ops.esimd_gemv_fp8_pert_bmg(input, weight, weight_scale, output)
