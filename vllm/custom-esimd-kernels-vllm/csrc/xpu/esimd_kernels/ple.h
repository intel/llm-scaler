#pragma once

#include <ATen/ATen.h>

namespace ple {

at::Tensor ngram_ids(
    at::Tensor input_ids,
    at::Tensor query_start_loc,
    at::Tensor ngram_context,
    at::Tensor layer_multipliers,
    at::Tensor ngram_heads_vocab_sizes,
    at::Tensor ngram_heads_offsets,
    at::Tensor output,
    int64_t eos_token_id,
    int64_t heads_per_ngram);

at::Tensor embedding_gather(
    at::Tensor ngram_ids,
    at::Tensor local_weight,
    at::Tensor local_vocab_start,
    at::Tensor local_num_rows,
    at::Tensor local_partial);

at::Tensor grouped_norm(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    double eps,
    int64_t group_size);

at::Tensor hc_grouped_norm_v1(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    double eps);

at::Tensor hc_gate_mix_v1(
    at::Tensor input,
    at::Tensor gate,
    at::Tensor output);

at::Tensor score_gate(
    at::Tensor key,
    at::Tensor query,
    at::Tensor output,
    int64_t hidden_size);

at::Tensor gated_value(
    at::Tensor gate,
    at::Tensor value,
    at::Tensor output,
    int64_t hc_count);

at::Tensor residual_add(
    at::Tensor gated_value_flat,
    at::Tensor conv_output,
    at::Tensor output);

at::Tensor short_conv_decode(
    at::Tensor input,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor has_initial_state,
    at::Tensor output,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

// Production-only entry points. Metadata values must be proven by the
// scheduler-owned CPU construction path before these functions are called.
// They retain device, dtype, shape, stride, alias, and capacity checks but do
// not copy XPU metadata to the host.
at::Tensor short_conv_decode_trusted(
    at::Tensor input,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor has_initial_state,
    at::Tensor output,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

at::Tensor short_conv_prefill(
    at::Tensor input,
    at::Tensor query_start_loc,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor has_initial_state,
    at::Tensor output,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

at::Tensor short_conv_prefill_trusted(
    at::Tensor input,
    at::Tensor query_start_loc,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor has_initial_state,
    at::Tensor output,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

at::Tensor short_conv_spec(
    at::Tensor input,
    at::Tensor query_start_loc,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor num_accepted_tokens,
    at::Tensor output,
    int64_t num_spec_tokens,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

at::Tensor short_conv_spec_trusted(
    at::Tensor input,
    at::Tensor query_start_loc,
    at::Tensor conv_state,
    at::Tensor conv_weights,
    at::Tensor state_indices,
    at::Tensor num_accepted_tokens,
    at::Tensor output,
    int64_t num_spec_tokens,
    int64_t dilation,
    bool state_dim_first,
    int64_t null_block_id);

}  // namespace ple
