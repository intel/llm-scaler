#include <torch/all.h>
#include <torch/library.h>

#include "xpu/esimd_kernels/ple.h"

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  m.def("ple_ngram_ids(Tensor input_ids, Tensor query_start_loc, "
        "Tensor ngram_context, Tensor layer_multipliers, "
        "Tensor ngram_heads_vocab_sizes, Tensor ngram_heads_offsets, "
        "Tensor(a!) output, int eos_token_id, int heads_per_ngram) -> ()");
  m.impl("ple_ngram_ids", torch::kXPU,
         [](at::Tensor input_ids, at::Tensor query_start_loc,
            at::Tensor ngram_context, at::Tensor layer_multipliers,
            at::Tensor ngram_heads_vocab_sizes, at::Tensor ngram_heads_offsets,
            at::Tensor output, int64_t eos_token_id,
            int64_t heads_per_ngram) -> void {
           ple::ngram_ids(input_ids, query_start_loc, ngram_context,
                          layer_multipliers, ngram_heads_vocab_sizes,
                          ngram_heads_offsets, output, eos_token_id,
                          heads_per_ngram);
         });

  m.def("ple_embedding_gather(Tensor ngram_ids, Tensor local_weight, "
        "Tensor local_vocab_start, Tensor local_num_rows, "
        "Tensor(a!) local_partial) -> ()");
  m.impl("ple_embedding_gather", torch::kXPU,
         [](at::Tensor ngram_ids, at::Tensor local_weight,
            at::Tensor local_vocab_start, at::Tensor local_num_rows,
            at::Tensor local_partial) -> void {
           ple::embedding_gather(ngram_ids, local_weight, local_vocab_start,
                                 local_num_rows, local_partial);
         });

  m.def("ple_grouped_norm(Tensor input, Tensor weight, Tensor(a!) output, "
        "float eps, int group_size) -> ()");
  m.impl("ple_grouped_norm", torch::kXPU,
         [](at::Tensor input, at::Tensor weight, at::Tensor output,
            double eps, int64_t group_size) -> void {
           ple::grouped_norm(input, weight, output, eps, group_size);
         });

  m.def("ple_score_gate(Tensor key, Tensor query, Tensor(a!) output, "
        "int hidden_size) -> ()");
  m.impl("ple_score_gate", torch::kXPU,
         [](at::Tensor key, at::Tensor query, at::Tensor output,
            int64_t hidden_size) -> void {
           ple::score_gate(key, query, output, hidden_size);
         });

  m.def("ple_gated_value(Tensor gate, Tensor value, Tensor(a!) output, "
        "int hc_count) -> ()");
  m.impl("ple_gated_value", torch::kXPU,
         [](at::Tensor gate, at::Tensor value, at::Tensor output,
            int64_t hc_count) -> void {
           ple::gated_value(gate, value, output, hc_count);
         });

  m.def("ple_residual_add(Tensor gated_value_flat, Tensor conv_output, "
        "Tensor(a!) output) -> ()");
  m.impl("ple_residual_add", torch::kXPU,
         [](at::Tensor gated_value_flat, at::Tensor conv_output,
            at::Tensor output) -> void {
           ple::residual_add(gated_value_flat, conv_output, output);
         });

  m.def("ple_short_conv_decode(Tensor input, Tensor(a!) conv_state, "
        "Tensor conv_weights, Tensor state_indices, "
        "Tensor has_initial_state, Tensor(b!) output, int dilation, "
        "bool state_dim_first, int null_block_id) -> ()");
  m.impl("ple_short_conv_decode", torch::kXPU,
         [](at::Tensor input, at::Tensor conv_state,
            at::Tensor conv_weights, at::Tensor state_indices,
            at::Tensor has_initial_state, at::Tensor output,
            int64_t dilation, bool state_dim_first,
            int64_t null_block_id) -> void {
           ple::short_conv_decode(input, conv_state, conv_weights,
                                  state_indices, has_initial_state, output,
                                  dilation, state_dim_first, null_block_id);
         });

  m.def("ple_short_conv_decode_trusted(Tensor input, Tensor(a!) conv_state, "
        "Tensor conv_weights, Tensor state_indices, "
        "Tensor has_initial_state, Tensor(b!) output, int dilation, "
        "bool state_dim_first, int null_block_id) -> ()");
  m.impl("ple_short_conv_decode_trusted", torch::kXPU,
         [](at::Tensor input, at::Tensor conv_state,
            at::Tensor conv_weights, at::Tensor state_indices,
            at::Tensor has_initial_state, at::Tensor output,
            int64_t dilation, bool state_dim_first,
            int64_t null_block_id) -> void {
           ple::short_conv_decode_trusted(
               input, conv_state, conv_weights, state_indices,
               has_initial_state, output, dilation, state_dim_first,
               null_block_id);
         });

  m.def("ple_short_conv_prefill(Tensor input, Tensor query_start_loc, "
        "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
        "Tensor has_initial_state, Tensor(b!) output, int dilation, "
        "bool state_dim_first, int null_block_id) -> ()");
  m.impl("ple_short_conv_prefill", torch::kXPU,
         [](at::Tensor input, at::Tensor query_start_loc,
            at::Tensor conv_state, at::Tensor conv_weights,
            at::Tensor state_indices, at::Tensor has_initial_state,
            at::Tensor output, int64_t dilation, bool state_dim_first,
            int64_t null_block_id) -> void {
           ple::short_conv_prefill(input, query_start_loc, conv_state,
                                   conv_weights, state_indices,
                                   has_initial_state, output, dilation,
                                   state_dim_first, null_block_id);
         });

  m.def("ple_short_conv_prefill_trusted(Tensor input, Tensor query_start_loc, "
        "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
        "Tensor has_initial_state, Tensor(b!) output, int dilation, "
        "bool state_dim_first, int null_block_id) -> ()");
  m.impl("ple_short_conv_prefill_trusted", torch::kXPU,
         [](at::Tensor input, at::Tensor query_start_loc,
            at::Tensor conv_state, at::Tensor conv_weights,
            at::Tensor state_indices, at::Tensor has_initial_state,
            at::Tensor output, int64_t dilation, bool state_dim_first,
            int64_t null_block_id) -> void {
           ple::short_conv_prefill_trusted(
               input, query_start_loc, conv_state, conv_weights,
               state_indices, has_initial_state, output, dilation,
               state_dim_first, null_block_id);
         });

  m.def("ple_short_conv_spec(Tensor input, Tensor query_start_loc, "
        "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
        "Tensor num_accepted_tokens, Tensor(b!) output, "
        "int num_spec_tokens, int dilation, bool state_dim_first, "
        "int null_block_id) -> ()");
  m.impl("ple_short_conv_spec", torch::kXPU,
         [](at::Tensor input, at::Tensor query_start_loc,
            at::Tensor conv_state, at::Tensor conv_weights,
            at::Tensor state_indices, at::Tensor num_accepted_tokens,
            at::Tensor output, int64_t num_spec_tokens, int64_t dilation,
            bool state_dim_first, int64_t null_block_id) -> void {
           ple::short_conv_spec(input, query_start_loc, conv_state,
                                conv_weights, state_indices,
                                num_accepted_tokens, output, num_spec_tokens,
                                dilation, state_dim_first, null_block_id);
         });

  m.def("ple_short_conv_spec_trusted(Tensor input, Tensor query_start_loc, "
        "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
        "Tensor num_accepted_tokens, Tensor(b!) output, "
        "int num_spec_tokens, int dilation, bool state_dim_first, "
        "int null_block_id) -> ()");
  m.impl("ple_short_conv_spec_trusted", torch::kXPU,
         [](at::Tensor input, at::Tensor query_start_loc,
            at::Tensor conv_state, at::Tensor conv_weights,
            at::Tensor state_indices, at::Tensor num_accepted_tokens,
            at::Tensor output, int64_t num_spec_tokens, int64_t dilation,
            bool state_dim_first, int64_t null_block_id) -> void {
           ple::short_conv_spec_trusted(
               input, query_start_loc, conv_state, conv_weights,
               state_indices, num_accepted_tokens, output, num_spec_tokens,
               dilation, state_dim_first, null_block_id);
         });
}
