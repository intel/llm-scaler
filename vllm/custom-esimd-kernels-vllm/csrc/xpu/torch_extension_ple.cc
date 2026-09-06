#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include <array>
#include <cmath>

#include "xpu/esimd_kernels/ple.h"

namespace {

// Metadata-only batching: preserve the complete ATen alias rule, including
// shared-owner storages, without 30 Python/C++ crossings per HC boundary.
static bool hc_outputs_alias_inputs_v1(
    at::TensorList outputs, at::TensorList inputs) {
  for (const auto& output : outputs) {
    for (const auto& input : inputs) {
      if (output.is_alias_of(input)) {
        return true;
      }
    }
  }
  return false;
}

// The existing HC ESIMD kernels use an even FP16 storage offset as their
// 4-byte alignment contract.  Keep the same conservative rule for every
// tensor in the host chain, including the caller-owned scratch outputs.
static inline void check_hc_chain_tensor(
    const at::Tensor& tensor,
    const at::Device& expected_device,
    const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_xpu(), name, " must be on XPU");
  TORCH_CHECK(
      tensor.device() == expected_device,
      name, " must be on the same XPU device as hidden");
  TORCH_CHECK(
      tensor.scalar_type() == at::kHalf,
      name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.storage_offset() % 2 == 0,
      name, " must be 4-byte aligned");
}

static void check_hc_chain_no_output_alias(
    const at::Tensor& hidden,
    const at::Tensor& block,
    const at::Tensor& injection,
    const at::Tensor& norm_weight,
    const at::Tensor& down_weight,
    const at::Tensor& up_weight,
    const at::Tensor& combined,
    const at::Tensor& normed,
    const at::Tensor& down,
    const at::Tensor& gate,
    const at::Tensor& mixed) {
  const std::array<const at::Tensor*, 6> inputs = {
      &hidden, &block, &injection, &norm_weight, &down_weight, &up_weight};
  const std::array<const at::Tensor*, 5> outputs = {
      &combined, &normed, &down, &gate, &mixed};
  const std::array<const char*, 5> output_names = {
      "combined", "normed", "down", "gate", "mixed"};

  for (size_t output_index = 0; output_index < outputs.size();
       ++output_index) {
    for (const at::Tensor* input : inputs) {
      TORCH_CHECK(
          !outputs[output_index]->is_alias_of(*input),
          "hc_combine_mix_m1_v1 output ", output_names[output_index],
          " must not share storage with any input");
    }
    for (size_t other_index = output_index + 1;
         other_index < outputs.size(); ++other_index) {
      TORCH_CHECK(
          !outputs[output_index]->is_alias_of(*outputs[other_index]),
          "hc_combine_mix_m1_v1 outputs ", output_names[output_index],
          " and ", output_names[other_index],
          " must not share storage");
    }
  }
}

static void validate_hc_combine_mix_m1(
    const at::Tensor& hidden,
    const at::Tensor& block,
    const at::Tensor& injection,
    const at::Tensor& norm_weight,
    const at::Tensor& down_weight,
    const at::Tensor& up_weight,
    const at::Tensor& combined,
    const at::Tensor& normed,
    const at::Tensor& down,
    const at::Tensor& gate,
    const at::Tensor& mixed,
    double eps) {
  TORCH_CHECK(hidden.defined(), "hidden must be defined");
  TORCH_CHECK(hidden.device().is_xpu(), "hidden must be on XPU");
  const at::Device device = hidden.device();

  check_hc_chain_tensor(hidden, device, "hidden");
  check_hc_chain_tensor(block, device, "block");
  check_hc_chain_tensor(injection, device, "injection");
  check_hc_chain_tensor(norm_weight, device, "norm_weight");
  check_hc_chain_tensor(down_weight, device, "down_weight");
  check_hc_chain_tensor(up_weight, device, "up_weight");
  check_hc_chain_tensor(combined, device, "combined");
  check_hc_chain_tensor(normed, device, "normed");
  check_hc_chain_tensor(down, device, "down");
  check_hc_chain_tensor(gate, device, "gate");
  check_hc_chain_tensor(mixed, device, "mixed");

  TORCH_CHECK(
      hidden.dim() == 2 && hidden.size(0) == 1 &&
          hidden.size(1) == 10240 &&
          block.dim() == 2 && block.size(0) == 1 &&
          block.size(1) == 2560 &&
          injection.dim() == 2 && injection.size(0) == 1 &&
          injection.size(1) == 4 &&
          norm_weight.dim() == 1 && norm_weight.size(0) == 10240 &&
          down_weight.dim() == 2 && down_weight.size(0) == 336 &&
          down_weight.size(1) == 10240 &&
          up_weight.dim() == 2 && up_weight.size(0) == 10240 &&
          up_weight.size(1) == 320 &&
          combined.dim() == 2 && combined.size(0) == 1 &&
          combined.size(1) == 10240 &&
          normed.dim() == 2 && normed.size(0) == 1 &&
          normed.size(1) == 10240 &&
          down.dim() == 2 && down.size(0) == 1 && down.size(1) == 336 &&
          gate.dim() == 2 && gate.size(0) == 1 && gate.size(1) == 10240 &&
          mixed.dim() == 2 && mixed.size(0) == 1 && mixed.size(1) == 2560,
      "hc_combine_mix_m1_v1 expects hidden [1, 10240], block [1, 2560], "
      "injection [1, 4], norm_weight [10240], down_weight [336, 10240], "
      "up_weight [10240, 320], combined/normed/gate [1, 10240], "
      "down [1, 336], and mixed [1, 2560]");

  check_hc_chain_no_output_alias(
      hidden, block, injection, norm_weight, down_weight, up_weight,
      combined, normed, down, gate, mixed);

  const float eps_fp32 = static_cast<float>(eps);
  TORCH_CHECK(
      std::isfinite(eps) && eps > 0.0 &&
          std::isfinite(eps_fp32) && eps_fp32 > 0.0f,
      "hc_combine_mix_m1_v1 eps must remain finite and positive in FP32");
}

// Host batching only: this sequences existing kernels on the current
// in-order stream.  It is intentionally not a GPU-fused implementation; it
// adds no host wait, GPU algorithm change, or cached device pointer.
//
// The PLE-only build does not link esimd_kernel.sycl.  Resolve the two GEMV
// operators through the dispatcher instead of taking direct symbol references
// so that this translation unit remains linkable in that standalone artifact.
template<bool FUSED_UP_GATE = false>
static void hc_combine_mix_m1_v1(
    at::Tensor hidden,
    at::Tensor block,
    at::Tensor injection,
    at::Tensor norm_weight,
    at::Tensor down_weight,
    at::Tensor up_weight,
    at::Tensor combined,
    at::Tensor normed,
    at::Tensor down,
    at::Tensor gate,
    at::Tensor mixed,
    double eps) {
  validate_hc_combine_mix_m1(
      hidden, block, injection, norm_weight, down_weight, up_weight,
      combined, normed, down, gate, mixed, eps);

  using HcDownFunction = void(at::Tensor, at::Tensor, at::Tensor);
  using HcGemvFunction = at::Tensor(at::Tensor, at::Tensor, at::Tensor);
  // Resolve before the first submit so a PLE-only DSO fails closed without
  // partially launching the chain when the GEMV schemas are unavailable.
  static const auto down_op = c10::Dispatcher::singleton()
                           .findSchemaOrThrow(
                               "custom_esimd_kernels_vllm::"
                               "esimd_hc_down_fp16_out",
                               "")
                           .typed<HcDownFunction>();
  if constexpr (FUSED_UP_GATE) {
    using FusedUpFunction = void(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
    static const auto fused_op = c10::Dispatcher::singleton()
        .findSchemaOrThrow(
            "custom_esimd_kernels_vllm::esimd_hc_up_gate_mix_m1_v1", "")
        .typed<FusedUpFunction>();
    const auto down_narrow = down.narrow(1, 0, 320);
    (void)ple::hc_combine_norm_v1(
        hidden, block, injection, norm_weight, combined, normed, eps);
    down_op.call(normed, down_weight, down);
    fused_op.call(down_narrow, up_weight, normed, mixed);
    return;
  }
  static const auto gemv_op = c10::Dispatcher::singleton()
                           .findSchemaOrThrow(
                               "custom_esimd_kernels_vllm::esimd_gemv_fp16",
                               "")
                           .typed<HcGemvFunction>();

  // The HC down kernel owns the 336-wide buffer; the up projection consumes
  // its first 320; entries [320:324] are injection, [324:336] are padding.
  const at::Tensor down_narrow = down.narrow(/*dim=*/1, /*start=*/0,
                                             /*length=*/320);
  (void)ple::hc_combine_norm_v1(
      hidden, block, injection, norm_weight, combined, normed, eps);
  down_op.call(normed, down_weight, down);
  (void)gemv_op.call(down_narrow, up_weight, gate);
  (void)ple::hc_gate_mix_v1(normed, gate, mixed);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  m.def("hc_outputs_alias_inputs_v1(Tensor[] outputs, Tensor[] inputs) -> bool",
        &hc_outputs_alias_inputs_v1);
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

  m.def("hc_grouped_norm_v1(Tensor input, Tensor weight, "
        "Tensor(a!) output, float eps) -> ()");
  m.impl("hc_grouped_norm_v1", torch::kXPU,
         [](at::Tensor input, at::Tensor weight, at::Tensor output,
            double eps) -> void {
           ple::hc_grouped_norm_v1(input, weight, output, eps);
         });

  m.def("hc_gate_mix_v1(Tensor input, Tensor gate, Tensor(a!) output) -> ()");
  m.impl("hc_gate_mix_v1", torch::kXPU,
         [](at::Tensor input, at::Tensor gate, at::Tensor output) -> void {
           ple::hc_gate_mix_v1(input, gate, output);
         });

  m.def("hc_gate_mix_m4_v1(Tensor input, Tensor gate, "
        "Tensor(a!) output) -> ()");
  m.impl("hc_gate_mix_m4_v1", torch::kXPU,
         [](at::Tensor input, at::Tensor gate, at::Tensor output) -> void {
           ple::hc_gate_mix_m4_v1(input, gate, output);
         });

  m.def("hc_combine_v1(Tensor hidden_states, Tensor block_output, "
        "Tensor injection, Tensor(a!) output) -> ()");
  m.impl("hc_combine_v1", torch::kXPU,
         [](at::Tensor hidden_states, at::Tensor block_output,
            at::Tensor injection, at::Tensor output) -> void {
           ple::hc_combine_v1(
               hidden_states, block_output, injection, output);
         });

  m.def("hc_combine_norm_v1(Tensor hidden_states, Tensor block_output, "
        "Tensor injection, Tensor weight, Tensor(a!) combined_output, "
        "Tensor(b!) normed_output, float eps) -> ()");
  m.impl("hc_combine_norm_v1", torch::kXPU,
         [](at::Tensor hidden_states, at::Tensor block_output,
            at::Tensor injection, at::Tensor weight,
            at::Tensor combined_output, at::Tensor normed_output,
            double eps) -> void {
           ple::hc_combine_norm_v1(
               hidden_states, block_output, injection, weight,
               combined_output, normed_output, eps);
         });

  m.def(
      "hc_combine_mix_m1_v1(Tensor hidden, Tensor prev_block, "
      "Tensor prev_injection, Tensor norm_weight, Tensor down_weight, "
      "Tensor up_weight, Tensor(a!) combined, Tensor(b!) normed, "
      "Tensor(c!) down, Tensor(d!) gate, Tensor(e!) mixed, float eps) -> ()");
  m.impl("hc_combine_mix_m1_v1", torch::kXPU,
         [](at::Tensor hidden, at::Tensor prev_block,
            at::Tensor prev_injection, at::Tensor norm_weight,
            at::Tensor down_weight, at::Tensor up_weight,
            at::Tensor combined, at::Tensor normed, at::Tensor down,
            at::Tensor gate, at::Tensor mixed, double eps) -> void {
           hc_combine_mix_m1_v1(
               hidden, prev_block, prev_injection, norm_weight, down_weight,
               up_weight, combined, normed, down, gate, mixed, eps);
         });

  m.def(
      "hc_combine_mix_m1_v2(Tensor hidden, Tensor prev_block, "
      "Tensor prev_injection, Tensor norm_weight, Tensor down_weight, "
      "Tensor up_weight, Tensor(a!) combined, Tensor(b!) normed, "
      "Tensor(c!) down, Tensor(d!) gate, Tensor(e!) mixed, float eps) -> ()");
  m.impl("hc_combine_mix_m1_v2", torch::kXPU,
         &hc_combine_mix_m1_v1<true>);

  m.def("hc_combine_norm_m4_v1(Tensor hidden_states, Tensor block_output, "
        "Tensor injection, Tensor weight, Tensor(a!) combined_output, "
        "Tensor(b!) normed_output, float eps) -> ()");
  m.impl("hc_combine_norm_m4_v1", torch::kXPU,
         [](at::Tensor hidden_states, at::Tensor block_output,
            at::Tensor injection, at::Tensor weight,
            at::Tensor combined_output, at::Tensor normed_output,
            double eps) -> void {
           ple::hc_combine_norm_m4_v1(
               hidden_states, block_output, injection, weight,
               combined_output, normed_output, eps);
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

  m.def("ple_gated_value_grouped_norm(Tensor gate, Tensor value, "
        "Tensor weight, Tensor(a!) raw_output, "
        "Tensor(b!) normalized_output, float eps) -> ()");
  m.impl("ple_gated_value_grouped_norm", torch::kXPU,
         [](at::Tensor gate, at::Tensor value, at::Tensor weight,
            at::Tensor raw_output, at::Tensor normalized_output,
            double eps) -> void {
           ple::gated_value_grouped_norm(
               gate, value, weight, raw_output, normalized_output, eps);
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
