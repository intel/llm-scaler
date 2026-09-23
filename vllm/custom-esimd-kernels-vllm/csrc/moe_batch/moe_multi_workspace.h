// SPDX-License-Identifier: Apache-2.0
// Current-call preflight and stream/M scratch for the existing M2..8 kernels.
#pragma once
#include <ATen/core/dispatch/Dispatcher.h>
#include <array>

class Qwen38MoeMultiWorkspace final {
 public:
  std::optional<at::Tensor> try_run(
      at::Tensor x, at::Tensor router, at::Tensor router_scale,
      std::vector<at::Tensor> weights, int64_t intermediate, bool grouped) {
    TORCH_CHECK(weights.size() == 7, "MoE multi workspace expects seven weights");
    if (!x.defined() || !x.device().is_xpu() || x.dim() != 2 ||
        x.size(0) < 2 || x.size(0) > 8 ||
        (intermediate != 80 && intermediate != 160))
      return std::nullopt;
    const auto device = x.device();
    const auto rows = x.size(0);
    const auto compatible = [&](const at::Tensor& t, at::ScalarType dtype,
                                at::IntArrayRef shape) {
      return t.defined() && t.device() == device && t.layout() == at::kStrided &&
          t.scalar_type() == dtype && t.sizes() == shape && t.is_contiguous() &&
          !t.is_neg() && !t.is_conj() &&
          reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0;
    };
    if (!compatible(x, at::kHalf, {rows, 2560}) ||
        !compatible(router, at::kByte, {512, 1280}) ||
        !compatible(router_scale, at::kHalf, {512, 20}))
      return std::nullopt;
    for (const auto& t : weights)
      if (!t.defined() || t.layout() != at::kStrided || t.is_neg() || t.is_conj())
        return std::nullopt;
    for (const int i : {0, 2})
      if (weights[i].scalar_type() == at::kChar)
        weights[i] = weights[i].view(at::kByte);
    const auto status = intermediate == 160
        ? qwen38_moe_compact160_weight_contract_v1(weights, device)
        : qwen38_moe_compact80_weight_contract_v1(weights, device);
    if (status != 0) return std::nullopt;

    // Resolve the same router DPAS entry as SymInt4LinearMethod before submit.
    using Router = at::Tensor(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
    static const auto router_op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("custom_esimd_kernels_vllm::esimd_gemm_int4_pgrp", "")
        .typed<Router>();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto stream = c10::xpu::getCurrentXPUStream(device.index());
    const auto& queue = stream.queue();
    TORCH_CHECK(queue.is_in_order() &&
                    queue.get_context() == c10::xpu::get_device_context() &&
                    queue.get_device() == c10::xpu::get_raw_device(device.index()),
                "MoE multi workspace requires current in-order XPU queue");
    auto& scratch = buffers_[stream.unwrap()][rows];
    if (scratch.output.defined()) {
      TORCH_CHECK(compatible(scratch.output, at::kHalf, {rows, 2560}) &&
                      compatible(scratch.logits, at::kHalf, {rows, 512}),
                  "MoE multi workspace output cache is inconsistent");
      bool aliases = false;
      for (const auto& output : {scratch.output, scratch.logits}) {
        for (const auto& input : {x, router, router_scale})
          aliases |= moe_asymmetric_v1_tensors_overlap(output, input);
        for (const auto& input : weights)
          aliases |= moe_asymmetric_v1_tensors_overlap(output, input);
      }
      if (aliases) scratch = {};
    }
    if (!scratch.output.defined()) {
      const auto options = x.options().requires_grad(false);
      scratch = {at::empty({rows, 512}, options), at::empty({rows, 2560}, options)};
    }
    // Validate/allocate before router; no fallback or replay after this point.
    if (intermediate == 160) (void)compact160_buffers(stream, device);
    (void)router_op.call(x, router, router_scale, scratch.logits);
    if (intermediate == 160)
      return compact160_submit<false>(
          x, scratch.logits, {}, weights[0], weights[1], weights[2], weights[3],
          weights[4], weights[5], weights[6], scratch.output);
    const auto operation = grouped
        ? moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1
        : moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1;
    return operation(x, scratch.logits, weights[0], weights[1], weights[2],
                     weights[3], weights[4], weights[5], weights[6],
                     scratch.output, 10, 1, 512);
  }

 private:
  struct Scratch { at::Tensor logits, output; };
  std::mutex mutex_;
  std::unordered_map<c10::Stream, std::array<Scratch, 9>> buffers_;
};
