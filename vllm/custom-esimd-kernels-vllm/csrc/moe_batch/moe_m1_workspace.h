// SPDX-License-Identifier: Apache-2.0
// One host entry for Qwen3.8 compact TP4/TP8 M=1. Device math is unchanged.
#pragma once
#include <torch/custom_class.h>

class Qwen38MoeM1Workspace final : public torch::CustomClassHolder {
 public:
  std::optional<at::Tensor> try_run(
      at::Tensor x, at::Tensor router, at::Tensor router_scale,
      std::vector<at::Tensor> weights, int64_t intermediate) {
    TORCH_CHECK(weights.size() == 7, "MoE workspace expects seven weights");
    if ((intermediate != 80 && intermediate != 160) || !x.device().is_xpu())
      return std::nullopt;
    const auto device = x.device();
    const auto compatible = [&](const at::Tensor& t, at::ScalarType dtype,
                                at::IntArrayRef shape) {
      return t.defined() && t.device() == device && t.layout() == at::kStrided &&
          t.scalar_type() == dtype && t.sizes() == shape && t.is_contiguous() &&
          !t.is_neg() && !t.is_conj() &&
          reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0;
    };
    if (!compatible(x, at::kHalf, {1, 2560}) ||
        !compatible(router, at::kByte, {512, 1280}) ||
        !compatible(router_scale, at::kHalf, {512, 20}))
      return std::nullopt;
    // Direct C++ calls do not run dispatcher Negative/Conjugate fallbacks.
    // Keep those views on the established Python/dispatcher path.
    for (const auto& t : weights)
      if (!t.defined() || t.layout() != at::kStrided || t.is_neg() || t.is_conj())
        return std::nullopt;
    // Reinterpret the LIVE TensorImpl, never a cached Python int8->uint8 view.
    // Byte views neither copy nor requantize weights, including group128 tails.
    for (const int i : {0, 2}) {
      if (weights[i].scalar_type() == at::kChar)
        weights[i] = weights[i].view(at::kByte);
    }
    const int status = intermediate == 160
        ? qwen38_moe_compact160_weight_contract_v1(weights, device)
        : qwen38_moe_compact80_weight_contract_v1(weights, device);
    if (status != 0) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);
    const auto stream = c10::xpu::getCurrentXPUStream(device.index()).unwrap();
    auto found = outputs_.find(stream);
    if (found != outputs_.end()) {
      TORCH_CHECK(compatible(found->second, at::kHalf, {1, 2560}),
                  "MoE workspace output cache is inconsistent");
      bool aliases = moe_asymmetric_v1_tensors_overlap(found->second, x) ||
          moe_asymmetric_v1_tensors_overlap(found->second, router) ||
          moe_asymmetric_v1_tensors_overlap(found->second, router_scale);
      for (const auto& t : weights)
        aliases |= moe_asymmetric_v1_tensors_overlap(found->second, t);
      if (aliases) {
        // Retain the input's owner while replacing scratch that now aliases it.
        found->second = at::empty({1, 2560}, x.options());
      }
    } else {
      found = outputs_.emplace(stream, at::empty({1, 2560}, x.options())).first;
    }
    auto output = found->second;
    // These established entries still validate the full transaction before
    // their first submit (including queue, lazy bits and output overlap).
    // Their exceptions are hard errors; never convert them into a fallback.
    if (intermediate == 160) {
      return moe_forward_compact160_router_out_v1(
          x, router, router_scale, weights[0], weights[1], weights[2], weights[3],
          weights[4], weights[5], weights[6], output, 10, 1, 512);
    }
    return moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1(
        x, router, router_scale, weights[0], weights[1], weights[2], weights[3],
        weights[4], weights[5], weights[6], output, 10, 1, 512);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<c10::Stream, at::Tensor> outputs_;
};

static auto register_qwen38_moe_m1_workspace =
    torch::class_<Qwen38MoeM1Workspace>("moe_int4_ops", "Qwen38M1WorkspaceV1")
        .def(torch::init<>())
        .def("try_run", &Qwen38MoeM1Workspace::try_run);
