/* esimd_kernel_topk_v2.sycl — Standalone TopK V2 kernel (no DPAS dependency) */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/python.h>

#include <cstdint>
#include <sycl/sycl.hpp>

// Include only moe_ops.h (TopK kernels, no DPAS)
#include "esimd_kernels/moe_ops.h"

static inline sycl::queue& get_device_queue(const at::Tensor& tensor) {
    return c10::xpu::getCurrentXPUStream(tensor.device().index()).queue();
}

at::Tensor esimd_moe_topk_v2(
    at::Tensor router_logits,
    at::Tensor top_values,
    at::Tensor top_indices,
    int64_t T, int64_t num_experts, int64_t topk)
{
    auto& q = get_device_queue(router_logits);
    const auto* logits_ptr = reinterpret_cast<const fp16*>(router_logits.data_ptr());
    auto* values_ptr = reinterpret_cast<fp16*>(top_values.data_ptr());
    auto* indices_ptr = top_indices.data_ptr<int32_t>();

    if (num_experts == 512 && topk == 10) {
        moe_topk_v2_host<512, 10>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else if (num_experts == 512 && topk == 8) {
        moe_topk_v2_host<512, 8>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else if (num_experts == 256 && topk == 10) {
        moe_topk_v2_host<256, 10>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else if (num_experts == 256 && topk == 8) {
        moe_topk_v2_host<256, 8>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else if (num_experts == 128 && topk == 8) {
        moe_topk_v2_host<128, 8>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else if (num_experts == 128 && topk == 10) {
        moe_topk_v2_host<128, 10>(logits_ptr, values_ptr, indices_ptr, (int)T, q);
    } else {
        TORCH_CHECK(false, "esimd_moe_topk_v2: unsupported (num_experts=", num_experts,
                    ", topk=", topk, ")");
    }
    return top_values;
}
