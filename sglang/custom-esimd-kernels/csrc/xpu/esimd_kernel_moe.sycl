/* esimd_kernel_moe.sycl — MoE ESIMD kernels compiled WITHOUT doubleGRF.
 * Standard GRF (256 regs) → 8 threads/XVE → 2× occupancy vs doubleGRF.
 * MoE kernels use ~42 GRFs — well within 256 limit.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/python.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "esimd_kernels/moe_ops.h"
#include "esimd_kernels/fp8_moe_gemm.h"

// Get queue for the specific device (multi-card safe)
static inline sycl::queue& get_device_queue(const at::Tensor& tensor) {
    return c10::xpu::getCurrentXPUStream(tensor.device().index()).queue();
}

// ======================== MoE Auxiliary Ops ========================

at::Tensor esimd_moe_topk(
    at::Tensor router_logits,
    at::Tensor top_values,
    at::Tensor top_indices,
    int64_t T)
{
    auto& q = get_device_queue(router_logits);
    moe_topk_host(
        reinterpret_cast<const fp16*>(router_logits.data_ptr()),
        reinterpret_cast<fp16*>(top_values.data_ptr()),
        top_indices.data_ptr<int32_t>(),
        (int)T, q);
    return top_values;
}

at::Tensor esimd_moe_scatter(
    at::Tensor hidden_states,
    at::Tensor router_top_value,
    at::Tensor sorted_token_ids,
    at::Tensor scattered_hidden,
    at::Tensor scattered_weights,
    int64_t K, int64_t topk, int64_t total_expanded)
{
    auto& q = get_device_queue(hidden_states);
    moe_scatter_host(
        reinterpret_cast<const fp16*>(hidden_states.data_ptr()),
        reinterpret_cast<const fp16*>(router_top_value.data_ptr()),
        sorted_token_ids.data_ptr<int32_t>(),
        reinterpret_cast<fp16*>(scattered_hidden.data_ptr()),
        reinterpret_cast<fp16*>(scattered_weights.data_ptr()),
        (int)K, (int)topk, (int)total_expanded, q);
    return scattered_hidden;
}

at::Tensor esimd_moe_scatter_fused(
    at::Tensor hidden_states,
    at::Tensor top_values,
    at::Tensor top_indices,
    at::Tensor scattered_hidden,
    at::Tensor scattered_weights,
    at::Tensor topk_ids,
    at::Tensor expert_start,
    at::Tensor max_tokens_out,
    int64_t K, int64_t topk, int64_t T, int64_t num_experts)
{
    auto& q = get_device_queue(hidden_states);
    auto experts_token_count = at::zeros({num_experts}, top_indices.options());
    auto token_to_scatter_offset = at::empty({T * topk}, top_indices.options());

    moe_scatter_fused_host(
        reinterpret_cast<const fp16*>(hidden_states.data_ptr()),
        reinterpret_cast<const fp16*>(top_values.data_ptr()),
        top_indices.data_ptr<int32_t>(),
        reinterpret_cast<fp16*>(scattered_hidden.data_ptr()),
        reinterpret_cast<fp16*>(scattered_weights.data_ptr()),
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(expert_start.data_ptr()),
        max_tokens_out.data_ptr<int32_t>(),
        experts_token_count.data_ptr<int32_t>(),
        token_to_scatter_offset.data_ptr<int32_t>(),
        (int)K, (int)topk, (int)T, (int)num_experts, q);
    return scattered_hidden;
}

at::Tensor esimd_moe_silu_mul(
    at::Tensor input,
    at::Tensor output,
    int64_t N_gate_up, int64_t N_half, int64_t total_rows)
{
    auto& q = get_device_queue(input);
    moe_silu_mul_host(
        reinterpret_cast<const fp16*>(input.data_ptr()),
        reinterpret_cast<fp16*>(output.data_ptr()),
        (int)N_gate_up, (int)N_half, (int)total_rows, q);
    return output;
}

at::Tensor esimd_moe_gather(
    at::Tensor moe_output,
    at::Tensor topk_ids,
    at::Tensor scattered_weights,
    at::Tensor final_hidden,
    int64_t K, int64_t topk, int64_t T)
{
    auto& q = get_device_queue(moe_output);
    moe_gather_host(
        reinterpret_cast<const fp16*>(moe_output.data_ptr()),
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<const fp16*>(scattered_weights.data_ptr()),
        reinterpret_cast<fp16*>(final_hidden.data_ptr()),
        (int)K, (int)topk, (int)T, q);
    return final_hidden;
}

at::Tensor esimd_moe_gemm_fp8(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor scale,
    at::Tensor output,
    at::Tensor expert_idx,
    int64_t N, int64_t K, int64_t num_experts, int64_t max_tokens_per_expert)
{
    auto& q = get_device_queue(input);
    moe_gemm_fp8_e5m2_dispatch(
        reinterpret_cast<const fp16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<fp16*>(output.data_ptr()),
        reinterpret_cast<const uint32_t*>(expert_idx.data_ptr()),
        (int)N, (int)K, (int)num_experts, (int)max_tokens_per_expert, q);
    return output;
}

at::Tensor esimd_moe_gemm_fp8_pert(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor scale,
    at::Tensor output,
    at::Tensor expert_idx,
    int64_t N, int64_t K, int64_t num_experts, int64_t max_tokens_per_expert)
{
    auto& q = get_device_queue(input);
    moe_gemm_fp8_e5m2_dispatch_pert(
        reinterpret_cast<const fp16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<fp16*>(output.data_ptr()),
        reinterpret_cast<const uint32_t*>(expert_idx.data_ptr()),
        (int)N, (int)K, (int)num_experts, (int)max_tokens_per_expert, q);
    return output;
}
