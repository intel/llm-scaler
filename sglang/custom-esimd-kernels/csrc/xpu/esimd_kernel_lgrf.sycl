/* esimd_kernel_lgrf.sycl — SYCL compilation unit for doubleGRF ESIMD kernels.
 * Compiled with -doubleGRF flags for 512 register access.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/python.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "esimd_kernels/gdn_conv_fused.h"
#include "esimd_kernels/gdn_conv_fused_seq.h"

// Get queue for the specific device (multi-card safe)
static inline sycl::queue& get_device_queue(const at::Tensor& tensor) {
    return c10::xpu::getCurrentXPUStream(tensor.device().index()).queue();
}

at::Tensor esimd_gdn_conv_fused(
    at::Tensor qkvz,                // [N, qkvz_dim] fp16 — projected_states_qkvz
    at::Tensor conv_state,           // [num_cache, 3, 2048] fp16, strided dim0
    at::Tensor conv_weight,          // [2048, 4] fp16
    at::Tensor conv_bias,            // [2048] fp16 (zeros if model has no bias)
    at::Tensor conv_state_indices,   // [N] int32
    at::Tensor A_log,                // [HV] fp16
    at::Tensor dt_bias,              // [HV] fp16
    at::Tensor ba,                   // [N, 2*HV] fp16 — projected_states_ba
    at::Tensor ssm_state,            // [num_states, HV, V, K] fp16, strided dim0
    at::Tensor ssm_state_indices,    // [N] int32
    at::Tensor output,               // [N, HV, V] fp16
    at::Tensor z_out,                // [N, HV, V] fp16
    int64_t N, int64_t H, int64_t HV,
    int64_t K, int64_t V,
    double scale)
{
    auto& dpcpp_queue = get_device_queue(qkvz);

    // Extract strides (in fp16 elements)
    int64_t qkvz_stride0 = qkvz.stride(0);
    int64_t conv_stride0 = conv_state.stride(0);
    int64_t ssm_stride0 = ssm_state.stride(0);
    int64_t ba_stride0 = ba.stride(0);

    // Extract raw fp16 pointers
    auto* p_qkvz    = reinterpret_cast<const fp16*>(qkvz.data_ptr());
    auto* p_cstate  = reinterpret_cast<fp16*>(conv_state.data_ptr());
    auto* p_cweight = reinterpret_cast<const fp16*>(conv_weight.data_ptr());
    auto* p_cbias   = reinterpret_cast<const fp16*>(conv_bias.data_ptr());
    auto* p_csidx   = conv_state_indices.data_ptr<int>();
    auto* p_alog    = reinterpret_cast<const fp16*>(A_log.data_ptr());
    auto* p_dtbias  = reinterpret_cast<const fp16*>(dt_bias.data_ptr());
    auto* p_ba      = reinterpret_cast<const fp16*>(ba.data_ptr());
    auto* p_sstate  = reinterpret_cast<fp16*>(ssm_state.data_ptr());
    auto* p_ssidx   = ssm_state_indices.data_ptr<int>();
    auto* p_out     = reinterpret_cast<fp16*>(output.data_ptr());
    auto* p_zout    = reinterpret_cast<fp16*>(z_out.data_ptr());

    gdn_conv_fused_host(
        p_qkvz, qkvz_stride0, p_cstate, p_cweight, p_cbias, p_csidx,
        p_alog, p_dtbias, p_ba, ba_stride0,
        p_sstate, p_ssidx, p_out, p_zout,
        (int)N, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_stride0, ssm_stride0,
        dpcpp_queue);

    return output;
}

at::Tensor esimd_gdn_conv_fused_seq(
    at::Tensor qkvz,                // [N, qkvz_dim] fp16 — sequential [q|k|v|z]
    at::Tensor conv_state,
    at::Tensor conv_weight,
    at::Tensor conv_bias,
    at::Tensor conv_state_indices,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor ba,                   // [N, 2*HV] fp16 — sequential [b|a]
    at::Tensor ssm_state,
    at::Tensor ssm_state_indices,
    at::Tensor output,
    at::Tensor z_out,
    int64_t N, int64_t H, int64_t HV,
    int64_t K, int64_t V,
    double scale)
{
    auto& dpcpp_queue = get_device_queue(qkvz);

    int64_t qkvz_stride0 = qkvz.stride(0);
    int64_t conv_stride0 = conv_state.stride(0);
    int64_t ssm_stride0 = ssm_state.stride(0);
    int64_t ba_stride0 = ba.stride(0);

    auto* p_qkvz    = reinterpret_cast<const fp16*>(qkvz.data_ptr());
    auto* p_cstate  = reinterpret_cast<fp16*>(conv_state.data_ptr());
    auto* p_cweight = reinterpret_cast<const fp16*>(conv_weight.data_ptr());
    auto* p_cbias   = reinterpret_cast<const fp16*>(conv_bias.data_ptr());
    auto* p_csidx   = conv_state_indices.data_ptr<int>();
    auto* p_alog    = reinterpret_cast<const fp16*>(A_log.data_ptr());
    auto* p_dtbias  = reinterpret_cast<const fp16*>(dt_bias.data_ptr());
    auto* p_ba      = reinterpret_cast<const fp16*>(ba.data_ptr());
    auto* p_sstate  = reinterpret_cast<fp16*>(ssm_state.data_ptr());
    auto* p_ssidx   = ssm_state_indices.data_ptr<int>();
    auto* p_out     = reinterpret_cast<fp16*>(output.data_ptr());
    auto* p_zout    = reinterpret_cast<fp16*>(z_out.data_ptr());

    gdn_conv_fused_seq_host(
        p_qkvz, qkvz_stride0, p_cstate, p_cweight, p_cbias, p_csidx,
        p_alog, p_dtbias, p_ba, ba_stride0,
        p_sstate, p_ssidx, p_out, p_zout,
        (int)N, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_stride0, ssm_stride0,
        dpcpp_queue);

    return output;
}
