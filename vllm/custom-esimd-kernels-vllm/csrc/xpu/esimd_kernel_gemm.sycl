/* esimd_kernel_gemm.sycl — Compilation unit for FP8 GEMM (M>1) ESIMD kernels.
 * Compiled with AOT for BMG (no doubleGRF). Uses DPAS for M>=2.
 * Separate from esimd_kernel.sycl to avoid ODR conflicts with fp8_GEMV_v2.h
 * (both define fp8_dequant, select_vl_ks with identical implementations).
 */
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/python.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "esimd_kernels/fp8_GEMM_pert.h"

#define EXTRACT_PTR(var, tensor) auto var = reinterpret_cast<uint8_t*>(tensor.data_ptr())

inline int get_fp8_mode(const at::Tensor& weight) {
    return (weight.scalar_type() == at::ScalarType::Float8_e5m2) ? 1 : 0;
}

static inline sycl::queue& get_device_queue(const at::Tensor& tensor) {
    return c10::xpu::getCurrentXPUStream(tensor.device().index()).queue();
}

// ---- FP8 GEMM per-tensor scale: input [M, K], weight [N, K], output [M, N] ----
// Auto-dispatches: M=1-3 → batched GEMV, M>=2 E4M3 → DPAS V9, else → WS
at::Tensor esimd_gemm_fp8_pert(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor output) {
    int64_t M = input.size(0);
    int64_t K = weight.size(1);
    int64_t N = weight.size(0);
    EXTRACT_PTR(p_in, input); EXTRACT_PTR(p_w, weight);
    EXTRACT_PTR(p_sc, weight_scale); EXTRACT_PTR(p_out, output);
    auto& dpcpp_queue = get_device_queue(input);
    GEMM_fp8_pert_dispatch(
        reinterpret_cast<const fp16*>(p_in),
        reinterpret_cast<const uint8_t*>(p_w),
        reinterpret_cast<const float*>(p_sc),
        reinterpret_cast<fp16*>(p_out),
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        get_fp8_mode(weight), dpcpp_queue);
    return output;
}
