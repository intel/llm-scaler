// ============================================================================
// Normalization Kernels - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance RMSNorm and LayerNorm using Intel ESIMD
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using ST = at::ScalarType;

using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {

// ============================================================================
// RMSNorm Kernel
// ============================================================================
template<typename IT, const int GS, const int BS>
void rms_norm_kernel(
    const void* weight_ptr,
    const void* input_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    assert(hidden_size % BS == 0);
    assert(hidden_size <= 8 * 1024);

    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    const int acc_offset = hidden_size * sizeof(IT);

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + GS * sizeof(float)>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                const IT* weight = (const IT*)weight_ptr;
                const IT* input = (const IT*)input_ptr + hidden_size * (size_t)rid;
                IT* output = (IT*)output_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                simd<float, BS> accv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<IT, BS> xv = block_load<IT, BS>(input + i * BS);
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), xv);
                    simd<float, BS> xv_f32 = xv;
                    accv += xv_f32 * xv_f32;
                }
                float acc = sycl::ext::intel::esimd::detail::sum<float, float, BS>(accv) / hidden_size;
                slm_block_store<float, 1>(acc_offset + tid * sizeof(float), acc);

                barrier();

                simd<float, GS> accs = slm_block_load<float, GS>(acc_offset);
                float mean = sycl::ext::intel::esimd::detail::sum<float, float, GS>(accs);
                float scale = rsqrt(mean + eps);

                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                    simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                    simd<IT, BS> result = xv * scale * yv;
                    block_store<IT, BS>(output + i * BS, result);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "rms_norm_esimd");
}

// ============================================================================
// LayerNorm Kernel
// ============================================================================
template<typename IT, const int GS, const int BS>
void layer_norm_kernel(
    const void* input_ptr,
    const uint64_t weight_ptr,
    const uint64_t bias_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    assert(hidden_size % BS == 0);
    assert(hidden_size <= 8 * 1024);

    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    const int mean_offset = hidden_size * sizeof(IT);
    const int var_offset = mean_offset + GS * sizeof(float);

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + 2 * GS * sizeof(float)>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                const IT* input = (const IT*)input_ptr + hidden_size * (size_t)rid;
                IT* output = (IT*)output_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                // Compute mean
                simd<float, BS> sumv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<IT, BS> xv = block_load<IT, BS>(input + i * BS);
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), xv);
                    simd<float, BS> xv_f32 = xv;
                    sumv += xv_f32;
                }
                float par_mean = sycl::ext::intel::esimd::detail::sum<float, float, BS>(sumv) / hidden_size;
                slm_block_store<float, 1>(mean_offset + tid * sizeof(float), par_mean);

                barrier();

                simd<float, GS> means = slm_block_load<float, GS>(mean_offset);
                float mean = sycl::ext::intel::esimd::detail::sum<float, float, GS>(means);

                // Compute variance
                simd<float, BS> varv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                    varv += (xv - mean) * (xv - mean);
                }
                float par_var = sycl::ext::intel::esimd::detail::sum<float, float, BS>(varv) / hidden_size;
                slm_block_store<float, 1>(var_offset + tid * sizeof(float), par_var);

                barrier();

                simd<float, GS> vars = slm_block_load<float, GS>(var_offset);
                float var = sycl::ext::intel::esimd::detail::sum<float, float, GS>(vars);
                float scale = rsqrt(var + eps);

                // Normalize and apply weight/bias
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                    simd<float, BS> result = (xv - mean) * scale;

                    if (weight_ptr != 0) {
                        simd<float, BS> yv = block_load<IT, BS>((const IT*)weight_ptr + i * BS);
                        result = result * yv;
                    }

                    if (bias_ptr != 0) {
                        simd<float, BS> bv = block_load<IT, BS>((const IT*)bias_ptr + i * BS);
                        result = result + bv;
                    }

                    block_store<IT, BS>(output + i * BS, result);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "layer_norm_esimd");
}

// ============================================================================
// Public C++ API
// ============================================================================

torch::Tensor rms_norm(
    torch::Tensor weight,
    torch::Tensor input,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D tensor [hidden_size]");
    TORCH_CHECK(weight.size(0) == input.size(1), "Weight size must match hidden_size");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "Input and weight dtype must match");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int64_t input_size = input.size(0);
    int64_t hidden_size = input.size(1);

    auto output = torch::empty({input_size, hidden_size},
        torch::device(input.device()).dtype(input.dtype()));

    auto func = [&]() {
        switch (input.scalar_type()) {
            case ST::Float: return rms_norm_kernel<float, 32, 32>;
            case ST::Half: return rms_norm_kernel<fp16, 32, 32>;
            case ST::BFloat16: return rms_norm_kernel<bf16, 32, 32>;
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    }();

    func(
        weight.data_ptr(), input.data_ptr(), output.data_ptr(),
        static_cast<float>(eps), input_size, hidden_size,
        input.device()
    );

    return output;
}

torch::Tensor layer_norm(
    torch::Tensor input,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> bias,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    int64_t input_size = input.size(0);
    int64_t hidden_size = input.size(1);

    if (weight.has_value()) {
        TORCH_CHECK(weight->numel() == hidden_size, "Weight size must match hidden_size");
        TORCH_CHECK(weight->scalar_type() == input.scalar_type(), "Weight dtype must match input");
        TORCH_CHECK(weight->is_contiguous(), "Weight must be contiguous");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == hidden_size, "Bias size must match hidden_size");
        TORCH_CHECK(bias->scalar_type() == input.scalar_type(), "Bias dtype must match input");
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }

    auto output = torch::empty({input_size, hidden_size},
        torch::device(input.device()).dtype(input.dtype()));

    auto func = [&]() {
        switch (input.scalar_type()) {
            case ST::Float: return layer_norm_kernel<float, 32, 32>;
            case ST::Half: return layer_norm_kernel<fp16, 32, 32>;
            case ST::BFloat16: return layer_norm_kernel<bf16, 32, 32>;
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    }();

    const uint64_t weight_ptr = weight.has_value() ? (uint64_t)(weight->data_ptr()) : 0;
    const uint64_t bias_ptr = bias.has_value() ? (uint64_t)(bias->data_ptr()) : 0;

    func(
        input.data_ptr(), weight_ptr, bias_ptr, output.data_ptr(),
        static_cast<float>(eps), input_size, hidden_size,
        input.device()
    );

    return output;
}

}  // namespace norm
}  // namespace omni_xpu
