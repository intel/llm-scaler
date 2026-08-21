#pragma once

#include "utils.h"

template<int VL>
struct DualRmsNormResidualScalarKernel {
    const fp16* x1;
    const fp16* weight1;
    const fp16* x2;
    const fp16* weight2;
    const fp16* weight3;
    const fp16* residual;
    fp16* output;
    int rows;
    int hidden_size;
    float eps1;
    float eps2;
    float eps3;
    float scalar;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = item.get_group(0);
        if (row >= rows) {
            return;
        }
        const fp16* row_x1 = x1 + static_cast<size_t>(row) * hidden_size;
        const fp16* row_x2 = x2 + static_cast<size_t>(row) * hidden_size;
        const fp16* row_residual =
            residual + static_cast<size_t>(row) * hidden_size;
        fp16* row_output =
            output + static_cast<size_t>(row) * hidden_size;
        const int chunks = hidden_size / VL;
        float sum_sq1 = 0.0f;
        float sum_sq2 = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values1 =
                block_load<fp16, VL>(row_x1 + offset);
            simd<float, VL> values2 =
                block_load<fp16, VL>(row_x2 + offset);
            sum_sq1 += reduce<float>(values1 * values1, std::plus<>());
            sum_sq2 += reduce<float>(values2 * values2, std::plus<>());
        }
        const float inv_rms1 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq1 / hidden_size + eps1))[0];
        const float inv_rms2 = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq2 / hidden_size + eps2))[0];

        float combined_sum_sq = 0.0f;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values1 =
                block_load<fp16, VL>(row_x1 + offset);
            simd<float, VL> values2 =
                block_load<fp16, VL>(row_x2 + offset);
            simd<float, VL> scale1 =
                block_load<fp16, VL>(weight1 + offset);
            simd<float, VL> scale2 =
                block_load<fp16, VL>(weight2 + offset);
            simd<float, VL> combined =
                values1 * inv_rms1 * scale1
                + values2 * inv_rms2 * scale2;
            combined_sum_sq +=
                reduce<float>(combined * combined, std::plus<>());
        }
        const float combined_inv_rms =
            sycl::ext::intel::esimd::rsqrt(
                simd<float, 8>(
                    combined_sum_sq / hidden_size + eps3))[0];

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int offset = chunk * VL;
            simd<float, VL> values1 =
                block_load<fp16, VL>(row_x1 + offset);
            simd<float, VL> values2 =
                block_load<fp16, VL>(row_x2 + offset);
            simd<float, VL> scale1 =
                block_load<fp16, VL>(weight1 + offset);
            simd<float, VL> scale2 =
                block_load<fp16, VL>(weight2 + offset);
            simd<float, VL> scale3 =
                block_load<fp16, VL>(weight3 + offset);
            simd<float, VL> residual_values =
                block_load<fp16, VL>(row_residual + offset);
            simd<float, VL> combined =
                values1 * inv_rms1 * scale1
                + values2 * inv_rms2 * scale2;
            simd<float, VL> result =
                (combined * combined_inv_rms * scale3 + residual_values)
                * scalar;
            block_store<fp16, VL>(
                row_output + offset, simd<fp16, VL>(result));
        }
    }
};

inline void dual_rmsnorm_residual_scalar_host(
    const fp16* x1,
    const fp16* weight1,
    const fp16* x2,
    const fp16* weight2,
    const fp16* weight3,
    const fp16* residual,
    fp16* output,
    int rows,
    int hidden_size,
    float eps1,
    float eps2,
    float eps3,
    float scalar,
    sycl::queue& queue) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(rows, 1),
            DualRmsNormResidualScalarKernel<256>{
                x1,
                weight1,
                x2,
                weight2,
                weight3,
                residual,
                output,
                rows,
                hidden_size,
                eps1,
                eps2,
                eps3,
                scalar});
    });
}
