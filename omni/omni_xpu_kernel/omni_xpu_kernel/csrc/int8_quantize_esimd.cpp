// ============================================================================
// ESIMD Fused INT8 Quantization Kernels
// ============================================================================
// High-performance single-pass quantization kernels that fuse:
//   absmax reduction + scale computation + divide + round + clamp + cast
// into minimal kernel launches, eliminating Python-level multi-op overhead.
//
// Kernels:
//   1. quantize_int8_rowwise_esimd: Per-row quantization for activations
//      Input:  [M, K] bf16/f16
//      Output: [M, K] int8 + [M, 1] float32 scales
//
//   2. quantize_int8_tensorwise_esimd: Single-scale tensor quantization
//      Input:  [M, K] bf16/f16
//      Output: [M, K] int8 + scalar float32 scale
//
// Design: Each work-item processes one row (K elements). For typical ComfyUI
// shapes (K=4096), this means each WI reads 4096 bf16 values (8KB), computes
// absmax via ESIMD SIMD reduction, then quantizes in vectorized chunks.
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {

// ============================================================================
// Kernel: Fused per-row INT8 quantization
// ============================================================================
// Each work-item processes one full row of K elements:
//   1. Compute absmax across the row (vectorized reduce)
//   2. Compute scale = absmax / 127
//   3. Quantize: round(x / scale).clamp(-128, 127) → int8
//
// For K ≤ 8192, single WI per row is efficient (memory-bound, good cache line use).
// For larger K, could split into multi-WI per row with sub-group reduction.
// ============================================================================

template <typename InputT>
void quantize_int8_rowwise_esimd_kernel(
    const InputT* __restrict__ input,  // [M, K]
    int8_t* __restrict__ output,       // [M, K]
    float* __restrict__ scales,        // [M]
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    // Single-pass cooperative design (mirrors rms_norm_kernel pattern):
    //   GS work-items per row collaborate via SLM.
    //   Pass 1: each WI reads its K/GS portion from HBM → stores to SLM → computes partial absmax
    //   Barrier + cross-WI max reduction in SLM
    //   Pass 2: each WI reads from SLM (not HBM) → quantize → write int8 to HBM
    //
    // HBM traffic: K*2 bytes read + K*1 bytes write = 3 B/elem (single pass)
    // vs 2-pass: K*2*2 + K*1 = 5 B/elem → 40% savings
    constexpr int GS = 4;          // Work-items per row (cooperative group size)
    constexpr int BS = 32;         // SIMD block size per load/store

    const int64_t nb = K / BS;     // Total blocks per row
    const int64_t sub_nb = nb / GS;
    const int64_t rem_nb = nb % GS;

    // SLM layout: [K * sizeof(InputT)] for row data + [GS * sizeof(float)] for partial max
    constexpr int slm_max_offset_factor = 8 * 1024;  // max K=8192 supported in SLM for bf16
    // For K up to 8192 bf16: 16KB data + 16 bytes accumulators
    // For K=12288: exceeds 8K*2=16KB limit, fall back to 2-pass for large K

    if (K <= 8192 && K % BS == 0) {
        // Fast single-pass path with SLM
        const int slm_data_bytes = static_cast<int>(K * sizeof(InputT));
        constexpr int slm_acc_align = ((GS * (int)sizeof(float) + 15) / 16) * 16;

        sycl::range<2> global_size(M, GS);
        sycl::range<2> local_size(1, GS);

        auto cgf = [&](sycl::handler& handle) {
            handle.parallel_for(
                sycl::nd_range<2>(global_size, local_size),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    slm_init<8 * 1024 * sizeof(InputT) + slm_acc_align>();

                    const int64_t row = item.get_global_id(0);
                    const int tid = item.get_local_id(1);

                    const InputT* row_ptr = input + row * K;
                    int8_t* out_ptr = output + row * K;

                    // Compute this WI's block range
                    const int64_t start_blk = sub_nb * tid + std::min((int64_t)tid, rem_nb);
                    const int64_t end_blk = start_blk + sub_nb + (tid < rem_nb);

                    // Pass 1: Read from HBM → store to SLM → compute partial absmax
                    simd<float, BS> max_vec(0.0f);
                    for (int64_t i = start_blk; i < end_blk; ++i) {
                        simd<InputT, BS> xv = block_load<InputT, BS>(row_ptr + i * BS);
                        // Cache in SLM
                        slm_block_store<InputT, BS>(
                            static_cast<uint32_t>(i * BS * sizeof(InputT)), xv);
                        // Partial absmax
                        simd<float, BS> xf = xv;
                        simd<float, BS> ax = __ESIMD_NS::abs<float, BS>(xf);
                        max_vec = __ESIMD_NS::max<float, BS>(max_vec, ax);
                    }

                    // Reduce partial max to scalar within this WI
                    // Tree reduction for partial max within WI (BS=32 → scalar)
                    simd<float, 16> pm16 = __ESIMD_NS::max<float, 16>(
                        max_vec.template select<16, 1>(0), max_vec.template select<16, 1>(16));
                    simd<float, 8> pm8 = __ESIMD_NS::max<float, 8>(
                        pm16.template select<8, 1>(0), pm16.template select<8, 1>(8));
                    simd<float, 4> pm4 = __ESIMD_NS::max<float, 4>(
                        pm8.template select<4, 1>(0), pm8.template select<4, 1>(4));
                    simd<float, 2> pm2 = __ESIMD_NS::max<float, 2>(
                        pm4.template select<2, 1>(0), pm4.template select<2, 1>(2));
                    float partial_max = (pm2[0] > pm2[1]) ? pm2[0] : pm2[1];

                    // Store partial max to SLM for cross-WI reduction
                    const uint32_t acc_offset = static_cast<uint32_t>(K * sizeof(InputT));
                    slm_block_store<float, 1>(acc_offset + tid * sizeof(float), partial_max);
                    barrier();

                    // Cross-WI reduction: read all GS partial maxes from SLM
                    simd<float, GS> all_maxes = slm_block_load<float, GS>(acc_offset);
                    // Cross-WI reduction: GS=4 partial maxes → global max
                    float row_max = all_maxes[0];
                    for (int g = 1; g < GS; ++g) {
                        if (all_maxes[g] > row_max) row_max = all_maxes[g];
                    }

                    float scale = row_max / 127.0f;
                    if (scale < 1e-30f) scale = 1e-30f;
                    float inv_scale = 1.0f / scale;

                    // Only WI 0 writes the scale
                    if (tid == 0) scales[row] = scale;

                    // Pass 2: Read from SLM → quantize → write int8 to HBM
                    for (int64_t i = start_blk; i < end_blk; ++i) {
                        simd<InputT, BS> xv = slm_block_load<InputT, BS>(
                            static_cast<uint32_t>(i * BS * sizeof(InputT)));
                        simd<float, BS> xf = xv;
                        simd<float, BS> scaled = xf * inv_scale;
                        simd<float, BS> rounded = __ESIMD_NS::rnde<float, BS>(scaled);
                        rounded = __ESIMD_NS::max<float, BS>(rounded, simd<float, BS>(-128.0f));
                        rounded = __ESIMD_NS::min<float, BS>(rounded, simd<float, BS>(127.0f));

                        simd<int32_t, BS> iv = rounded;
                        simd<int8_t, BS> i8v;
                        #pragma unroll
                        for (int j = 0; j < BS; ++j) i8v[j] = static_cast<int8_t>(iv[j]);
                        i8v.copy_to(out_ptr + i * BS);
                    }
                }
            );
        };
        utils::submit_kernel(cgf, device, "quantize_int8_rowwise_esimd_slm");

    } else {
        // Fallback 2-pass for large K or non-aligned K
        constexpr int WG_SIZE = 32;
        const int64_t padded = (M + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

        auto cgf = [&](sycl::handler& handle) {
            handle.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG_SIZE)),
                [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    const int64_t row = item.get_global_id(0);
                    if (row >= M) return;

                    const InputT* row_ptr = input + row * K;
                    int8_t* out_ptr = output + row * K;

                    // Pass 1: absmax
                    simd<float, BS> max_vec(0.0f);
                    int64_t k = 0;
                    for (; k + BS <= K; k += BS) {
                        simd<InputT, BS> data;
                        data.copy_from(row_ptr + k);
                        simd<float, BS> df = data;
                        max_vec = __ESIMD_NS::max<float, BS>(max_vec, __ESIMD_NS::abs<float, BS>(df));
                    }
                    for (int64_t j = k; j < K; ++j) {
                        float v = static_cast<float>(row_ptr[j]);
                        float av = (v >= 0) ? v : -v;
                        if (av > max_vec[0]) max_vec[0] = av;
                    }

                    // Tree reduction for absmax (fallback path)
                    simd<float, 16> fm16 = __ESIMD_NS::max<float, 16>(
                        max_vec.template select<16, 1>(0), max_vec.template select<16, 1>(16));
                    simd<float, 8> fm8 = __ESIMD_NS::max<float, 8>(
                        fm16.template select<8, 1>(0), fm16.template select<8, 1>(8));
                    simd<float, 4> fm4 = __ESIMD_NS::max<float, 4>(
                        fm8.template select<4, 1>(0), fm8.template select<4, 1>(4));
                    simd<float, 2> fm2 = __ESIMD_NS::max<float, 2>(
                        fm4.template select<2, 1>(0), fm4.template select<2, 1>(2));
                    float row_max = (fm2[0] > fm2[1]) ? fm2[0] : fm2[1];
                    float scale = row_max / 127.0f;
                    if (scale < 1e-30f) scale = 1e-30f;
                    float inv_scale = 1.0f / scale;
                    scales[row] = scale;

                    // Pass 2: quantize
                    k = 0;
                    for (; k + BS <= K; k += BS) {
                        simd<InputT, BS> data;
                        data.copy_from(row_ptr + k);
                        simd<float, BS> df = data;
                        simd<float, BS> sc = df * inv_scale;
                        simd<float, BS> rd = __ESIMD_NS::rnde<float, BS>(sc);
                        rd = __ESIMD_NS::max<float, BS>(rd, simd<float, BS>(-128.0f));
                        rd = __ESIMD_NS::min<float, BS>(rd, simd<float, BS>(127.0f));
                        simd<int32_t, BS> iv = rd;
                        simd<int8_t, BS> i8v;
                        #pragma unroll
                        for (int j = 0; j < BS; ++j) i8v[j] = static_cast<int8_t>(iv[j]);
                        i8v.copy_to(out_ptr + k);
                    }
                    for (int64_t j = k; j < K; ++j) {
                        float v = static_cast<float>(row_ptr[j]) * inv_scale;
                        int32_t rv = static_cast<int32_t>(v + (v >= 0 ? 0.5f : -0.5f));
                        out_ptr[j] = static_cast<int8_t>(rv < -128 ? -128 : (rv > 127 ? 127 : rv));
                    }
                }
            );
        };
        utils::submit_kernel(cgf, device, "quantize_int8_rowwise_esimd_2pass");
    }
}

// ============================================================================
// Public C++ API for ESIMD quantization
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kHalf,
        "x must be bf16 or f16, got ", x.scalar_type()
    );

    x = x.contiguous();
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);

    torch::Tensor output = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    if (x.scalar_type() == torch::kBFloat16) {
        quantize_int8_rowwise_esimd_kernel<bf16>(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    } else {
        quantize_int8_rowwise_esimd_kernel<fp16>(
            reinterpret_cast<const fp16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    }

    // Reshape scales to [..., 1] to match PyTorch convention
    auto scale_shape = x.sizes().vec();
    scale_shape.back() = 1;
    return {output.reshape(x.sizes()), scales.reshape(scale_shape)};
}

}  // namespace int8_ops
}  // namespace omni_xpu
