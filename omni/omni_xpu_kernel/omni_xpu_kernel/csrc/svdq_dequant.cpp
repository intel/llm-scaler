// ============================================================================
// SVDQuant W4A4 Dequantization - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance dequantization kernels for nunchaku SVDQuant format
//
// SVDQuant W4A4 data layout:
//   Weights:  packed uint8 [N, K/2]
//   Scales:   bf16/f16     [num_groups, N]  where num_groups = K / group_size
//   INT4 packing: low nibble = even index, high nibble = odd index
//   Signed INT4: range [-8, 7]
//   group_size = 64 (always)
//
// Dequantization formula:
//   result[n, k] = int4_val[n, k] * scale[k / group_size, n]
//
// Two kernels:
//   1. dequantize_svdq_w4  — Fused unpack + scale → bf16/f32 output
//      Input:  packed [N, K/2] uint8 + scales [G, N]
//      Output: [N, K] dequantized values
//
//   2. unpack_svdq_int4  — Unpack only (no scaling), used for activation path
//      Input:  packed [M, K/2] uint8
//      Output: [M, K] int8 values
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace svdq {

// ============================================================================
// Constants
// ============================================================================
constexpr int SVDQ_GROUP_SIZE = 64;    // nunchaku always uses group_size=64
constexpr int HALF_GROUP = SVDQ_GROUP_SIZE / 2;  // 32 packed bytes per group

// ============================================================================
// Kernel 1: dequantize_svdq_w4
// ============================================================================
// Fused INT4 unpack + per-group scale application.
// Each work-item processes one (row, group) pair.
//
// packed shape: [N, K/2] stored as contiguous uint8
// scales shape: [G, N]   stored as contiguous bf16/f16/f32
// output shape: [N, K]   stored as contiguous bf16/f16/f32
//
// For each row n, group g:
//   packed bytes are at packed[n, g*32 .. (g+1)*32 - 1]  (32 bytes = 64 nibbles)
//   scale is at scales[g, n]
//   output[n, g*64 .. (g+1)*64 - 1] = unpack(packed) * scale
// ============================================================================

#ifndef OMNI_SVDQ_DEQUANT_GROUPS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_DEQUANT_GROUPS_PER_WI 60
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_SVDQ_DEQUANT_GROUPS_PER_WI 1
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<
    typename OT,
    bool Signed = true,
    int GroupsPerWorkItem = OMNI_SVDQ_DEQUANT_GROUPS_PER_WI>
void dequantize_svdq_w4_kernel(
    const uint8_t* __restrict__ packed,
    const OT* __restrict__ scales,
    OT* __restrict__ output,
    const int64_t N,           // number of rows
    const int64_t K,           // full unpacked width
    const int64_t num_groups,  // K / 64
    const at::Device& device
) {
    const int64_t group_chunks =
        (num_groups + GroupsPerWorkItem - 1) / GroupsPerWorkItem;
    const int64_t total_items = N * group_chunks;
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (total_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= total_items) return;

                const int64_t row = gid / group_chunks;
                const int64_t first_grp =
                    (gid % group_chunks) * GroupsPerWorkItem;

                for (int local_grp = 0; local_grp < GroupsPerWorkItem;
                     ++local_grp) {
                    const int64_t grp = first_grp + local_grp;
                    if (grp >= num_groups) break;

                    // Read scale for this (group, row): scales[grp * N + row]
                    const OT scale_val = scales[grp * N + row];

                    // Read 32 packed bytes for this group
                    const uint8_t* src =
                        packed + row * (K / 2) + grp * HALF_GROUP;
                    OT* dst = output + row * K + grp * SVDQ_GROUP_SIZE;

                    // Process in 2 chunks of 16 bytes (32 elements each)
                    // to stay within ESIMD vector width limits
#pragma unroll
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        simd<uint8_t, 16> bytes =
                            block_load<uint8_t, 16>(src + chunk * 16);

                        // Unpack: low nibble = even index, high nibble = odd index
                        simd<uint8_t, 16> low_u = bytes & (uint8_t)0x0F;
                        simd<uint8_t, 16> high_u = (bytes >> 4) & (uint8_t)0x0F;

                        // Signed conversion: values 8-15 map to -8..-1
                        simd<int16_t, 16> low_s;
                        simd<int16_t, 16> high_s;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            int16_t lv = static_cast<int16_t>(low_u[i]);
                            int16_t hv = static_cast<int16_t>(high_u[i]);
                            low_s[i] = Signed && lv >= 8 ? (lv - 16) : lv;
                            high_s[i] = Signed && hv >= 8 ? (hv - 16) : hv;
                        }

                        // Interleave low and high nibbles to match nunchaku's
                        // stack([low, high], dim=-1).reshape(..., K) layout.
                        simd<float, 32> interleaved;
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            interleaved[i * 2] = static_cast<float>(low_s[i]);
                            interleaved[i * 2 + 1] = static_cast<float>(high_s[i]);
                        }

                        // Apply scale
                        float scale_f = static_cast<float>(scale_val);
                        interleaved = interleaved * scale_f;

                        // Store 32 output elements
                        OT* out_ptr = dst + chunk * 32;
                        if constexpr (std::is_same_v<OT, float>) {
                            block_store<float, 32>(out_ptr, interleaved);
                        } else if constexpr (std::is_same_v<OT, bf16>) {
                            simd<bf16, 32> result_bf = interleaved;
                            block_store<bf16, 32>(
                                reinterpret_cast<bf16*>(out_ptr), result_bf);
                        } else if constexpr (std::is_same_v<OT, fp16>) {
                            simd<fp16, 32> result_fp = interleaved;
                            block_store<fp16, 32>(
                                reinterpret_cast<fp16*>(out_ptr), result_fp);
                        }
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "dequantize_svdq_w4");
}

#if defined(OMNI_XPU_ARCH_PTL_H)
// The PTL signed path benefits from expressing sign extension and low/high
// interleave as ESIMD vector regions. Keep this as a distinct kernel type:
// its register footprint limits the work-group to 32, while the original
// unsigned path above remains faster at WG64.
template<
    typename OT,
    int GroupsPerWorkItem = OMNI_SVDQ_DEQUANT_GROUPS_PER_WI>
void dequantize_svdq_w4_signed_ptl_kernel(
    const uint8_t* __restrict__ packed,
    const OT* __restrict__ scales,
    OT* __restrict__ output,
    const int64_t N,
    const int64_t K,
    const int64_t num_groups,
    const at::Device& device) {
    const int64_t group_chunks =
        (num_groups + GroupsPerWorkItem - 1) / GroupsPerWorkItem;
    const int64_t total_items = N * group_chunks;
    constexpr int WG_SIZE = 32;
    const int64_t padded_size =
        (total_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= total_items) return;

                const int64_t row = gid / group_chunks;
                const int64_t first_grp =
                    (gid % group_chunks) * GroupsPerWorkItem;
                for (int local_grp = 0; local_grp < GroupsPerWorkItem;
                     ++local_grp) {
                    const int64_t grp = first_grp + local_grp;
                    if (grp >= num_groups) break;
                    const float scale_f =
                        static_cast<float>(scales[grp * N + row]);
                    const uint8_t* src =
                        packed + row * (K / 2) + grp * HALF_GROUP;
                    OT* dst = output + row * K + grp * SVDQ_GROUP_SIZE;

#pragma unroll
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        simd<uint8_t, 16> bytes =
                            block_load<uint8_t, 16>(src + chunk * 16);
                        simd<int16_t, 16> low_s = bytes & uint8_t(0x0F);
                        simd<int16_t, 16> high_s =
                            (bytes >> 4) & uint8_t(0x0F);
                        low_s.merge(low_s - 16, low_s >= 8);
                        high_s.merge(high_s - 16, high_s >= 8);

                        simd<float, 32> interleaved;
                        interleaved.template select<16, 2>(0) = low_s;
                        interleaved.template select<16, 2>(1) = high_s;
                        interleaved *= scale_f;

                        OT* out_ptr = dst + chunk * 32;
                        if constexpr (std::is_same_v<OT, float>) {
                            block_store<float, 32>(out_ptr, interleaved);
                        } else if constexpr (std::is_same_v<OT, bf16>) {
                            simd<bf16, 32> result = interleaved;
                            block_store<bf16, 32>(
                                reinterpret_cast<bf16*>(out_ptr), result);
                        } else if constexpr (std::is_same_v<OT, fp16>) {
                            simd<fp16, 32> result = interleaved;
                            block_store<fp16, 32>(
                                reinterpret_cast<fp16*>(out_ptr), result);
                        }
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "dequantize_svdq_w4_signed_ptl");
}
#endif

// ============================================================================
// Kernel 2: unpack_svdq_int4
// ============================================================================
// Pure INT4 unpack without scaling. Produces int8 output.
// Used for the activation quantization path where scales are applied separately.
//
// packed shape: [M, K/2] uint8
// output shape: [M, K]  int8
//
// Packing: low nibble (bits 0-3) = even element, high nibble (bits 4-7) = odd
// Signed: range [-8, 7]
// ============================================================================

#ifndef OMNI_SVDQ_UNPACK_COLS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_UNPACK_COLS_PER_WI 3840
#elif defined(OMNI_XPU_ARCH_BMG)
// Keep the established launch geometry until it is retuned on BMG.
#define OMNI_SVDQ_UNPACK_COLS_PER_WI 64
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

#ifndef OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION 64
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION 16
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

#ifndef OMNI_SVDQ_UNPACK_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_UNPACK_WG_SIZE 32
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_SVDQ_UNPACK_WG_SIZE 64
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<bool Signed = true, int COLS_PER_WI = OMNI_SVDQ_UNPACK_COLS_PER_WI>
void unpack_svdq_int4_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ output,
    const int64_t M,
    const int64_t K_half,  // K/2 = number of packed bytes per row
    const at::Device& device
) {
    const int64_t K = K_half * 2;
    // Amortize row/chunk indexing over a platform-tuned contiguous tile.
    const int64_t cols_per_wi_half = COLS_PER_WI / 2;
    const int64_t num_col_chunks = (K_half + cols_per_wi_half - 1) / cols_per_wi_half;
    const int64_t total_items = M * num_col_chunks;
    const bool aligned_rows = (K_half % 16) == 0;

    constexpr int WG_SIZE = OMNI_SVDQ_UNPACK_WG_SIZE;
    const int64_t padded_size = (total_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= total_items) return;

                const int64_t row = gid / num_col_chunks;
                const int64_t col_chunk = gid % num_col_chunks;
                const int64_t col_start = col_chunk * cols_per_wi_half;

                // Clamp to actual number of bytes remaining
                const int64_t n_bytes = std::min(cols_per_wi_half, K_half - col_start);

                const uint8_t* src = packed + row * K_half + col_start;
                int8_t* dst = output + row * K + col_start * 2;

                constexpr int BytesPerIteration =
                    OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION;
                for (int64_t b = 0; b < n_bytes; b += BytesPerIteration) {
                    const int64_t chunk_size = std::min(
                        static_cast<int64_t>(BytesPerIteration), n_bytes - b);

                    simd<uint8_t, BytesPerIteration> bytes;
                    if (chunk_size == BytesPerIteration && aligned_rows) {
                        bytes = block_load<uint8_t, BytesPerIteration>(src + b);
                    } else {
                        // Scalar fallback for an unaligned row or final tail.
#pragma unroll
                        for (int i = 0; i < BytesPerIteration; ++i) {
                            bytes[i] = (i < static_cast<int>(chunk_size))
                                           ? *(src + b + i) : uint8_t(0);
                        }
                    }

                    simd<uint8_t, BytesPerIteration> low_u =
                        bytes & uint8_t(0x0F);
                    simd<uint8_t, BytesPerIteration> high_u =
                        (bytes >> 4) & uint8_t(0x0F);

                    simd<int16_t, BytesPerIteration> low_wide = low_u;
                    simd<int16_t, BytesPerIteration> high_wide = high_u;
                    if constexpr (Signed) {
                        low_wide.merge(low_wide - 16, low_wide >= 8);
                        high_wide.merge(high_wide - 16, high_wide >= 8);
                    }
                    simd<int8_t, BytesPerIteration> low_s = low_wide;
                    simd<int8_t, BytesPerIteration> high_s = high_wide;

                    simd<int8_t, BytesPerIteration * 2> interleaved;
                    interleaved.template select<BytesPerIteration, 2>(0) = low_s;
                    interleaved.template select<BytesPerIteration, 2>(1) = high_s;

                    if (chunk_size == BytesPerIteration && aligned_rows) {
                        block_store<int8_t, BytesPerIteration * 2>(
                            dst + b * 2, interleaved);
                    } else {
                        for (int i = 0;
                             i < static_cast<int>(chunk_size) * 2; ++i) {
                            dst[b * 2 + i] = interleaved[i];
                        }
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "unpack_svdq_int4");
}

// ============================================================================
// Kernel 3: quantize_svdq_act_int4
// ============================================================================
// Quantize fp16/bf16/f32 activation with per-group absmax scaling.
// Used in the activation quantization path.
//
// input shape:  [M, K] bf16/f32
// output shape: [M, K/2] packed uint8
// scales shape: [num_groups, M] where num_groups = K/64
//
// Per-group quantization:
//   signed:   scale = absmax / 7,  q clamped symmetrically to [-7, 7]
//   unsigned: scale = absmax / 15, q clamped to [0, 15]
//   packed[m, k/2] = (q[even] & 0xF) | (q[odd] << 4)
// ============================================================================

#ifndef OMNI_SVDQ_QUANT_GROUPS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_QUANT_GROUPS_PER_WI 60
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_SVDQ_QUANT_GROUPS_PER_WI 1
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

template<
    typename IT,
    bool Unsigned = false,
    int GroupsPerWorkItem = OMNI_SVDQ_QUANT_GROUPS_PER_WI>
void quantize_svdq_act_int4_kernel(
    const IT* __restrict__ input,
    uint8_t* __restrict__ output,
    IT* __restrict__ scales,
    const int64_t M,
    const int64_t K,
    const int64_t num_groups,
    const at::Device& device
) {
    const int64_t group_chunks =
        (num_groups + GroupsPerWorkItem - 1) / GroupsPerWorkItem;
    const int64_t total_items = M * group_chunks;
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (total_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= total_items) return;

                const int64_t row = gid / group_chunks;
                const int64_t first_grp =
                    (gid % group_chunks) * GroupsPerWorkItem;

                for (int local_grp = 0; local_grp < GroupsPerWorkItem;
                     ++local_grp) {
                    const int64_t grp = first_grp + local_grp;
                    if (grp >= num_groups) break;

                    const IT* src =
                        input + row * K + grp * SVDQ_GROUP_SIZE;
                    uint8_t* dst =
                        output + row * (K / 2) + grp * HALF_GROUP;

                    // Load 64 input values in two chunks and find absmax.
                    simd<float, 32> vals_0, vals_1;
                    if constexpr (std::is_same_v<IT, float>) {
                        vals_0 = block_load<float, 32>(src);
                        vals_1 = block_load<float, 32>(src + 32);
                    } else if constexpr (std::is_same_v<IT, bf16>) {
                        simd<bf16, 32> bvals_0 = block_load<bf16, 32>(
                            reinterpret_cast<const bf16*>(src));
                        simd<bf16, 32> bvals_1 = block_load<bf16, 32>(
                            reinterpret_cast<const bf16*>(src + 32));
                        vals_0 = bvals_0;
                        vals_1 = bvals_1;
                    } else {
                        simd<fp16, 32> hvals_0 = block_load<fp16, 32>(
                            reinterpret_cast<const fp16*>(src));
                        simd<fp16, 32> hvals_1 = block_load<fp16, 32>(
                            reinterpret_cast<const fp16*>(src + 32));
                        vals_0 = hvals_0;
                        vals_1 = hvals_1;
                    }

                    simd<float, 32> abs_0 =
                        sycl::ext::intel::esimd::abs<float, 32>(vals_0);
                    simd<float, 32> abs_1 =
                        sycl::ext::intel::esimd::abs<float, 32>(vals_1);
                    simd<float, 32> abs_max =
                        sycl::ext::intel::esimd::max<float, 32>(abs_0, abs_1);
                    // Use the ESIMD reduction tree instead of a 32-step
                    // scalar dependency chain.
                    float group_max = hmax<float>(abs_max);

                    constexpr float qmax = Unsigned ? 15.0f : 7.0f;
                    constexpr float qmin = Unsigned ? 0.0f : -7.0f;
                    float scale = group_max / qmax;
                    if (scale < 1e-10f) scale = 1e-10f;
                    float rscale =
                        qmax /
                        (group_max < 1e-10f ? 1e-10f : group_max);
                    scales[grp * M + row] = static_cast<IT>(scale);

                    simd<float, 32> q_0 =
                        sycl::ext::intel::esimd::rnde<float, 32>(
                            vals_0 * rscale);
                    simd<float, 32> q_1 =
                        sycl::ext::intel::esimd::rnde<float, 32>(
                            vals_1 * rscale);
                    simd<float, 32> clamp_lo(qmin);
                    simd<float, 32> clamp_hi(qmax);
                    q_0 = sycl::ext::intel::esimd::max<float, 32>(
                        sycl::ext::intel::esimd::min<float, 32>(
                            q_0, clamp_hi),
                        clamp_lo);
                    q_1 = sycl::ext::intel::esimd::max<float, 32>(
                        sycl::ext::intel::esimd::min<float, 32>(
                            q_1, clamp_hi),
                        clamp_lo);

                    // Pack even elements into low nibbles and odd elements
                    // into high nibbles.
                    simd<uint8_t, 16> packed_0, packed_1;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int8_t even_0 = static_cast<int8_t>(q_0[i * 2]);
                        int8_t odd_0 = static_cast<int8_t>(q_0[i * 2 + 1]);
                        packed_0[i] =
                            static_cast<uint8_t>(even_0 & 0x0F) |
                            static_cast<uint8_t>((odd_0 & 0x0F) << 4);

                        int8_t even_1 = static_cast<int8_t>(q_1[i * 2]);
                        int8_t odd_1 = static_cast<int8_t>(q_1[i * 2 + 1]);
                        packed_1[i] =
                            static_cast<uint8_t>(even_1 & 0x0F) |
                            static_cast<uint8_t>((odd_1 & 0x0F) << 4);
                    }

                    simd<uint32_t, 16> store_offsets;
#pragma unroll
                    for (int i = 0; i < 16; ++i) store_offsets[i] = i;
                    scatter<uint8_t, 16>(dst, store_offsets, packed_0);
                    scatter<uint8_t, 16>(
                        dst + 16, store_offsets, packed_1);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "quantize_svdq_act_int4");
}

// ============================================================================
// Public API
// ============================================================================

torch::Tensor dequantize_svdq_w4(
    const torch::Tensor& packed,     // [N, K/2] uint8
    const torch::Tensor& scales,     // [num_groups, N]
    torch::ScalarType out_dtype
) {
    TORCH_CHECK(packed.is_contiguous(), "packed tensor must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales tensor must be contiguous");
    TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D [N, K/2]");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D [num_groups, N]");

    const int64_t N = packed.size(0);
    const int64_t K_half = packed.size(1);
    const int64_t K = K_half * 2;
    const int64_t num_groups = scales.size(0);

    TORCH_CHECK(scales.size(1) == N,
        "scales dim 1 (", scales.size(1), ") must match packed dim 0 (N=", N, ")");
    TORCH_CHECK(K % SVDQ_GROUP_SIZE == 0,
        "K (", K, ") must be divisible by group_size (", SVDQ_GROUP_SIZE, ")");
    TORCH_CHECK(K / SVDQ_GROUP_SIZE == num_groups,
        "num_groups mismatch: K/64=", K / SVDQ_GROUP_SIZE, " vs scales dim 0=", num_groups);

    auto output = torch::empty({N, K},
        torch::TensorOptions().dtype(out_dtype).device(packed.device()));

    // Convert scales to match output dtype if needed
    auto scales_cast = scales.to(out_dtype).contiguous();

    const uint8_t* packed_ptr = packed.data_ptr<uint8_t>();

    if (out_dtype == torch::kFloat32) {
#if defined(OMNI_XPU_ARCH_PTL_H)
        dequantize_svdq_w4_signed_ptl_kernel<float>(
#else
        dequantize_svdq_w4_kernel<float>(
#endif
            packed_ptr, scales_cast.data_ptr<float>(),
            output.data_ptr<float>(), N, K, num_groups, packed.device());
    } else if (out_dtype == torch::kBFloat16) {
#if defined(OMNI_XPU_ARCH_PTL_H)
        dequantize_svdq_w4_signed_ptl_kernel<bf16>(
#else
        dequantize_svdq_w4_kernel<bf16>(
#endif
            packed_ptr, reinterpret_cast<const bf16*>(scales_cast.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()), N, K, num_groups, packed.device());
    } else if (out_dtype == torch::kFloat16) {
#if defined(OMNI_XPU_ARCH_PTL_H)
        dequantize_svdq_w4_signed_ptl_kernel<fp16>(
#else
        dequantize_svdq_w4_kernel<fp16>(
#endif
            packed_ptr, reinterpret_cast<const fp16*>(scales_cast.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()), N, K, num_groups, packed.device());
    } else {
        TORCH_CHECK(false, "Unsupported output dtype: ", out_dtype);
    }

    return output;
}

torch::Tensor dequantize_svdq_u4(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    torch::ScalarType out_dtype
) {
    TORCH_CHECK(packed.is_contiguous(), "packed tensor must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales tensor must be contiguous");
    TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2, "packed and scales must be 2D");
    const int64_t M = packed.size(0);
    const int64_t K = packed.size(1) * 2;
    const int64_t groups = scales.size(0);
    TORCH_CHECK(scales.size(1) == M && groups == K / SVDQ_GROUP_SIZE,
                "unsigned activation scale shape mismatch");
    auto output = torch::empty({M, K}, torch::TensorOptions().dtype(out_dtype).device(packed.device()));
    auto scales_cast = scales.to(out_dtype).contiguous();
    if (out_dtype == torch::kFloat32) {
        dequantize_svdq_w4_kernel<float, false>(packed.data_ptr<uint8_t>(), scales_cast.data_ptr<float>(), output.data_ptr<float>(), M, K, groups, packed.device());
    } else if (out_dtype == torch::kBFloat16) {
        dequantize_svdq_w4_kernel<bf16, false>(packed.data_ptr<uint8_t>(), reinterpret_cast<const bf16*>(scales_cast.data_ptr()), reinterpret_cast<bf16*>(output.data_ptr()), M, K, groups, packed.device());
    } else if (out_dtype == torch::kFloat16) {
        dequantize_svdq_w4_kernel<fp16, false>(packed.data_ptr<uint8_t>(), reinterpret_cast<const fp16*>(scales_cast.data_ptr()), reinterpret_cast<fp16*>(output.data_ptr()), M, K, groups, packed.device());
    } else {
        TORCH_CHECK(false, "Unsupported output dtype: ", out_dtype);
    }
    return output;
}


torch::Tensor unpack_svdq_int4(
    const torch::Tensor& packed,     // [M, K/2] uint8
    bool is_signed                   // true for signed [-8,7], false for unsigned [0,15]
) {
    TORCH_CHECK(packed.is_contiguous(), "packed tensor must be contiguous");
    TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D [M, K/2]");

    const int64_t M = packed.size(0);
    const int64_t K_half = packed.size(1);

    auto output = torch::empty({M, K_half * 2},
        torch::TensorOptions().dtype(torch::kInt8).device(packed.device()));

    if (is_signed) {
        unpack_svdq_int4_kernel<true>(packed.data_ptr<uint8_t>(), output.data_ptr<int8_t>(), M, K_half, packed.device());
    } else {
        unpack_svdq_int4_kernel<false>(packed.data_ptr<uint8_t>(), output.data_ptr<int8_t>(), M, K_half, packed.device());
    }

    return output;
}


std::tuple<torch::Tensor, torch::Tensor> quantize_svdq_act_int4(
    const torch::Tensor& input,      // [M, K] bf16/f32
    int64_t group_size
) {
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(group_size == SVDQ_GROUP_SIZE,
        "Only group_size=64 is supported, got ", group_size);

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K % SVDQ_GROUP_SIZE == 0,
        "K (", K, ") must be divisible by group_size (", SVDQ_GROUP_SIZE, ")");

    const int64_t num_groups = K / SVDQ_GROUP_SIZE;

    auto packed = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    auto scales_out = torch::empty({num_groups, M}, input.options());

    auto input_dtype = input.scalar_type();
    if (input_dtype == torch::kFloat32) {
        quantize_svdq_act_int4_kernel<float>(
            input.data_ptr<float>(),
            packed.data_ptr<uint8_t>(),
            scales_out.data_ptr<float>(),
            M, K, num_groups, input.device());
    } else if (input_dtype == torch::kBFloat16) {
        quantize_svdq_act_int4_kernel<bf16>(
            reinterpret_cast<const bf16*>(input.data_ptr()),
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<bf16*>(scales_out.data_ptr()),
            M, K, num_groups, input.device());
    } else if (input_dtype == torch::kFloat16) {
        quantize_svdq_act_int4_kernel<fp16>(
            reinterpret_cast<const fp16*>(input.data_ptr()),
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<fp16*>(scales_out.data_ptr()),
            M, K, num_groups, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported input dtype: ", input_dtype);
    }

    return std::make_tuple(packed, scales_out);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_svdq_act_uint4(
    const torch::Tensor& input,
    int64_t group_size
) {
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(group_size == SVDQ_GROUP_SIZE, "Only group_size=64 is supported");
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K % SVDQ_GROUP_SIZE == 0, "K must be divisible by 64");
    const int64_t groups = K / SVDQ_GROUP_SIZE;
    auto packed = torch::empty({M, K / 2}, torch::TensorOptions().dtype(torch::kByte).device(input.device()));
    auto scales_out = torch::empty({groups, M}, input.options());
    if (input.scalar_type() == torch::kFloat32) {
        quantize_svdq_act_int4_kernel<float, true>(input.data_ptr<float>(), packed.data_ptr<uint8_t>(), scales_out.data_ptr<float>(), M, K, groups, input.device());
    } else if (input.scalar_type() == torch::kBFloat16) {
        quantize_svdq_act_int4_kernel<bf16, true>(reinterpret_cast<const bf16*>(input.data_ptr()), packed.data_ptr<uint8_t>(), reinterpret_cast<bf16*>(scales_out.data_ptr()), M, K, groups, input.device());
    } else if (input.scalar_type() == torch::kFloat16) {
        quantize_svdq_act_int4_kernel<fp16, true>(reinterpret_cast<const fp16*>(input.data_ptr()), packed.data_ptr<uint8_t>(), reinterpret_cast<fp16*>(scales_out.data_ptr()), M, K, groups, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }
    return {packed, scales_out};
}


}  // namespace svdq
}  // namespace omni_xpu
