// ============================================================================
// GGUF Dequantization - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance dequantization kernels for ComfyUI-GGUF
// 
// Supported formats (matching ComfyUI-GGUF layout):
//   Q4_0: Block=32, Size=18 bytes (2 scale + 16 data)
//   Q8_0: Block=32, Size=34 bytes (2 scale + 32 data)
//   Q4_K: Block=256, Size=144 bytes (2+2+12+128)
//   Q6_K: Block=256, Size=210 bytes (128+64+16+2)
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace gguf {

// ============================================================================
// Constants
// ============================================================================

// Q4_0: 18 bytes per block, 32 elements
constexpr int Q4_0_BLOCK_SIZE = 18;
constexpr int Q4_0_QK = 32;

// Q8_0: 34 bytes per block, 32 elements
constexpr int Q8_0_BLOCK_SIZE = 34;
constexpr int Q8_0_QK = 32;

// Q4_K: 144 bytes per block, 256 elements
constexpr int Q4_K_BLOCK_SIZE = 144;
constexpr int Q4_K_QK = 256;
constexpr int K_SCALE_SIZE = 12;

// Q6_K: 210 bytes per block, 256 elements
constexpr int Q6_K_BLOCK_SIZE = 210;
constexpr int Q6_K_QK = 256;

// ============================================================================
// Q4_0 Kernel (ComfyUI Sequential Layout)
// Output layout: [0-15]=low nibbles, [16-31]=high nibbles
// ============================================================================
template<typename OT, int SBS = 16>
void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    constexpr int WG_SIZE = 64;
    const int64_t n_work_items = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_size = (n_work_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_work_items) return;
                
                const int64_t start_block = gid * SBS;
                const int64_t end_block = std::min(start_block + SBS, n_blocks);
                
                simd<uint32_t, 16> offsets;
                #pragma unroll
                for (int i = 0; i < 16; ++i) offsets[i] = i;
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q4_0_BLOCK_SIZE;
                    OT* block_dst = dst + blk * Q4_0_QK;
                    
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    simd<uint8_t, 16> packed = gather<uint8_t, 16>(block_src + 2, offsets);
                    
                    // Sequential layout: low nibbles first, then high nibbles
                    simd<uint8_t, Q4_0_QK> unpacked;
                    unpacked.select<16, 1>(0) = packed & (uint8_t)0x0F;
                    unpacked.select<16, 1>(16) = packed >> 4;
                    
                    simd<int16_t, Q4_0_QK> signed_vals = unpacked;
                    signed_vals = signed_vals - (int16_t)8;
                    simd<fp16, Q4_0_QK> result = signed_vals * scale;
                    
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, Q4_0_QK>(reinterpret_cast<fp16*>(block_dst), result);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, Q4_0_QK> bf_result = result;
                        block_store<bf16, Q4_0_QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                    } else {
                        simd<float, Q4_0_QK> f_result = result;
                        block_store<float, Q4_0_QK>(block_dst, f_result);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0");
}

// ============================================================================
// Q8_0 Kernel
// ============================================================================
template<typename OT, int SBS = 16>
void dequantize_q8_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    constexpr int WG_SIZE = 64;
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_groups = (n_groups + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_groups), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_groups) return;
                
                const int64_t start_block = gid * SBS;
                const int64_t end_block = std::min(start_block + (int64_t)SBS, n_blocks);
                
                simd<uint32_t, Q8_0_QK> offsets;
                #pragma unroll
                for (int i = 0; i < Q8_0_QK; ++i) offsets[i] = i;
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q8_0_BLOCK_SIZE;
                    OT* block_dst = dst + blk * Q8_0_QK;
                    
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    simd<uint8_t, Q8_0_QK> uint8_data = gather<uint8_t, Q8_0_QK>(block_src + 2, offsets);
                    
                    simd<int16_t, Q8_0_QK> signed_vals;
                    #pragma unroll
                    for (int i = 0; i < Q8_0_QK; ++i) {
                        signed_vals[i] = static_cast<int8_t>(uint8_data[i]);
                    }
                    
                    simd<fp16, Q8_0_QK> result = signed_vals * scale;
                    
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, Q8_0_QK>(reinterpret_cast<fp16*>(block_dst), result);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, Q8_0_QK> bf_result = result;
                        block_store<bf16, Q8_0_QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                    } else {
                        simd<float, Q8_0_QK> f_result = result;
                        block_store<float, Q8_0_QK>(block_dst, f_result);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q8_0");
}

// ============================================================================
// Q4_K Kernel (ComfyUI layout)
// ============================================================================
template<typename OT>
void dequantize_q4_k_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_blocks + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t blk = item.get_global_id(0);
                if (blk >= n_blocks) return;
                
                const uint8_t* block_src = src + blk * Q4_K_BLOCK_SIZE;
                OT* block_dst = dst + blk * Q4_K_QK;
                
                const fp16 d = *reinterpret_cast<const fp16*>(block_src);
                const fp16 dmin = *reinterpret_cast<const fp16*>(block_src + 2);
                
                simd<uint8_t, K_SCALE_SIZE> scales_data;
                const uint8_t* scales_ptr = block_src + 4;
                #pragma unroll
                for (int i = 0; i < K_SCALE_SIZE; ++i) scales_data[i] = scales_ptr[i];
                
                const uint8_t* qs = block_src + 4 + K_SCALE_SIZE;
                
                simd<uint32_t, 32> offsets32;
                #pragma unroll
                for (int i = 0; i < 32; ++i) offsets32[i] = i;
                
                // Process 4 super-groups (PyTorch layout)
                #pragma unroll
                for (int sg = 0; sg < 4; ++sg) {
                    const int j_low = sg * 2;
                    const int j_high = sg * 2 + 1;
                    
                    // Extract scales and mins
                    uint8_t sc_low, m_low, sc_high, m_high;
                    if (j_low < 4) {
                        sc_low = scales_data[j_low] & 63;
                        m_low = scales_data[j_low + 4] & 63;
                    } else {
                        sc_low = (scales_data[j_low + 4] & 0xF) | ((scales_data[j_low - 4] >> 6) << 4);
                        m_low = (scales_data[j_low + 4] >> 4) | ((scales_data[j_low] >> 6) << 4);
                    }
                    if (j_high < 4) {
                        sc_high = scales_data[j_high] & 63;
                        m_high = scales_data[j_high + 4] & 63;
                    } else {
                        sc_high = (scales_data[j_high + 4] & 0xF) | ((scales_data[j_high - 4] >> 6) << 4);
                        m_high = (scales_data[j_high + 4] >> 4) | ((scales_data[j_high] >> 6) << 4);
                    }
                    
                    fp16 d_sc_low = d * fp16(sc_low);
                    fp16 dm_m_low = dmin * fp16(m_low);
                    fp16 d_sc_high = d * fp16(sc_high);
                    fp16 dm_m_high = dmin * fp16(m_high);
                    
                    simd<uint8_t, 32> packed = gather<uint8_t, 32>(qs + sg * 32, offsets32);
                    simd<uint8_t, 32> low_nibbles = packed & (uint8_t)0x0F;
                    simd<uint8_t, 32> high_nibbles = packed >> 4;
                    
                    simd<fp16, 32> result_low, result_high;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        result_low[i] = d_sc_low * fp16(low_nibbles[i]) - dm_m_low;
                        result_high[i] = d_sc_high * fp16(high_nibbles[i]) - dm_m_high;
                    }
                    
                    OT* out_low = block_dst + j_low * 32;
                    OT* out_high = block_dst + j_high * 32;
                    
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_low), result_low);
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_high), result_high);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, 32> bf_low = result_low, bf_high = result_high;
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_low), bf_low);
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_high), bf_high);
                    } else {
                        simd<float, 32> f_low = result_low, f_high = result_high;
                        block_store<float, 32>(out_low, f_low);
                        block_store<float, 32>(out_high, f_high);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_k");
}

// ============================================================================
// Q6_K Kernel (ComfyUI/PyTorch layout)
// ============================================================================
template<typename OT, int SBS = 8>
void dequantize_q6_k_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_groups), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                const int64_t start_block = gid * SBS;
                const int64_t end_block = std::min(start_block + (int64_t)SBS, n_blocks);
                
                if (start_block >= n_blocks) return;
                
                // Layout mapping (derived from PyTorch reshape operations)
                constexpr int ql_byte_starts[8] = {0, 32, 0, 32, 64, 96, 64, 96};
                constexpr int use_high_nibble[8] = {0, 0, 1, 1, 0, 0, 1, 1};
                constexpr int qh_byte_starts[8] = {0, 0, 0, 0, 32, 32, 32, 32};
                constexpr int qh_bit_shifts[8] = {0, 2, 4, 6, 0, 2, 4, 6};
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q6_K_BLOCK_SIZE;
                    OT* block_dst = dst + blk * Q6_K_QK;
                    
                    const uint8_t* ql = block_src;
                    const uint8_t* qh = block_src + 128;
                    const int8_t* scales_ptr = reinterpret_cast<const int8_t*>(block_src + 192);
                    const fp16 d = *reinterpret_cast<const fp16*>(block_src + 208);
                    
                    simd<int8_t, 16> scales;
                    simd<uint32_t, 16> scale_offsets;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) scale_offsets[i] = i;
                    scales = gather<int8_t, 16>(scales_ptr, scale_offsets);
                    
                    simd<uint32_t, 32> offsets;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) offsets[i] = i;
                    
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        simd<uint8_t, 32> ql_bytes = gather<uint8_t, 32>(ql + ql_byte_starts[g], offsets);
                        simd<uint8_t, 32> qh_bytes = gather<uint8_t, 32>(qh + qh_byte_starts[g], offsets);
                        
                        simd<uint8_t, 32> ql_vals = use_high_nibble[g] ? 
                            ((ql_bytes >> 4) & (uint8_t)0x0F) : (ql_bytes & (uint8_t)0x0F);
                        simd<uint8_t, 32> qh_vals = (qh_bytes >> qh_bit_shifts[g]) & (uint8_t)0x03;
                        
                        simd<int16_t, 32> q6 = ql_vals | (qh_vals << 4);
                        q6 = q6 - (int16_t)32;
                        
                        fp16 d_scale0 = d * fp16(scales[g * 2]);
                        fp16 d_scale1 = d * fp16(scales[g * 2 + 1]);
                        
                        simd<fp16, 32> result;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) result[i] = d_scale0 * fp16(q6[i]);
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) result[i + 16] = d_scale1 * fp16(q6[i + 16]);
                        
                        OT* out_ptr = block_dst + g * 32;
                        if constexpr (std::is_same_v<OT, fp16>) {
                            block_store<fp16, 32>(reinterpret_cast<fp16*>(out_ptr), result);
                        } else if constexpr (std::is_same_v<OT, bf16>) {
                            simd<bf16, 32> bf_result = result;
                            block_store<bf16, 32>(reinterpret_cast<bf16*>(out_ptr), bf_result);
                        } else {
                            simd<float, 32> f_result = result;
                            block_store<float, 32>(out_ptr, f_result);
                        }
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q6_k");
}

// ============================================================================
// Public API
// ============================================================================

torch::Tensor dequantize_q4_0(const torch::Tensor& input, torch::ScalarType dtype) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q4_0_BLOCK_SIZE;
    TORCH_CHECK(n_bytes % Q4_0_BLOCK_SIZE == 0, "Input size must be multiple of 18 bytes");
    
    auto output = torch::empty({n_blocks * Q4_0_QK}, 
        torch::TensorOptions().dtype(dtype).device(input.device()));
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q4_0_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q4_0_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q4_0_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    return output;
}

torch::Tensor dequantize_q8_0(const torch::Tensor& input, torch::ScalarType dtype) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q8_0_BLOCK_SIZE;
    TORCH_CHECK(n_bytes % Q8_0_BLOCK_SIZE == 0, "Input size must be multiple of 34 bytes");
    
    auto output = torch::empty({n_blocks * Q8_0_QK},
        torch::TensorOptions().dtype(dtype).device(input.device()));
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q8_0_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q8_0_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q8_0_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    return output;
}

torch::Tensor dequantize_q4_k(const torch::Tensor& input, torch::ScalarType dtype) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q4_K_BLOCK_SIZE;
    TORCH_CHECK(n_bytes % Q4_K_BLOCK_SIZE == 0, "Input size must be multiple of 144 bytes");
    
    auto output = torch::empty({n_blocks * Q4_K_QK},
        torch::TensorOptions().dtype(dtype).device(input.device()));
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q4_k_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q4_k_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q4_k_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    return output;
}

torch::Tensor dequantize_q6_k(const torch::Tensor& input, torch::ScalarType dtype) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q6_K_BLOCK_SIZE;
    TORCH_CHECK(n_bytes % Q6_K_BLOCK_SIZE == 0, "Input size must be multiple of 210 bytes");
    
    auto output = torch::empty({n_blocks * Q6_K_QK},
        torch::TensorOptions().dtype(dtype).device(input.device()));
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q6_k_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q6_k_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q6_k_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    return output;
}

}  // namespace gguf
}  // namespace omni_xpu
