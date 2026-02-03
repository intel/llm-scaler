// ============================================================================
// GGUF Dequantization - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance dequantization using Intel ESIMD (Explicit SIMD)
// 
// Supported formats:
//   Q4_0: Block=32, Size=18 bytes (2 scale + 16 data), output = scale * (nibble - 8)
//   Q8_0: Block=32, Size=34 bytes (2 scale + 32 data), output = scale * int8_val
//   Q4_K: Block=256, Size=144 bytes, output = d*sc*nibble - dmin*m
//
// Output layouts:
//   - INTERLEAVED: output[2i]=low, output[2i+1]=high (standard GGUF)
//   - SEQUENTIAL: output[0-15]=low, output[16-31]=high (ComfyUI)
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <chrono>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace gguf {

// Q4_0 format constants
constexpr int QK = 32;                     // Elements per block
constexpr int BLK_DATA_SIZE = QK / 2;      // Bytes for quantized data (16 bytes)
constexpr int SCALE_SIZE = 2;              // Bytes for scale (FP16)
constexpr int BLOCK_TOTAL_SIZE = 18;       // Total bytes per block (2 + 16)

// Q8_0 format constants
constexpr int Q8_0_QK = 32;                // Elements per block
constexpr int Q8_0_BLOCK_SIZE = 34;        // 2 bytes scale + 32 bytes int8 data

// Q4_K format constants
constexpr int QK_K = 256;                  // Elements per block
constexpr int Q4_K_BLOCK_SIZE = 144;       // 2+2+12+128 bytes
constexpr int K_SCALE_SIZE = 12;           // Scale bytes

// Q6_K format constants
// Block = 256 elements, 6 bits per element
// Layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
constexpr int Q6_K_BLOCK_SIZE = 210;       // Total bytes per block

// Output layout enum
enum class OutputLayout { INTERLEAVED = 0, SEQUENTIAL = 1 };

// ============================================================================
// ESIMD Kernel - Unified kernel handling all blocks (main + remainder)
// Each work-item processes up to SBS blocks with bounds checking
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED, int SBS = 16>
void dequantize_q4_0_unified_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    // Calculate number of work-items needed
    const int64_t n_work_items = (n_blocks + SBS - 1) / SBS;
    
    // Use work-group size of 64 (ESIMD kernel limit)
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_work_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    sycl::range<1> global_size(padded_size);
    sycl::range<1> local_size(WG_SIZE);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_work_items) return;
                
                const int64_t start_block = gid * SBS;
                const int64_t end_block = std::min(start_block + SBS, n_blocks);
                
                const uint8_t* group_src = src + start_block * BLOCK_TOTAL_SIZE;
                OT* group_dst = dst + start_block * QK;
                
                // Process blocks with bounds checking
                for (int64_t blk = 0; blk < end_block - start_block; ++blk) {
                    const uint8_t* block_src = group_src + blk * BLOCK_TOTAL_SIZE;
                    OT* block_dst = group_dst + blk * QK;
                    
                    // Load scale (2 bytes as FP16)
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    
                    // Load 16 bytes of quantized data using gather with offsets 0-15
                    // This avoids byte reordering issues with block_load on uint8
                    simd<uint32_t, BLK_DATA_SIZE> offsets;
                    #pragma unroll
                    for (int i = 0; i < BLK_DATA_SIZE; ++i) {
                        offsets[i] = i;
                    }
                    simd<uint8_t, BLK_DATA_SIZE> packed_data = gather<uint8_t, BLK_DATA_SIZE>(
                        block_src + SCALE_SIZE, offsets);
                    
                    // Extract nibbles and build output
                    simd<uint8_t, QK> unpacked = 0;
                    if constexpr (LAYOUT == OutputLayout::INTERLEAVED) {
                        unpacked.select<BLK_DATA_SIZE, 2>(0) = packed_data & (uint8_t)0x0F;
                        unpacked.select<BLK_DATA_SIZE, 2>(1) = packed_data >> 4;
                    } else {
                        unpacked.select<BLK_DATA_SIZE, 1>(0) = packed_data & (uint8_t)0x0F;
                        unpacked.select<BLK_DATA_SIZE, 1>(BLK_DATA_SIZE) = packed_data >> 4;
                    }
                    
                    // Dequantize: (nibble - 8) * scale
                    // Convert uint8 to int16 first, then subtract 8
                    simd<int16_t, QK> signed_vals = unpacked;
                    signed_vals = signed_vals - (int16_t)8;
                    simd<fp16, QK> result = signed_vals * scale;
                    
                    // Store result using vectorized store
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, QK>(reinterpret_cast<fp16*>(block_dst), result);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, QK> bf_result = result;
                        block_store<bf16, QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                    } else {
                        simd<float, QK> f_result = result;
                        block_store<float, QK>(block_dst, f_result);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0_esimd_unified");
}

// ============================================================================
// ESIMD Kernel - High performance version with vectorized loads/stores
// Each work-item processes SBS blocks
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED, int SBS = 16>
void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = n_blocks / SBS;
    if (n_groups == 0) return;
    
    // Use work-group size of 64 (ESIMD kernel limit)
    constexpr int WG_SIZE = 64;
    const int64_t padded_groups = (n_groups + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    sycl::range<1> global_size(padded_groups);
    sycl::range<1> local_size(WG_SIZE);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_groups) return;
                
                const uint8_t* group_src = src + gid * SBS * BLOCK_TOTAL_SIZE;
                OT* group_dst = dst + gid * SBS * QK;
                
                // Prepare offsets for gather (reusable across blocks)
                simd<uint32_t, BLK_DATA_SIZE> offsets;
                #pragma unroll
                for (int i = 0; i < BLK_DATA_SIZE; ++i) {
                    offsets[i] = i;
                }
                
                // Process all SBS blocks
                #pragma unroll
                for (int blk = 0; blk < SBS; ++blk) {
                    const uint8_t* block_src = group_src + blk * BLOCK_TOTAL_SIZE;
                    OT* block_dst = group_dst + blk * QK;
                    
                    // Load scale (2 bytes as FP16)
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    
                    // Load 16 bytes using gather to avoid byte reordering issues
                    simd<uint8_t, BLK_DATA_SIZE> packed_data = gather<uint8_t, BLK_DATA_SIZE>(
                        block_src + SCALE_SIZE, offsets);
                    
                    // Extract nibbles and build output
                    simd<uint8_t, QK> unpacked = 0;
                    if constexpr (LAYOUT == OutputLayout::INTERLEAVED) {
                        unpacked.select<BLK_DATA_SIZE, 2>(0) = packed_data & (uint8_t)0x0F;
                        unpacked.select<BLK_DATA_SIZE, 2>(1) = packed_data >> 4;
                    } else {
                        unpacked.select<BLK_DATA_SIZE, 1>(0) = packed_data & (uint8_t)0x0F;
                        unpacked.select<BLK_DATA_SIZE, 1>(BLK_DATA_SIZE) = packed_data >> 4;
                    }
                    
                    // Dequantize: (nibble - 8) * scale
                    // Convert uint8 to int16 first, then subtract 8
                    simd<int16_t, QK> signed_vals = unpacked;
                    signed_vals = signed_vals - (int16_t)8;
                    simd<fp16, QK> result = signed_vals * scale;
                    
                    // Store result using vectorized store
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, QK>(reinterpret_cast<fp16*>(block_dst), result);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, QK> bf_result = result;
                        block_store<bf16, QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                    } else {
                        simd<float, QK> f_result = result;
                        block_store<float, QK>(block_dst, f_result);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0_esimd");
}

// ============================================================================
// Remainder kernel for blocks not divisible by SBS
// Optimized with proper work-group size
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED>
void dequantize_q4_0_remainder_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t start_block,
    const int64_t end_block,
    const at::Device& device
) {
    const int64_t n_blocks = end_block - start_block;
    if (n_blocks == 0) return;
    
    // Use work-group size up to 64 for better efficiency
    constexpr int MAX_WG_SIZE = 64;
    const int wg_size = std::min((int64_t)MAX_WG_SIZE, n_blocks);
    const int64_t padded_size = (n_blocks + wg_size - 1) / wg_size * wg_size;
    
    sycl::range<1> global_size(padded_size);
    sycl::range<1> local_size(wg_size);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t lid = item.get_global_id(0);
                if (lid >= n_blocks) return;
                
                const int64_t block_idx = start_block + lid;
                const uint8_t* block_src = src + block_idx * BLOCK_TOTAL_SIZE;
                OT* block_dst = dst + block_idx * QK;
                
                const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                
                // Use gather to avoid byte reordering issues
                simd<uint32_t, BLK_DATA_SIZE> offsets;
                #pragma unroll
                for (int i = 0; i < BLK_DATA_SIZE; ++i) {
                    offsets[i] = i;
                }
                simd<uint8_t, BLK_DATA_SIZE> packed_data = gather<uint8_t, BLK_DATA_SIZE>(
                    block_src + SCALE_SIZE, offsets);
                
                simd<uint8_t, BLK_DATA_SIZE> low_nibbles = packed_data & (uint8_t)0x0F;
                simd<uint8_t, BLK_DATA_SIZE> high_nibbles = packed_data >> 4;
                
                simd<uint8_t, QK> unpacked = 0;
                if constexpr (LAYOUT == OutputLayout::INTERLEAVED) {
                    unpacked.select<BLK_DATA_SIZE, 2>(0) = low_nibbles;
                    unpacked.select<BLK_DATA_SIZE, 2>(1) = high_nibbles;
                } else {
                    unpacked.select<BLK_DATA_SIZE, 1>(0) = low_nibbles;
                    unpacked.select<BLK_DATA_SIZE, 1>(BLK_DATA_SIZE) = high_nibbles;
                }
                
                // Dequantize: (nibble - 8) * scale
                // Convert uint8 to int16 first, then subtract 8
                simd<int16_t, QK> signed_vals = unpacked;
                signed_vals = signed_vals - (int16_t)8;
                simd<fp16, QK> result = signed_vals * scale;
                
                if constexpr (std::is_same_v<OT, fp16>) {
                    block_store<fp16, QK>(reinterpret_cast<fp16*>(block_dst), result);
                } else if constexpr (std::is_same_v<OT, bf16>) {
                    simd<bf16, QK> bf_result = result;
                    block_store<bf16, QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                } else {
                    simd<float, QK> f_result = result;
                    block_store<float, QK>(block_dst, f_result);
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0_esimd_remainder");
}

// ============================================================================
// Dispatch function with SBS handling
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED, int SBS = 32>
void dequantize_q4_0_dispatch(
    const uint8_t* src,
    OT* dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_main_blocks = (n_blocks / SBS) * SBS;
    
    if (n_main_blocks > 0) {
        dequantize_q4_0_kernel<OT, LAYOUT, SBS>(src, dst, n_main_blocks, device);
    }
    
    if (n_main_blocks < n_blocks) {
        dequantize_q4_0_remainder_kernel<OT, LAYOUT>(
            src, dst, n_main_blocks, n_blocks, device
        );
    }
}

// ============================================================================
// Adaptive dispatch based on data size
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED>
void dequantize_q4_0_adaptive(
    const uint8_t* src,
    OT* dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    // Use optimized dispatch with unrolled loops
    // SBS=16 provides good balance across all sizes
    dequantize_q4_0_dispatch<OT, LAYOUT, 16>(src, dst, n_blocks, device);
}

// ============================================================================
// Main implementation
// ============================================================================
torch::Tensor dequantize_q4_0_impl(
    const torch::Tensor& input,
    torch::ScalarType dtype,
    bool sequential_layout
) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / BLOCK_TOTAL_SIZE;
    const int64_t n_elements = n_blocks * QK;
    
    TORCH_CHECK(n_bytes % BLOCK_TOTAL_SIZE == 0, 
        "Input size must be multiple of block size (18 bytes)");
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(input.device());
    torch::Tensor output = torch::empty({n_elements}, options);
    
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (sequential_layout) {
        if (dtype == torch::kFloat32) {
            dequantize_q4_0_adaptive<float, OutputLayout::SEQUENTIAL>(
                src, output.data_ptr<float>(), n_blocks, input.device()
            );
        } else if (dtype == torch::kFloat16) {
            dequantize_q4_0_adaptive<fp16, OutputLayout::SEQUENTIAL>(
                src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else if (dtype == torch::kBFloat16) {
            dequantize_q4_0_adaptive<bf16, OutputLayout::SEQUENTIAL>(
                src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else {
            TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
        }
    } else {
        if (dtype == torch::kFloat32) {
            dequantize_q4_0_adaptive<float, OutputLayout::INTERLEAVED>(
                src, output.data_ptr<float>(), n_blocks, input.device()
            );
        } else if (dtype == torch::kFloat16) {
            dequantize_q4_0_adaptive<fp16, OutputLayout::INTERLEAVED>(
                src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else if (dtype == torch::kBFloat16) {
            dequantize_q4_0_adaptive<bf16, OutputLayout::INTERLEAVED>(
                src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else {
            TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
        }
    }
    
    return output;
}

// ============================================================================
// Public C++ API
// ============================================================================

// Standard API (interleaved output)
torch::Tensor dequantize_q4_0(const torch::Tensor& input, torch::ScalarType dtype) {
    return dequantize_q4_0_impl(input, dtype, false);
}

// ComfyUI API (sequential output)
torch::Tensor dequantize_q4_0_comfyui(const torch::Tensor& input, torch::ScalarType dtype) {
    return dequantize_q4_0_impl(input, dtype, true);
}

// Benchmark helper
double benchmark(
    const torch::Tensor& input,
    torch::ScalarType dtype,
    int warmup_iters,
    int bench_iters
) {
    for (int i = 0; i < warmup_iters; ++i) {
        auto out = dequantize_q4_0(input, dtype);
    }
    
    utils::synchronize(input.device());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        auto out = dequantize_q4_0(input, dtype);
    }
    
    utils::synchronize(input.device());
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / bench_iters;
}

// ============================================================================
// Q8_0 Dequantization ESIMD Kernel
// Format: 2 bytes scale (fp16) + 32 bytes int8 data = 34 bytes per block
// Output: scale * int8_value
// ============================================================================
template<typename OT, int SBS = 16>
void dequantize_q8_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    
    sycl::range<1> global_size(n_groups);
    sycl::range<1> local_size(1);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                const int64_t start_block = gid * SBS;
                const int64_t potential_end = start_block + (int64_t)SBS;
                const int64_t end_block = (potential_end < n_blocks) ? potential_end : n_blocks;
                
                if (start_block >= n_blocks) return;
                
                // Prepare offsets for gather (32 bytes of int8 data)
                simd<uint32_t, Q8_0_QK> offsets;
                #pragma unroll
                for (int i = 0; i < Q8_0_QK; ++i) {
                    offsets[i] = i;
                }
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q8_0_BLOCK_SIZE;
                    OT* block_dst = dst + blk * Q8_0_QK;
                    
                    // Load scale (2 bytes as FP16)
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    
                    // Load 32 bytes of int8 data using gather
                    simd<uint8_t, Q8_0_QK> uint8_data = gather<uint8_t, Q8_0_QK>(
                        block_src + 2, offsets);
                    
                    // Convert uint8 to signed int8 (values are stored as int8 but loaded as uint8)
                    // int8 value = uint8 value if < 128, else uint8 - 256
                    simd<int16_t, Q8_0_QK> signed_vals;
                    #pragma unroll
                    for (int i = 0; i < Q8_0_QK; ++i) {
                        int16_t val = uint8_data[i];
                        signed_vals[i] = (val > 127) ? (val - 256) : val;
                    }
                    
                    // Dequantize: scale * int8_value
                    simd<fp16, Q8_0_QK> result = signed_vals * scale;
                    
                    // Store result
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
    
    utils::submit_kernel(cgf, device, "dequantize_q8_0_esimd");
}

// Q8_0 implementation
torch::Tensor dequantize_q8_0_impl(
    const torch::Tensor& input,
    torch::ScalarType dtype
) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q8_0_BLOCK_SIZE;
    const int64_t n_elements = n_blocks * Q8_0_QK;
    
    TORCH_CHECK(n_bytes % Q8_0_BLOCK_SIZE == 0, 
        "Input size must be multiple of Q8_0 block size (34 bytes)");
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(input.device());
    torch::Tensor output = torch::empty({n_elements}, options);
    
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q8_0_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q8_0_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q8_0_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
    }
    
    return output;
}

// Public Q8_0 API
torch::Tensor dequantize_q8_0(const torch::Tensor& input, torch::ScalarType dtype) {
    return dequantize_q8_0_impl(input, dtype);
}

// ============================================================================
// Q4_K Dequantization ESIMD Kernel
// Format: 2 bytes d + 2 bytes dmin + 12 bytes scales + 128 bytes data = 144 bytes
// Block size: 256 elements (QK_K)
// Output: d * sc * nibble - dmin * m
// ============================================================================

// Helper to extract scale and min from packed scales
// scales format: 12 bytes containing 8 scales (6 bits each) and 8 mins (6 bits each)
// Note: This function is inlined directly in the kernel due to ESIMD restrictions

template<typename OT, int SBS = 4>
void dequantize_q4_k_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    
    sycl::range<1> global_size(n_groups);
    sycl::range<1> local_size(1);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                const int64_t start_block = gid * SBS;
                const int64_t potential_end = start_block + (int64_t)SBS;
                const int64_t end_block = (potential_end < n_blocks) ? potential_end : n_blocks;
                
                if (start_block >= n_blocks) return;
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q4_K_BLOCK_SIZE;
                    OT* block_dst = dst + blk * QK_K;
                    
                    // Load d and dmin (2 bytes each, as FP16)
                    const fp16 d = *reinterpret_cast<const fp16*>(block_src);
                    const fp16 dmin = *reinterpret_cast<const fp16*>(block_src + 2);
                    
                    // Load scales (12 bytes) using scalar loads
                    simd<uint8_t, K_SCALE_SIZE> scales_data;
                    const uint8_t* scales_ptr = block_src + 4;
                    #pragma unroll
                    for (int i = 0; i < K_SCALE_SIZE; ++i) {
                        scales_data[i] = scales_ptr[i];
                    }
                    
                    // Load quantized data (128 bytes = 256 nibbles)
                    const uint8_t* qs = block_src + 4 + K_SCALE_SIZE;
                    
                    // Process 8 groups of 32 elements each
                    #pragma unroll
                    for (int j = 0; j < QK_K / 32; ++j) {
                        // Get scale and min for this group (inlined get_scale_min_k4)
                        uint8_t sc, m;
                        if (j < 4) {
                            sc = scales_data[j] & 63;
                            m = scales_data[j + 4] & 63;
                        } else {
                            sc = (scales_data[j + 4] & 0xF) | ((scales_data[j - 4] >> 6) << 4);
                            m = (scales_data[j + 4] >> 4) | ((scales_data[j] >> 6) << 4);
                        }
                        
                        fp16 d_sc = d * fp16(sc);
                        fp16 dm_m = dmin * fp16(m);
                        
                        // Process 32 elements (16 bytes of packed nibbles)
                        const uint8_t* q_ptr = qs + j * 16;
                        OT* out_ptr = block_dst + j * 32;
                        
                        // Load 16 bytes using gather
                        simd<uint32_t, 16> offsets;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            offsets[i] = i;
                        }
                        simd<uint8_t, 16> packed_data = gather<uint8_t, 16>(q_ptr, offsets);
                        
                        // Extract low and high nibbles
                        simd<uint8_t, 16> low_nibbles = packed_data & (uint8_t)0x0F;
                        simd<uint8_t, 16> high_nibbles = packed_data >> 4;
                        
                        // Build output: first 16 are low nibbles, next 16 are high nibbles
                        simd<fp16, 32> result;
                        
                        // Low nibbles -> positions 0-15
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            result[i] = d_sc * fp16(low_nibbles[i]) - dm_m;
                        }
                        
                        // High nibbles -> positions 16-31
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            result[16 + i] = d_sc * fp16(high_nibbles[i]) - dm_m;
                        }
                        
                        // Store result
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
    
    utils::submit_kernel(cgf, device, "dequantize_q4_k_esimd");
}

// Q4_K implementation
torch::Tensor dequantize_q4_k_impl(
    const torch::Tensor& input,
    torch::ScalarType dtype
) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q4_K_BLOCK_SIZE;
    const int64_t n_elements = n_blocks * QK_K;
    
    TORCH_CHECK(n_bytes % Q4_K_BLOCK_SIZE == 0, 
        "Input size must be multiple of Q4_K block size (144 bytes)");
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(input.device());
    torch::Tensor output = torch::empty({n_elements}, options);
    
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q4_k_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q4_k_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q4_k_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
    }
    
    return output;
}

// Public Q4_K API
torch::Tensor dequantize_q4_k(const torch::Tensor& input, torch::ScalarType dtype) {
    return dequantize_q4_k_impl(input, dtype);
}

// ============================================================================
// Q6_K Dequantization ESIMD Kernel
// Format: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per block
// Each element uses 6 bits: 4 bits in ql, 2 bits in qh
// Output: d * scale * (q6_value - 32)
// ============================================================================
template<typename OT, int SBS = 4>
void dequantize_q6_k_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    
    sycl::range<1> global_size(n_groups);
    sycl::range<1> local_size(1);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                const int64_t start_block = gid * SBS;
                const int64_t potential_end = start_block + (int64_t)SBS;
                const int64_t end_block = (potential_end < n_blocks) ? potential_end : n_blocks;
                
                if (start_block >= n_blocks) return;
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q6_K_BLOCK_SIZE;
                    OT* block_dst = dst + blk * QK_K;
                    
                    // Q6_K layout: ql[128] + qh[64] + scales[16] + d[2]
                    const uint8_t* ql = block_src;           // 128 bytes: low 4 bits
                    const uint8_t* qh = block_src + 128;     // 64 bytes: high 2 bits
                    const int8_t* scales = reinterpret_cast<const int8_t*>(block_src + 192);  // 16 bytes: int8 scales
                    const fp16 d = *reinterpret_cast<const fp16*>(block_src + 208);  // 2 bytes: super-block scale
                    
                    // Process 256 elements in 8 groups of 32
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        // Get scale for this group (2 groups share one scale in Q6_K)
                        int8_t scale = scales[j];
                        fp16 d_scale = d * fp16(scale);
                        
                        // For each group of 32 elements:
                        // - ql contains 16 bytes (32 elements, 4 bits each)
                        // - qh contains 8 bytes (32 elements, 2 bits each)
                        const uint8_t* ql_ptr = ql + j * 16;
                        const uint8_t* qh_ptr = qh + j * 8;
                        OT* out_ptr = block_dst + j * 32;
                        
                        // Load ql (16 bytes) and qh (8 bytes) using gather
                        simd<uint32_t, 16> ql_offsets;
                        simd<uint32_t, 8> qh_offsets;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            ql_offsets[i] = i;
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            qh_offsets[i] = i;
                        }
                        
                        simd<uint8_t, 16> ql_data = gather<uint8_t, 16>(ql_ptr, ql_offsets);
                        simd<uint8_t, 8> qh_data = gather<uint8_t, 8>(qh_ptr, qh_offsets);
                        
                        // Reconstruct 6-bit values and dequantize
                        simd<fp16, 32> result;
                        
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            // Two elements per ql byte
                            uint8_t ql_byte = ql_data[i];
                            uint8_t qh_byte = qh_data[i / 2];
                            
                            // Element 2*i (even): low nibble of ql, bits 0-1 of qh
                            uint8_t q_low = ql_byte & 0x0F;
                            uint8_t qh_shift_low = (i % 2 == 0) ? 0 : 4;
                            uint8_t q_high_low = (qh_byte >> qh_shift_low) & 0x03;
                            int8_t q6_even = (q_low | (q_high_low << 4)) - 32;
                            
                            // Element 2*i+1 (odd): high nibble of ql, bits 2-3 of qh
                            uint8_t q_high = (ql_byte >> 4) & 0x0F;
                            uint8_t qh_shift_high = (i % 2 == 0) ? 2 : 6;
                            uint8_t q_high_high = (qh_byte >> qh_shift_high) & 0x03;
                            int8_t q6_odd = (q_high | (q_high_high << 4)) - 32;
                            
                            result[2 * i] = d_scale * fp16(q6_even);
                            result[2 * i + 1] = d_scale * fp16(q6_odd);
                        }
                        
                        // Store result
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
    
    utils::submit_kernel(cgf, device, "dequantize_q6_k_esimd");
}

// Q6_K implementation
torch::Tensor dequantize_q6_k_impl(
    const torch::Tensor& input,
    torch::ScalarType dtype
) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kByte, "Input must be uint8");
    
    const int64_t n_bytes = input.numel();
    const int64_t n_blocks = n_bytes / Q6_K_BLOCK_SIZE;
    const int64_t n_elements = n_blocks * QK_K;
    
    TORCH_CHECK(n_bytes % Q6_K_BLOCK_SIZE == 0, 
        "Input size must be multiple of Q6_K block size (210 bytes)");
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(input.device());
    torch::Tensor output = torch::empty({n_elements}, options);
    
    const uint8_t* src = input.data_ptr<uint8_t>();
    
    if (dtype == torch::kFloat32) {
        dequantize_q6_k_kernel<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q6_k_kernel<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q6_k_kernel<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
    } else {
        TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
    }
    
    return output;
}

// Public Q6_K API
torch::Tensor dequantize_q6_k(const torch::Tensor& input, torch::ScalarType dtype) {
    return dequantize_q6_k_impl(input, dtype);
}

}  // namespace gguf
}  // namespace omni_xpu
