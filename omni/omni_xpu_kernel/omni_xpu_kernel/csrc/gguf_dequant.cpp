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
    // Use SBS kernel with work-group parallelism for best performance
    constexpr int SBS = 16;
    const int64_t n_main_blocks = (n_blocks / SBS) * SBS;
    
    if (n_main_blocks > 0) {
        dequantize_q4_0_kernel<OT, LAYOUT, SBS>(src, dst, n_main_blocks, device);
    }
    if (n_main_blocks < n_blocks) {
        dequantize_q4_0_remainder_kernel<OT, LAYOUT>(src, dst, n_main_blocks, n_blocks, device);
    }
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
    // Use WG_SIZE=64 with SBS blocks per work-item for optimal performance
    constexpr int WG_SIZE = 64;
    const int64_t n_groups = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_groups = (n_groups + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    sycl::range<1> global_size(padded_groups);
    sycl::range<1> local_size(WG_SIZE);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_groups) return;
                
                const int64_t start_block = gid * SBS;
                const int64_t potential_end = start_block + (int64_t)SBS;
                const int64_t end_block = (potential_end < n_blocks) ? potential_end : n_blocks;
                
                // Prepare offsets for gather (32 bytes of int8 data)
                simd<uint32_t, Q8_0_QK> offsets;
                #pragma unroll
                for (int i = 0; i < Q8_0_QK; ++i) {
                    offsets[i] = i;
                }
                
                #pragma unroll
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
                        signed_vals[i] = static_cast<int8_t>(uint8_data[i]);
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

// Optimized Q8_0 kernel with work-group parallelism
// Each work-item processes one block
template<typename OT>
void dequantize_q8_0_kernel_v2(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_blocks + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    sycl::range<1> global_size(padded_size);
    sycl::range<1> local_size(WG_SIZE);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t blk = item.get_global_id(0);
                if (blk >= n_blocks) return;
                
                const uint8_t* block_src = src + blk * Q8_0_BLOCK_SIZE;
                OT* block_dst = dst + blk * Q8_0_QK;
                
                // Load scale (2 bytes as FP16)
                const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                
                // Prepare offsets for gather (32 bytes of int8 data)
                simd<uint32_t, Q8_0_QK> offsets;
                #pragma unroll
                for (int i = 0; i < Q8_0_QK; ++i) {
                    offsets[i] = i;
                }
                
                // Load 32 bytes of int8 data using gather
                simd<uint8_t, Q8_0_QK> uint8_data = gather<uint8_t, Q8_0_QK>(
                    block_src + 2, offsets);
                
                // Convert uint8 to signed int8 and dequantize
                simd<int16_t, Q8_0_QK> signed_vals;
                #pragma unroll
                for (int i = 0; i < Q8_0_QK; ++i) {
                    signed_vals[i] = static_cast<int8_t>(uint8_data[i]);
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
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q8_0_esimd_v2");
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
    
    // Use original kernel with larger SBS for better performance
    // Q8_0 blocks are small (32 elements), so we need more blocks per work-item
    if (dtype == torch::kFloat32) {
        dequantize_q8_0_kernel<float, 16>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q8_0_kernel<fp16, 16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q8_0_kernel<bf16, 16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
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

// Optimized Q4_K kernel with work-group parallelism
// Each work-item processes one block, using simd operations for maximum throughput
template<typename OT>
void dequantize_q4_k_kernel_v2(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    // Each work-item processes 1 block (256 elements)
    // Use work-group size of 64 for good occupancy
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_blocks + WG_SIZE - 1) / WG_SIZE * WG_SIZE;
    
    sycl::range<1> global_size(padded_size);
    sycl::range<1> local_size(WG_SIZE);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t blk = item.get_global_id(0);
                if (blk >= n_blocks) return;
                
                const uint8_t* block_src = src + blk * Q4_K_BLOCK_SIZE;
                OT* block_dst = dst + blk * QK_K;
                
                // Load d and dmin (2 bytes each, as FP16)
                const fp16 d = *reinterpret_cast<const fp16*>(block_src);
                const fp16 dmin = *reinterpret_cast<const fp16*>(block_src + 2);
                
                // Load scales (12 bytes)
                simd<uint8_t, K_SCALE_SIZE> scales_data;
                const uint8_t* scales_ptr = block_src + 4;
                #pragma unroll
                for (int i = 0; i < K_SCALE_SIZE; ++i) {
                    scales_data[i] = scales_ptr[i];
                }
                
                // Load quantized data (128 bytes = 256 nibbles)
                const uint8_t* qs = block_src + 4 + K_SCALE_SIZE;
                
                // Static offsets for gather (32 bytes)
                simd<uint32_t, 32> offsets32;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    offsets32[i] = i;
                }
                
                // Process 4 super-groups of 64 elements each (matching PyTorch layout)
                // PyTorch layout: each 32 bytes -> group j (32 low nibbles) + group j+1 (32 high nibbles)
                #pragma unroll
                for (int sg = 0; sg < 4; ++sg) {
                    // Two consecutive groups share the same 32 bytes of quantized data
                    const int j_low = sg * 2;      // Group index for low nibbles
                    const int j_high = sg * 2 + 1; // Group index for high nibbles
                    
                    // Get scale and min for low nibble group (j_low)
                    uint8_t sc_low, m_low;
                    if (j_low < 4) {
                        sc_low = scales_data[j_low] & 63;
                        m_low = scales_data[j_low + 4] & 63;
                    } else {
                        sc_low = (scales_data[j_low + 4] & 0xF) | ((scales_data[j_low - 4] >> 6) << 4);
                        m_low = (scales_data[j_low + 4] >> 4) | ((scales_data[j_low] >> 6) << 4);
                    }
                    
                    // Get scale and min for high nibble group (j_high)
                    uint8_t sc_high, m_high;
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
                    
                    // Load 32 bytes of packed nibbles
                    const uint8_t* q_ptr = qs + sg * 32;
                    simd<uint8_t, 32> packed_data = gather<uint8_t, 32>(q_ptr, offsets32);
                    
                    // Extract low and high nibbles
                    simd<uint8_t, 32> low_nibbles = packed_data & (uint8_t)0x0F;
                    simd<uint8_t, 32> high_nibbles = packed_data >> 4;
                    
                    // Output for low nibble group (32 elements)
                    OT* out_ptr_low = block_dst + j_low * 32;
                    simd<fp16, 32> result_low;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        result_low[i] = d_sc_low * fp16(low_nibbles[i]) - dm_m_low;
                    }
                    
                    // Output for high nibble group (32 elements)
                    OT* out_ptr_high = block_dst + j_high * 32;
                    simd<fp16, 32> result_high;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        result_high[i] = d_sc_high * fp16(high_nibbles[i]) - dm_m_high;
                    }
                    
                    // Store results
                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_ptr_low), result_low);
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_ptr_high), result_high);
                    } else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, 32> bf_result_low = result_low;
                        simd<bf16, 32> bf_result_high = result_high;
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_ptr_low), bf_result_low);
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_ptr_high), bf_result_high);
                    } else {
                        simd<float, 32> f_result_low = result_low;
                        simd<float, 32> f_result_high = result_high;
                        block_store<float, 32>(out_ptr_low, f_result_low);
                        block_store<float, 32>(out_ptr_high, f_result_high);
                    }
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_k_esimd_v2");
}

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
    
    // Use optimized v2 kernel with work-group parallelism
    if (dtype == torch::kFloat32) {
        dequantize_q4_k_kernel_v2<float>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q4_k_kernel_v2<fp16>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q4_k_kernel_v2<bf16>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
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
// ============================================================================
// Q6_K Dequantization ESIMD Kernel - FIXED for PyTorch/ComfyUI-GGUF layout
// Format: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per block
// Layout mapping (derived from PyTorch reshape operations):
//   Group 0: ql[0:32] LOW nibble, qh[0:32] >> 0, scales[0, 1]
//   Group 1: ql[32:64] LOW nibble, qh[0:32] >> 2, scales[2, 3]
//   Group 2: ql[0:32] HIGH nibble, qh[0:32] >> 4, scales[4, 5]
//   Group 3: ql[32:64] HIGH nibble, qh[0:32] >> 6, scales[6, 7]
//   Group 4: ql[64:96] LOW nibble, qh[32:64] >> 0, scales[8, 9]
//   Group 5: ql[96:128] LOW nibble, qh[32:64] >> 2, scales[10, 11]
//   Group 6: ql[64:96] HIGH nibble, qh[32:64] >> 4, scales[12, 13]
//   Group 7: ql[96:128] HIGH nibble, qh[32:64] >> 6, scales[14, 15]
// ============================================================================
template<typename OT, int SBS = 8>
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
                
                // Mapping tables (computed at compile time)
                constexpr int ql_byte_starts[8] = {0, 32, 0, 32, 64, 96, 64, 96};
                constexpr int use_high_nibble[8] = {0, 0, 1, 1, 0, 0, 1, 1};
                constexpr int qh_byte_starts[8] = {0, 0, 0, 0, 32, 32, 32, 32};
                constexpr int qh_bit_shifts[8] = {0, 2, 4, 6, 0, 2, 4, 6};
                
                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q6_K_BLOCK_SIZE;
                    OT* block_dst = dst + blk * QK_K;
                    
                    const uint8_t* ql = block_src;
                    const uint8_t* qh = block_src + 128;
                    const int8_t* scales_ptr = reinterpret_cast<const int8_t*>(block_src + 192);
                    const fp16 d = *reinterpret_cast<const fp16*>(block_src + 208);
                    
                    // Load all scales
                    simd<int8_t, 16> scales;
                    simd<uint32_t, 16> scale_offsets;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) scale_offsets[i] = i;
                    scales = gather<int8_t, 16>(scales_ptr, scale_offsets);
                    
                    #pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const int ql_start = ql_byte_starts[g];
                        const int high_nibble = use_high_nibble[g];
                        const int qh_start = qh_byte_starts[g];
                        const int qh_shift = qh_bit_shifts[g];
                        
                        simd<uint32_t, 32> offsets;
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) offsets[i] = i;
                        
                        simd<uint8_t, 32> ql_bytes = gather<uint8_t, 32>(ql + ql_start, offsets);
                        simd<uint8_t, 32> qh_bytes = gather<uint8_t, 32>(qh + qh_start, offsets);
                        
                        simd<uint8_t, 32> ql_vals;
                        if (high_nibble) {
                            ql_vals = (ql_bytes >> 4) & (uint8_t)0x0F;
                        } else {
                            ql_vals = ql_bytes & (uint8_t)0x0F;
                        }
                        
                        simd<uint8_t, 32> qh_vals = (qh_bytes >> qh_shift) & (uint8_t)0x03;
                        
                        simd<int16_t, 32> q6 = ql_vals | (qh_vals << 4);
                        q6 = q6 - (int16_t)32;
                        
                        fp16 d_scale0 = d * fp16(scales[g * 2]);
                        fp16 d_scale1 = d * fp16(scales[g * 2 + 1]);
                        
                        simd<fp16, 32> result;
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            result[i] = d_scale0 * fp16(q6[i]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            result[i + 16] = d_scale1 * fp16(q6[i + 16]);
                        }
                        
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
    
    utils::submit_kernel(cgf, device, "dequantize_q6_k_esimd");
}

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
    
    // Use optimized v2 kernel with work-group parallelism for Q6_K
    // Q6_K blocks are larger (256 elements), so WG parallelism works well
    if (dtype == torch::kFloat32) {
        dequantize_q6_k_kernel<float, 8>(src, output.data_ptr<float>(), n_blocks, input.device());
    } else if (dtype == torch::kFloat16) {
        dequantize_q6_k_kernel<fp16, 8>(src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device());
    } else if (dtype == torch::kBFloat16) {
        dequantize_q6_k_kernel<bf16, 8>(src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device());
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
