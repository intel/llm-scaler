// ============================================================================
// GGUF Q4_0 Dequantization - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance Q4_0 dequantization using Intel ESIMD (Explicit SIMD)
// 
// Q4_0 Format (standard GGUF):
//   - Block size: 32 elements
//   - Type size: 18 bytes per block
//   - Layout: [scale (2 bytes, float16)] [data (16 bytes, 32 x 4-bit)]
//   - Dequantization: output = scale * (nibble - 8)
//
// Output layouts:
//   - INTERLEAVED: output[2i]=low, output[2i+1]=high (standard GGUF)
//   - SEQUENTIAL: output[0-15]=low, output[16-31]=high (ComfyUI)
//
// Performance: ~340 GB/s on Intel Data Center GPU Max (Ponte Vecchio)
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

// Output layout enum
enum class OutputLayout { INTERLEAVED = 0, SEQUENTIAL = 1 };

// ============================================================================
// ESIMD Kernel - processes SBS blocks per work-item
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED, int SBS = 8>
void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    const at::Device& device
) {
    const int64_t n_groups = n_blocks / SBS;
    if (n_groups == 0) return;
    
    sycl::range<1> global_size(n_groups);
    sycl::range<1> local_size(1);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                
                #pragma unroll
                for (int blk = 0; blk < SBS; ++blk) {
                    const int64_t block_idx = gid * SBS + blk;
                    const uint8_t* block_src = src + block_idx * BLOCK_TOTAL_SIZE;
                    OT* block_dst = dst + block_idx * QK;
                    
                    // Load scale (2 bytes as FP16)
                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    
                    // Load 16 bytes of quantized data
                    simd<uint8_t, BLK_DATA_SIZE> packed_data;
                    const uint8_t* data_ptr = block_src + SCALE_SIZE;
                    #pragma unroll
                    for (int i = 0; i < BLK_DATA_SIZE; ++i) {
                        packed_data[i] = data_ptr[i];
                    }
                    
                    // Extract nibbles
                    simd<uint8_t, BLK_DATA_SIZE> low_nibbles = packed_data & (uint8_t)0x0F;
                    simd<uint8_t, BLK_DATA_SIZE> high_nibbles = packed_data >> 4;
                    
                    // Build output based on layout
                    simd<uint8_t, QK> unpacked;
                    
                    if constexpr (LAYOUT == OutputLayout::INTERLEAVED) {
                        unpacked.select<BLK_DATA_SIZE, 2>(0) = low_nibbles;
                        unpacked.select<BLK_DATA_SIZE, 2>(1) = high_nibbles;
                    } else {
                        unpacked.select<BLK_DATA_SIZE, 1>(0) = low_nibbles;
                        unpacked.select<BLK_DATA_SIZE, 1>(BLK_DATA_SIZE) = high_nibbles;
                    }
                    
                    // Convert to signed values (-8 to +7)
                    simd<int16_t, QK> int16_vals = unpacked;
                    int16_vals = int16_vals - (int16_t)8;
                    
                    // Scale and convert to output type
                    simd<fp16, QK> fp16_vals = int16_vals;
                    fp16_vals = fp16_vals * scale;
                    
                    simd<OT, QK> result;
                    if constexpr (std::is_same_v<OT, fp16>) {
                        result = fp16_vals;
                    } else {
                        result = fp16_vals;
                    }
                    
                    block_store<OT, QK>(block_dst, result);
                }
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0_esimd");
}

// ============================================================================
// Remainder kernel for blocks not divisible by SBS
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
    
    sycl::range<1> global_size(n_blocks);
    sycl::range<1> local_size(1);
    
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t lid = item.get_global_id(0);
                const int64_t block_idx = start_block + lid;
                const uint8_t* block_src = src + block_idx * BLOCK_TOTAL_SIZE;
                OT* block_dst = dst + block_idx * QK;
                
                const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                
                simd<uint8_t, BLK_DATA_SIZE> packed_data;
                const uint8_t* data_ptr = block_src + SCALE_SIZE;
                #pragma unroll
                for (int i = 0; i < BLK_DATA_SIZE; ++i) {
                    packed_data[i] = data_ptr[i];
                }
                
                simd<uint8_t, BLK_DATA_SIZE> low_nibbles = packed_data & (uint8_t)0x0F;
                simd<uint8_t, BLK_DATA_SIZE> high_nibbles = packed_data >> 4;
                
                simd<uint8_t, QK> unpacked;
                if constexpr (LAYOUT == OutputLayout::INTERLEAVED) {
                    unpacked.select<BLK_DATA_SIZE, 2>(0) = low_nibbles;
                    unpacked.select<BLK_DATA_SIZE, 2>(1) = high_nibbles;
                } else {
                    unpacked.select<BLK_DATA_SIZE, 1>(0) = low_nibbles;
                    unpacked.select<BLK_DATA_SIZE, 1>(BLK_DATA_SIZE) = high_nibbles;
                }
                
                simd<int16_t, QK> int16_vals = unpacked;
                int16_vals = int16_vals - (int16_t)8;
                
                simd<fp16, QK> fp16_vals = int16_vals;
                fp16_vals = fp16_vals * scale;
                
                simd<OT, QK> result = fp16_vals;
                block_store<OT, QK>(block_dst, result);
            }
        );
    };
    
    utils::submit_kernel(cgf, device, "dequantize_q4_0_esimd_remainder");
}

// ============================================================================
// Dispatch function with SBS handling
// ============================================================================
template<typename OT, OutputLayout LAYOUT = OutputLayout::INTERLEAVED, int SBS = 8>
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
            dequantize_q4_0_dispatch<float, OutputLayout::SEQUENTIAL, 8>(
                src, output.data_ptr<float>(), n_blocks, input.device()
            );
        } else if (dtype == torch::kFloat16) {
            dequantize_q4_0_dispatch<fp16, OutputLayout::SEQUENTIAL, 8>(
                src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else if (dtype == torch::kBFloat16) {
            dequantize_q4_0_dispatch<bf16, OutputLayout::SEQUENTIAL, 8>(
                src, reinterpret_cast<bf16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else {
            TORCH_CHECK(false, "Unsupported output dtype: ", dtype);
        }
    } else {
        if (dtype == torch::kFloat32) {
            dequantize_q4_0_dispatch<float, OutputLayout::INTERLEAVED, 8>(
                src, output.data_ptr<float>(), n_blocks, input.device()
            );
        } else if (dtype == torch::kFloat16) {
            dequantize_q4_0_dispatch<fp16, OutputLayout::INTERLEAVED, 8>(
                src, reinterpret_cast<fp16*>(output.data_ptr()), n_blocks, input.device()
            );
        } else if (dtype == torch::kBFloat16) {
            dequantize_q4_0_dispatch<bf16, OutputLayout::INTERLEAVED, 8>(
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

}  // namespace gguf
}  // namespace omni_xpu
