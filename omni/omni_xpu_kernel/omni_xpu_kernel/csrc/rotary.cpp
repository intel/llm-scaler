// ============================================================================
// Rotary Embedding Kernel - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance fused rotary position embedding for transformer models.
//
// Fuses: bf16→f32 promotion + rotary rotation + f32→bf16 demotion
// into a single kernel, eliminating intermediate tensors and redundant
// memory round-trips.
//
// Math (for each pair [x0, x1] at positions 2k, 2k+1):
//   out[2k]   = x[2k]   * cos[k] - x[2k+1] * sin[k]
//   out[2k+1] = x[2k]   * sin[k] + x[2k+1] * cos[k]
//
// where cos/sin come from freqs_cis (complex64 tensor):
//   cos[k] = freqs_cis.real[k]
//   sin[k] = freqs_cis.imag[k]
//
// Input shapes:
//   x:         [batch_flat, head_dim]   bf16, contiguous
//              (reshaped from [B, S, heads, head_dim] to [B*heads, S, head_dim]
//               then processed per-sequence-position)
//   cos_cache: [seq_len, head_dim/2]    f32, contiguous
//   sin_cache: [seq_len, head_dim/2]    f32, contiguous
//
// Alternative: process full [B*S*heads, head_dim] with per-row freq lookup.
// We use the simpler approach: caller provides [batch_total, head_dim] where
// each row has its corresponding cos/sin row already indexed.
//
// Actually, the production call pattern is:
//   x_in:      [B, S, heads, head_dim]  bf16
//   freqs_cis: [S, head_dim/2]          complex64
//
// We reshape x to [B*heads, S*head_dim] ... no, better:
// Reshape x to [B*S*heads, head_dim], and expand freqs to match.
// Each row i of x corresponds to freqs row = (i / heads) % S.
//
// For simplicity and maximum kernel efficiency, the Python wrapper will:
// 1. Reshape x to [B*S*heads, head_dim]
// 2. Expand cos/sin to [B*S*heads, head_dim/2] via repeat/expand
// 3. Call kernel with these flat 2D tensors
// 4. Reshape output back
//
// But that expand is wasteful (99072 × 64 × 4 bytes = ~25MB).
// Better approach: pass cos/sin as [S, head_dim/2] and pass a stride.
// Each work-item computes its seq position: seq_idx = (row_id / heads) % S
// and reads cos/sin from row seq_idx.
//
// Kernel design:
//   - Each work-item processes one row of x (one head at one position)
//   - Reads head_dim bf16 values (128 values = 256 bytes)
//   - Reads head_dim/2 cos + head_dim/2 sin values (64+64 = 128 floats = 512 bytes)
//   - Computes rotary rotation in f32
//   - Writes head_dim bf16 values back
//   - Total: 256B read + 512B read + 256B write = 1024B per work-item
//   - With 99072 work-items, total memory traffic = ~99MB (memory-bound)
//
// Block size: head_dim=128, process as pairs → 64 pairs
// Use BS=32 for ESIMD SIMD lanes, process 2 blocks of 32 pairs... 
// Actually, we process elements, not pairs. head_dim=128 = 4 blocks of BS=32.
// But we need to interleave: process [x0,x1] pairs.
// Approach: load even/odd separately.
//   Load x as 128 bf16 values → separate into even[64] and odd[64]
//   Load cos[64], sin[64] as f32
//   Compute: out_even = even*cos - odd*sin, out_odd = even*sin + odd*cos
//   Interleave back to 128 values, convert to bf16, store
//
// ESIMD approach with BS=32:
//   Load x in 4 chunks of 32 bf16 values: [0..31], [32..63], [64..95], [96..127]
//   For chunk pair (chunk0, chunk1) = ([0..31], [32..63]):
//     even = chunk0[0,2,4,...,30], odd = chunk0[1,3,5,...,31]  -- needs gather/scatter
//   This is complex. Simpler: load 128 values, use 2-element stride.
//
// Simplest efficient approach:
//   Process head_dim/2 = 64 pairs, BS=32 → 2 blocks of 32 pairs
//   For each block of 32 pairs:
//     Load 32 cos, 32 sin (contiguous in cos/sin arrays)
//     Load 64 bf16 from x (32 pairs × 2 elements, stride-2 gather)
//     Actually, x[2k] and x[2k+1] are adjacent in memory.
//     Load 64 contiguous bf16 → reinterpret as 32 pairs
//     
// Best approach: load contiguous, use even/odd extraction.
// ESIMD has no native deinterleave, but we can use:
//   simd<bf16, 64> chunk = block_load(x + pair_start*2)
//   Then x_even = chunk.select<32,2>(0)  -- stride-2 select
//   And  x_odd  = chunk.select<32,2>(1)  -- stride-2 select

// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace rotary {

// ============================================================================
// Rotary Embedding Kernel
// ============================================================================
// Each work-item processes one row: [head_dim] elements of x
// Using stride-2 select for even/odd deinterleaving.
//
// Template parameters:
//   IT: input/output type (bf16, fp16, float)
//   HD: head_dim (must be power of 2, e.g. 128)
//   BS: SIMD block size for cos/sin processing (32)
// ============================================================================

template<typename IT, int HD, int BS>
void rotary_emb_kernel(
    const void* x_ptr,          // [total_rows, HD] input
    const float* cos_ptr,       // [S, HD/2] cosine values
    const float* sin_ptr,       // [S, HD/2] sine values
    void* out_ptr,              // [total_rows, HD] output
    int total_rows,             // B * S * heads
    int seq_len,                // S
    int heads,                  // number of attention heads
    const at::Device& device
) {
    static_assert(HD % (BS * 2) == 0, "HD must be divisible by 2*BS");
    constexpr int HALF_HD = HD / 2;
    constexpr int N_PAIR_BLOCKS = HALF_HD / BS;  // number of blocks of BS pairs

    sycl::range<1> global_size(total_rows);
    sycl::range<1> local_size(1);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int row = item.get_global_id(0);
                
                // Compute sequence position for this row
                // Layout: x was [B, S, heads, HD] reshaped to [B*S*heads, HD]
                // row = b * S * heads + s * heads + h
                // seq_idx = (row / heads) % S
                const int seq_idx = (row / heads) % seq_len;

                const IT* x_row = (const IT*)x_ptr + (size_t)row * HD;
                IT* out_row = (IT*)out_ptr + (size_t)row * HD;
                const float* cos_row = cos_ptr + (size_t)seq_idx * HALF_HD;
                const float* sin_row = sin_ptr + (size_t)seq_idx * HALF_HD;

                // Process in blocks of BS pairs (2*BS elements)
                #pragma unroll
                for (int blk = 0; blk < N_PAIR_BLOCKS; ++blk) {
                    const int pair_offset = blk * BS;
                    const int elem_offset = pair_offset * 2;

                    simd<IT, BS * 2> x_chunk = block_load<IT, BS * 2>(x_row + elem_offset);
                    
                    simd<float, BS * 2> x_f32 = x_chunk;
                    
                    simd<float, BS> x_even = x_f32.template select<BS, 2>(0);
                    simd<float, BS> x_odd  = x_f32.template select<BS, 2>(1);

                    simd<float, BS> cos_v = block_load<float, BS>(cos_row + pair_offset);
                    simd<float, BS> sin_v = block_load<float, BS>(sin_row + pair_offset);

                    simd<float, BS> out_even = x_even * cos_v - x_odd * sin_v;
                    simd<float, BS> out_odd  = x_even * sin_v + x_odd * cos_v;

                    simd<float, BS * 2> out_f32;
                    out_f32.template select<BS, 2>(0) = out_even;
                    out_f32.template select<BS, 2>(1) = out_odd;
                    
                    simd<IT, BS * 2> out_chunk = out_f32;
                    block_store<IT, BS * 2>(out_row + elem_offset, out_chunk);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "rotary_emb_esimd");
}

// ============================================================================
// Public C++ API
// ============================================================================

torch::Tensor rotary_emb(
    const torch::Tensor& x,         // [total_rows, head_dim] bf16/f16/f32
    const torch::Tensor& cos_cache, // [S, head_dim/2] f32
    const torch::Tensor& sin_cache, // [S, head_dim/2] f32
    int64_t seq_len,
    int64_t heads
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D [total_rows, head_dim]");
    TORCH_CHECK(cos_cache.dim() == 2, "cos_cache must be 2D [S, head_dim/2]");
    TORCH_CHECK(sin_cache.dim() == 2, "sin_cache must be 2D [S, head_dim/2]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(cos_cache.is_contiguous(), "cos_cache must be contiguous");
    TORCH_CHECK(sin_cache.is_contiguous(), "sin_cache must be contiguous");
    TORCH_CHECK(cos_cache.scalar_type() == at::ScalarType::Float, "cos_cache must be float32");
    TORCH_CHECK(sin_cache.scalar_type() == at::ScalarType::Float, "sin_cache must be float32");

    int64_t total_rows = x.size(0);
    int64_t head_dim = x.size(1);
    int64_t half_hd = head_dim / 2;

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(cos_cache.size(0) == seq_len, "cos_cache seq_len mismatch");
    TORCH_CHECK(cos_cache.size(1) == half_hd, "cos_cache head_dim/2 mismatch");
    TORCH_CHECK(sin_cache.size(0) == seq_len, "sin_cache seq_len mismatch");
    TORCH_CHECK(sin_cache.size(1) == half_hd, "sin_cache head_dim/2 mismatch");
    TORCH_CHECK(total_rows % (seq_len * heads) == 0, 
        "total_rows must be divisible by seq_len * heads");

    auto output = torch::empty_like(x);

    // We use BS=32. head_dim must be divisible by 64 (2*BS).
    // head_dim=128 → 64 pairs → 2 blocks of 32 pairs ✓
    TORCH_CHECK(head_dim % 64 == 0, "head_dim must be divisible by 64");

    const float* cos_ptr = cos_cache.data_ptr<float>();
    const float* sin_ptr = sin_cache.data_ptr<float>();

    // Dispatch by dtype and head_dim
    // We only support head_dim=128 for now (Z-Image model)
    if (head_dim == 128) {
        switch (x.scalar_type()) {
            case at::ScalarType::BFloat16:
                rotary_emb_kernel<bf16, 128, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            case at::ScalarType::Half:
                rotary_emb_kernel<fp16, 128, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            case at::ScalarType::Float:
                rotary_emb_kernel<float, 128, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            default:
                TORCH_CHECK(false, "Unsupported dtype, only bf16, fp16, f32 supported");
        }
    } else if (head_dim == 64) {
        switch (x.scalar_type()) {
            case at::ScalarType::BFloat16:
                rotary_emb_kernel<bf16, 64, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            case at::ScalarType::Half:
                rotary_emb_kernel<fp16, 64, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            case at::ScalarType::Float:
                rotary_emb_kernel<float, 64, 32>(
                    x.data_ptr(), cos_ptr, sin_ptr, output.data_ptr(),
                    total_rows, seq_len, heads, x.device());
                break;
            default:
                TORCH_CHECK(false, "Unsupported dtype, only bf16, fp16, f32 supported");
        }
    } else {
        TORCH_CHECK(false, "Only head_dim=64 and head_dim=128 are supported");
    }

    return output;
}

}  // namespace rotary
}  // namespace omni_xpu
