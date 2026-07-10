#pragma once
// ── Shared TopK kernel (used by both FP8 moe.sycl and INT4 moe_int4.sycl) ───
// Full softmax over all experts, then top-k selection with proper heap.
// One work-item per token.

#include <sycl/ext/intel/esimd.hpp>

using fp16 = sycl::half;
using namespace sycl::ext::intel::esimd;

template<typename KernelName>
void moe_topk_forward_kernel_impl(
    const fp16* logits,
    int* topk_idx,
    fp16* topk_weight,
    const int n_tokens,
    const int n_experts,
    const int top_k,
    const bool norm,
    sycl::queue& queue) {

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<KernelName>(
            sycl::range<1>(n_tokens),
            [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
                const int nid = (int)idx[0];
                const fp16* row_ptr = logits + nid * n_experts;

                // Load all expert scores (padded to 512)
                simd<fp16, 512> scores(fp16(-65504.f));
                for (int i = 0; i < n_experts; i += 16) {
                    scores.select<16, 1>(i) = block_load<fp16, 16>(row_ptr + i);
                }

                // Full softmax over all experts
                fp16 max_v = hmax<fp16>(scores);
                scores -= max_v;
                scores = exp(scores);

                // Zero out padding
                for (int i = n_experts; i < 512; i += 16)
                    scores.select<16, 1>(i) = fp16(0);

                // Sum and normalize in float32 to avoid fp16 precision loss
                float sum_f32 = 0.f;
                for (int i = 0; i < 512; i += 64) {
                    simd<float, 64> chunk = convert<float>(scores.select<64, 1>(i).read());
                    sum_f32 += sycl::ext::intel::esimd::detail::sum<float, float, 64>(chunk);
                }
                float inv_sum = 1.0f / sum_f32;
                for (int i = 0; i < 512; i += 64) {
                    simd<float, 64> chunk = convert<float>(scores.select<64, 1>(i).read()) * inv_sum;
                    scores.select<64, 1>(i) = convert<fp16>(chunk);
                }

                // Top-k selection with proper heap
                const simd<int, 32> iota(0, 1);
                simd<fp16, 32> heap(fp16(65504.f));
                simd<int, 32> hidx(0);

                // Seed heap with first top_k values
                for (int i = 0; i < top_k; ++i) {
                    heap[i] = scores[i];
                    hidx[i] = i;
                }

                // Scan remaining experts
                for (int i = top_k; i < n_experts; ++i) {
                    fp16 v = scores[i];
                    fp16 mn = hmin<fp16>(heap);
                    if (v > mn) {
                        uint32_t pos = fbl(pack_mask(heap == mn));
                        simd_mask<32> m = (iota == (int)pos);
                        heap.merge(simd<fp16, 32>(v), m);
                        hidx.merge(simd<int, 32>(i), m);
                    }
                }

                // Normalize top-k weights (float32 for precision)
                if (norm) {
                    float top_sum = 0.f;
                    for (int i = 0; i < top_k; ++i) {
                        fp16 hv = heap[i];
                        top_sum += (float)hv;
                    }
                    float inv_top = 1.f / top_sum;
                    for (int i = 0; i < top_k; ++i) {
                        fp16 hv = heap[i];
                        heap[i] = fp16((float)hv * inv_top);
                    }
                }

                // Store results
                int* idx_base = topk_idx + nid * top_k;
                fp16* weight_base = topk_weight + nid * top_k;
                for (int i = 0; i < top_k; ++i) {
                    idx_base[i] = hidx[i];
                    weight_base[i] = heap[i];
                }
            });
    });
}
