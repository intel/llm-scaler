#pragma once
#include <functional>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl;
using fp16 = sycl::half;

// ============================================================================
// Prefill FMHA — kBr=4, kBc=16, SIMD softmax (JIT mode)
//
// Uses scalar sycl::exp for safety (ESIMD exp has JIT issues with large negatives).
// Performance target: ~4x slower than IPEX (validated at 3.9x before NaN fix).
// ============================================================================

static constexpr uint32_t HEAD_DIM = 256;
static constexpr uint32_t HD_CHUNKS = HEAD_DIM / 16;
static constexpr uint32_t kBr = 4;
static constexpr uint32_t kBc = 16;

// SIMD exp with clamp to avoid hardware NaN on very negative inputs
SYCL_ESIMD_FUNCTION inline simd<float, 16> safe_exp16(simd<float, 16> x) {
    // Clamp: values below -80 produce exp=0, no need for hardware exp
    // Use merge to avoid NaN: if x < -80, result = 0, else result = exp(x)
    simd<float, 16> result = __ESIMD_NS::exp(x);
    result.merge(simd<float, 16>(0.0f), x < -80.0f);
    return result;
}

struct PrefillFMHAArgs {
    fp16* query;
    fp16* key_cache;
    fp16* value_cache;
    fp16* output;
    int32_t* block_table;
    int32_t* cu_seqlens_q;
    int32_t* seqused_k;
    float sm_scale;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t max_seqlen_q;
    uint32_t max_seqlen_k;
    uint32_t block_size;
    uint32_t max_blocks_per_seq;
    uint32_t batch_size;
    bool is_causal;
};

struct PrefillFMHAKernel_Best {
    PrefillFMHAArgs args;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        uint32_t wg_id = item.get_group(0);

        uint32_t num_q_tiles = (args.max_seqlen_q + kBr - 1) / kBr;
        uint32_t total_per_batch = num_q_tiles * args.num_q_heads;
        uint32_t batch_idx = wg_id / total_per_batch;
        uint32_t remainder = wg_id % total_per_batch;
        uint32_t head_idx = remainder / num_q_tiles;
        uint32_t q_tile_idx = remainder % num_q_tiles;

        if (batch_idx >= args.batch_size) return;

        uint32_t q_start = args.cu_seqlens_q[batch_idx];
        uint32_t q_end = args.cu_seqlens_q[batch_idx + 1];
        uint32_t seq_len_q = q_end - q_start;
        uint32_t seq_len_k = args.seqused_k[batch_idx];

        uint32_t q_row_start = q_tile_idx * kBr;
        if (q_row_start >= seq_len_q) return;
        uint32_t actual_q_rows = kBr;
        if (q_row_start + kBr > seq_len_q) actual_q_rows = seq_len_q - q_row_start;

        uint32_t kv_head_idx = head_idx * args.num_kv_heads / args.num_q_heads;
        uint32_t q_stride = args.num_q_heads * HEAD_DIM;
        uint32_t kv_stride = args.num_kv_heads * HEAD_DIM;

        // Preload Q
        simd<float, 16> q_r0[HD_CHUNKS], q_r1[HD_CHUNKS], q_r2[HD_CHUNKS], q_r3[HD_CHUNKS];
        {
            fp16* qp0 = args.query + (q_start + q_row_start + 0) * q_stride + head_idx * HEAD_DIM;
            fp16* qp1 = args.query + (q_start + q_row_start + 1) * q_stride + head_idx * HEAD_DIM;
            fp16* qp2 = args.query + (q_start + q_row_start + 2) * q_stride + head_idx * HEAD_DIM;
            fp16* qp3 = args.query + (q_start + q_row_start + 3) * q_stride + head_idx * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < HD_CHUNKS; i++) {
                q_r0[i] = block_load<fp16, 16>(qp0 + i * 16);
                q_r1[i] = (actual_q_rows > 1) ? simd<float,16>(block_load<fp16, 16>(qp1 + i * 16)) : simd<float,16>(0.0f);
                q_r2[i] = (actual_q_rows > 2) ? simd<float,16>(block_load<fp16, 16>(qp2 + i * 16)) : simd<float,16>(0.0f);
                q_r3[i] = (actual_q_rows > 3) ? simd<float,16>(block_load<fp16, 16>(qp3 + i * 16)) : simd<float,16>(0.0f);
            }
        }

        simd<float, 16> out_r0[HD_CHUNKS], out_r1[HD_CHUNKS], out_r2[HD_CHUNKS], out_r3[HD_CHUNKS];
        #pragma unroll
        for (int i = 0; i < HD_CHUNKS; i++) {
            out_r0[i] = 0.0f; out_r1[i] = 0.0f;
            out_r2[i] = 0.0f; out_r3[i] = 0.0f;
        }

        float row_max[kBr] = {-1e30f, -1e30f, -1e30f, -1e30f};
        float row_sum[kBr] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32_t seq_diff = (int32_t)seq_len_k - (int32_t)seq_len_q;

        // Main loop: kBc=16 KV tokens per iteration
        for (uint32_t kv_start = 0; kv_start < seq_len_k; kv_start += kBc) {
            if (args.is_causal) {
                int32_t last_q_visible = (int32_t)(q_row_start + actual_q_rows - 1) + seq_diff;
                if ((int32_t)kv_start > last_q_visible) break;
            }

            uint32_t kv_end = kv_start + kBc;
            if (kv_end > seq_len_k) kv_end = seq_len_k;
            uint32_t chunk_len = kv_end - kv_start;

            // ==== Phase A: Compute scores directly into simd ====
            simd<float, 16> s0 = -1e30f, s1 = -1e30f, s2 = -1e30f, s3 = -1e30f;

            #pragma unroll
            for (int c = 0; c < kBc; c++) {
                if ((uint32_t)c < chunk_len) {
                    uint32_t kv_pos = kv_start + c;
                    uint32_t blk = args.block_table[
                        batch_idx * args.max_blocks_per_seq + kv_pos / args.block_size];
                    uint32_t off = kv_pos % args.block_size;
                    fp16* k_ptr = args.key_cache
                        + (uint64_t)blk * args.block_size * kv_stride
                        + off * kv_stride + kv_head_idx * HEAD_DIM;

                    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < HD_CHUNKS; i++) {
                        simd<float, 16> kf = block_load<fp16, 16>(k_ptr + i * 16);
                        d0 += reduce<float>(q_r0[i] * kf, std::plus<>());
                        d1 += reduce<float>(q_r1[i] * kf, std::plus<>());
                        d2 += reduce<float>(q_r2[i] * kf, std::plus<>());
                        d3 += reduce<float>(q_r3[i] * kf, std::plus<>());
                    }
                    s0[c] = d0 * args.sm_scale;
                    s1[c] = d1 * args.sm_scale;
                    s2[c] = d2 * args.sm_scale;
                    s3[c] = d3 * args.sm_scale;
                }
            }

            // Causal mask directly on simd
            if (args.is_causal) {
                #pragma unroll
                for (int c = 0; c < kBc; c++) {
                    int32_t kv_pos_c = (int32_t)(kv_start + c);
                    if (kv_pos_c > (int32_t)(q_row_start + 0) + seq_diff) s0[c] = -1e30f;
                    if (kv_pos_c > (int32_t)(q_row_start + 1) + seq_diff) s1[c] = -1e30f;
                    if (kv_pos_c > (int32_t)(q_row_start + 2) + seq_diff) s2[c] = -1e30f;
                    if (kv_pos_c > (int32_t)(q_row_start + 3) + seq_diff) s3[c] = -1e30f;
                }
            }
            if (actual_q_rows <= 1) s1 = -1e30f;
            if (actual_q_rows <= 2) s2 = -1e30f;
            if (actual_q_rows <= 3) s3 = -1e30f;

            // ==== Phase B: SIMD softmax with safe exp (clamped) ====
            float cm0 = reduce<float>(s0, maximum<>());
            float cm1 = reduce<float>(s1, maximum<>());
            float cm2 = reduce<float>(s2, maximum<>());
            float cm3 = reduce<float>(s3, maximum<>());

            simd<float, 16> p0 = safe_exp16(s0 - cm0);
            simd<float, 16> p1 = safe_exp16(s1 - cm1);
            simd<float, 16> p2 = safe_exp16(s2 - cm2);
            simd<float, 16> p3 = safe_exp16(s3 - cm3);

            float cs0 = reduce<float>(p0, std::plus<>());
            float cs1 = reduce<float>(p1, std::plus<>());
            float cs2 = reduce<float>(p2, std::plus<>());
            float cs3 = reduce<float>(p3, std::plus<>());

            // Online softmax correction
            float nm0 = (row_max[0] > cm0) ? row_max[0] : cm0;
            float co0 = sycl::exp(sycl::clamp(row_max[0] - nm0, -80.0f, 0.0f));
            float cn0 = sycl::exp(sycl::clamp(cm0 - nm0, -80.0f, 0.0f));
            float nm1 = (row_max[1] > cm1) ? row_max[1] : cm1;
            float co1 = sycl::exp(sycl::clamp(row_max[1] - nm1, -80.0f, 0.0f));
            float cn1 = sycl::exp(sycl::clamp(cm1 - nm1, -80.0f, 0.0f));
            float nm2 = (row_max[2] > cm2) ? row_max[2] : cm2;
            float co2 = sycl::exp(sycl::clamp(row_max[2] - nm2, -80.0f, 0.0f));
            float cn2 = sycl::exp(sycl::clamp(cm2 - nm2, -80.0f, 0.0f));
            float nm3 = (row_max[3] > cm3) ? row_max[3] : cm3;
            float co3 = sycl::exp(sycl::clamp(row_max[3] - nm3, -80.0f, 0.0f));
            float cn3 = sycl::exp(sycl::clamp(cm3 - nm3, -80.0f, 0.0f));

            // Rescale output
            #pragma unroll
            for (int i = 0; i < HD_CHUNKS; i++) {
                out_r0[i] = out_r0[i] * co0;
                out_r1[i] = out_r1[i] * co1;
                out_r2[i] = out_r2[i] * co2;
                out_r3[i] = out_r3[i] * co3;
            }
            row_sum[0] = row_sum[0] * co0 + cs0 * cn0; row_max[0] = nm0;
            row_sum[1] = row_sum[1] * co1 + cs1 * cn1; row_max[1] = nm1;
            row_sum[2] = row_sum[2] * co2 + cs2 * cn2; row_max[2] = nm2;
            row_sum[3] = row_sum[3] * co3 + cs3 * cn3; row_max[3] = nm3;

            // Scale P by correction
            p0 = p0 * cn0; p1 = p1 * cn1; p2 = p2 * cn2; p3 = p3 * cn3;

            // ==== Phase C: P × V accumulate ====
            #pragma unroll
            for (int c = 0; c < kBc; c++) {
                float pp0 = (float)p0[c], pp1 = (float)p1[c];
                float pp2 = (float)p2[c], pp3 = (float)p3[c];

                if (pp0 == 0.0f && pp1 == 0.0f && pp2 == 0.0f && pp3 == 0.0f) continue;
                if ((uint32_t)c >= chunk_len) continue;

                uint32_t kv_pos = kv_start + c;
                uint32_t blk = args.block_table[
                    batch_idx * args.max_blocks_per_seq + kv_pos / args.block_size];
                uint32_t off = kv_pos % args.block_size;
                fp16* v_ptr = args.value_cache
                    + (uint64_t)blk * args.block_size * kv_stride
                    + off * kv_stride + kv_head_idx * HEAD_DIM;

                #pragma unroll
                for (int i = 0; i < HD_CHUNKS; i++) {
                    simd<float, 16> vf = block_load<fp16, 16>(v_ptr + i * 16);
                    out_r0[i] = out_r0[i] + vf * pp0;
                    out_r1[i] = out_r1[i] + vf * pp1;
                    out_r2[i] = out_r2[i] + vf * pp2;
                    out_r3[i] = out_r3[i] + vf * pp3;
                }
            }
        }

        // Normalize and write
        #pragma unroll
        for (int r = 0; r < kBr; r++) {
            if ((uint32_t)r >= actual_q_rows) continue;
            float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;
            fp16* o_ptr = args.output + (q_start + q_row_start + r) * q_stride + head_idx * HEAD_DIM;
            simd<float, 16>* out_row = (r == 0) ? out_r0 : (r == 1) ? out_r1 : (r == 2) ? out_r2 : out_r3;
            #pragma unroll
            for (int i = 0; i < HD_CHUNKS; i++) {
                simd<float, 16> o = out_row[i] * inv_sum;
                block_store<fp16, 16>(o_ptr + i * 16, simd<fp16, 16>(o));
            }
        }
    }
};

inline void prefill_fmha_launch(
    fp16* query, fp16* key_cache, fp16* value_cache, fp16* output,
    int32_t* block_table, int32_t* cu_seqlens_q, int32_t* seqused_k,
    uint32_t max_seqlen_q, uint32_t max_seqlen_k,
    float sm_scale, bool is_causal,
    uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t block_size, uint32_t max_blocks_per_seq,
    uint32_t batch_size,
    sycl::queue& q) {

    PrefillFMHAArgs args{
        query, key_cache, value_cache, output,
        block_table, cu_seqlens_q, seqused_k,
        sm_scale,
        num_q_heads, num_kv_heads,
        max_seqlen_q, max_seqlen_k,
        block_size, max_blocks_per_seq, batch_size,
        is_causal
    };

    uint32_t num_q_tiles = (max_seqlen_q + kBr - 1) / kBr;
    uint32_t total_wgs = batch_size * num_q_heads * num_q_tiles;

    q.submit([&](sycl::handler& h) {
        PrefillFMHAKernel_Best kernel{args};
        h.parallel_for(
            sycl::nd_range<1>({total_wgs}, {1}),
            kernel);
    }).wait();
}
