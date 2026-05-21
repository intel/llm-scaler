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
// Prefill FMHA — Split-KV architecture
//
// Sub-kernel: kBr=4, per-token online softmax, processes a RANGE of KV tokens.
//   Outputs: partial_out[float32], row_max[float32], row_sum[float32]
//
// Reduce kernel: merges partial results using log-sum-exp.
//
// Host launcher: dispatches N sub-kernels in parallel, then 1 reduce kernel.
// ============================================================================

static constexpr uint32_t HEAD_DIM = 256;
static constexpr uint32_t HD_CHUNKS = HEAD_DIM / 16;
static constexpr uint32_t kBr = 4;
static constexpr uint32_t PARTITION_SIZE = 512;  // KV tokens per partition

struct SubKernelArgs {
    fp16* query; fp16* key_cache; fp16* value_cache;
    float* partial_out;   // [num_q_tiles, num_heads, kBr, num_partitions, HEAD_DIM]
    float* partial_max;   // [num_q_tiles, num_heads, kBr, num_partitions]
    float* partial_sum;   // [num_q_tiles, num_heads, kBr, num_partitions]
    int32_t* block_table; int32_t* cu_seqlens_q; int32_t* seqused_k;
    float sm_scale;
    uint32_t num_q_heads, num_kv_heads, max_seqlen_q;
    uint32_t block_size, max_blocks_per_seq, batch_size;
    uint32_t num_partitions;
    bool is_causal;
};

// Sub-kernel: processes one Q-tile × one KV-partition
struct PrefillSubKernel {
    SubKernelArgs args;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        uint32_t wg_id = item.get_group(0);

        // Decode wg_id → (batch, head, q_tile, partition)
        uint32_t num_q_tiles = (args.max_seqlen_q + kBr - 1) / kBr;
        uint32_t tasks_per_batch = num_q_tiles * args.num_q_heads * args.num_partitions;
        uint32_t batch_idx = wg_id / tasks_per_batch;
        uint32_t rem = wg_id % tasks_per_batch;
        uint32_t head_idx = rem / (num_q_tiles * args.num_partitions);
        uint32_t rem2 = rem % (num_q_tiles * args.num_partitions);
        uint32_t q_tile_idx = rem2 / args.num_partitions;
        uint32_t part_idx = rem2 % args.num_partitions;

        if (batch_idx >= args.batch_size) return;
        uint32_t q_start = args.cu_seqlens_q[batch_idx];
        uint32_t seq_len_q = args.cu_seqlens_q[batch_idx + 1] - q_start;
        uint32_t seq_len_k = args.seqused_k[batch_idx];
        uint32_t q_row_start = q_tile_idx * kBr;
        if (q_row_start >= seq_len_q) return;
        uint32_t actual_q_rows = (q_row_start + kBr <= seq_len_q) ? kBr : (seq_len_q - q_row_start);

        // KV range for this partition
        uint32_t kv_range_start = part_idx * PARTITION_SIZE;
        uint32_t kv_range_end = kv_range_start + PARTITION_SIZE;
        if (kv_range_end > seq_len_k) kv_range_end = seq_len_k;
        if (kv_range_start >= seq_len_k) return;  // empty partition

        uint32_t kv_head_idx = head_idx * args.num_kv_heads / args.num_q_heads;
        uint32_t q_stride = args.num_q_heads * HEAD_DIM;
        uint32_t kv_stride = args.num_kv_heads * HEAD_DIM;

        // Load Q
        simd<float,16> q_r0[HD_CHUNKS], q_r1[HD_CHUNKS], q_r2[HD_CHUNKS], q_r3[HD_CHUNKS];
        {
            fp16* qp0 = args.query + (q_start+q_row_start+0)*q_stride + head_idx*HEAD_DIM;
            fp16* qp1 = args.query + (q_start+q_row_start+1)*q_stride + head_idx*HEAD_DIM;
            fp16* qp2 = args.query + (q_start+q_row_start+2)*q_stride + head_idx*HEAD_DIM;
            fp16* qp3 = args.query + (q_start+q_row_start+3)*q_stride + head_idx*HEAD_DIM;
            #pragma unroll
            for (int i=0;i<HD_CHUNKS;i++) {
                q_r0[i] = block_load<fp16,16>(qp0+i*16);
                q_r1[i] = (actual_q_rows>1)?simd<float,16>(block_load<fp16,16>(qp1+i*16)):simd<float,16>(0.0f);
                q_r2[i] = (actual_q_rows>2)?simd<float,16>(block_load<fp16,16>(qp2+i*16)):simd<float,16>(0.0f);
                q_r3[i] = (actual_q_rows>3)?simd<float,16>(block_load<fp16,16>(qp3+i*16)):simd<float,16>(0.0f);
            }
        }

        simd<float,16> o0[HD_CHUNKS], o1[HD_CHUNKS], o2[HD_CHUNKS], o3[HD_CHUNKS];
        #pragma unroll
        for (int i=0;i<HD_CHUNKS;i++) { o0[i]=0; o1[i]=0; o2[i]=0; o3[i]=0; }
        float rm0=-1e30f,rm1=-1e30f,rm2=-1e30f,rm3=-1e30f;
        float rs0=0,rs1=0,rs2=0,rs3=0;
        int32_t seq_diff = (int32_t)seq_len_k - (int32_t)seq_len_q;

        // Process KV range
        for (uint32_t kv_pos = kv_range_start; kv_pos < kv_range_end; kv_pos++) {
            if (args.is_causal) {
                if ((int32_t)kv_pos > (int32_t)(q_row_start+actual_q_rows-1)+seq_diff) break;
            }
            uint32_t blk = args.block_table[batch_idx*args.max_blocks_per_seq+kv_pos/args.block_size];
            uint32_t off = kv_pos % args.block_size;
            fp16* kv_base = args.key_cache+(uint64_t)blk*args.block_size*kv_stride+off*kv_stride+kv_head_idx*HEAD_DIM;

            float d0=0,d1=0,d2=0,d3=0;
            #pragma unroll
            for (int i=0;i<HD_CHUNKS;i++) {
                simd<float,16> kf = block_load<fp16,16>(kv_base+i*16);
                d0+=reduce<float>(q_r0[i]*kf,std::plus<>());
                d1+=reduce<float>(q_r1[i]*kf,std::plus<>());
                d2+=reduce<float>(q_r2[i]*kf,std::plus<>());
                d3+=reduce<float>(q_r3[i]*kf,std::plus<>());
            }
            d0*=args.sm_scale; d1*=args.sm_scale; d2*=args.sm_scale; d3*=args.sm_scale;

            if (args.is_causal) {
                if ((int32_t)kv_pos>(int32_t)(q_row_start+0)+seq_diff) d0=-1e30f;
                if ((int32_t)kv_pos>(int32_t)(q_row_start+1)+seq_diff) d1=-1e30f;
                if ((int32_t)kv_pos>(int32_t)(q_row_start+2)+seq_diff) d2=-1e30f;
                if ((int32_t)kv_pos>(int32_t)(q_row_start+3)+seq_diff) d3=-1e30f;
            }
            if (actual_q_rows<=1) d1=-1e30f;
            if (actual_q_rows<=2) d2=-1e30f;
            if (actual_q_rows<=3) d3=-1e30f;

            float nm0=(d0>rm0)?d0:rm0; float co0=sycl::exp(rm0-nm0); float p0=sycl::exp(d0-nm0);
            float nm1=(d1>rm1)?d1:rm1; float co1=sycl::exp(rm1-nm1); float p1=sycl::exp(d1-nm1);
            float nm2=(d2>rm2)?d2:rm2; float co2=sycl::exp(rm2-nm2); float p2=sycl::exp(d2-nm2);
            float nm3=(d3>rm3)?d3:rm3; float co3=sycl::exp(rm3-nm3); float p3=sycl::exp(d3-nm3);

            #pragma unroll
            for (int i=0;i<HD_CHUNKS;i++) { o0[i]*=co0; o1[i]*=co1; o2[i]*=co2; o3[i]*=co3; }
            rs0=rs0*co0+p0; rm0=nm0;
            rs1=rs1*co1+p1; rm1=nm1;
            rs2=rs2*co2+p2; rm2=nm2;
            rs3=rs3*co3+p3; rm3=nm3;

            fp16* vp = args.value_cache+(uint64_t)blk*args.block_size*kv_stride+off*kv_stride+kv_head_idx*HEAD_DIM;
            #pragma unroll
            for (int i=0;i<HD_CHUNKS;i++) {
                simd<float,16> vf = block_load<fp16,16>(vp+i*16);
                o0[i]+=vf*p0; o1[i]+=vf*p1; o2[i]+=vf*p2; o3[i]+=vf*p3;
            }
        }

        // Write partial results (float32)
        // Layout: [batch, head, q_tile, partition, row, HEAD_DIM] for output
        //         [batch, head, q_tile, partition, row] for max/sum
        uint32_t out_base = ((batch_idx * args.num_q_heads + head_idx) * num_q_tiles + q_tile_idx)
                           * args.num_partitions * kBr;
        uint32_t part_base = out_base + part_idx * kBr;

        // Store partial output (float32, not normalized)
        #define STORE_ROW(R, OUT, RM, RS) { \
            float* op = args.partial_out + (part_base + (R)) * HEAD_DIM; \
            _Pragma("unroll") for (int i=0;i<HD_CHUNKS;i++) \
                block_store<float,16>(op+i*16, OUT[i]); \
            args.partial_max[part_base + (R)] = RM; \
            args.partial_sum[part_base + (R)] = RS; }
        if (actual_q_rows > 0) STORE_ROW(0, o0, rm0, rs0)
        if (actual_q_rows > 1) STORE_ROW(1, o1, rm1, rs1)
        if (actual_q_rows > 2) STORE_ROW(2, o2, rm2, rs2)
        if (actual_q_rows > 3) STORE_ROW(3, o3, rm3, rs3)
        #undef STORE_ROW
    }
};

// Reduce kernel: merge partial results across partitions
struct ReduceKernel {
    float* partial_out;   // from sub-kernels
    float* partial_max;
    float* partial_sum;
    fp16* final_out;      // [total_tokens, num_q_heads, HEAD_DIM]
    uint32_t num_q_heads, max_seqlen_q, num_partitions;
    uint32_t q_stride;  // num_q_heads * HEAD_DIM
    int32_t* cu_seqlens_q;
    uint32_t batch_size;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // Each WG reduces one (batch, head, q_tile, row)
        uint32_t wg_id = item.get_group(0);
        uint32_t num_q_tiles = (max_seqlen_q + kBr - 1) / kBr;
        uint32_t total_rows = batch_size * num_q_heads * num_q_tiles * kBr;
        if (wg_id >= total_rows) return;

        uint32_t batch_idx = wg_id / (num_q_heads * num_q_tiles * kBr);
        uint32_t rem = wg_id % (num_q_heads * num_q_tiles * kBr);
        uint32_t head_idx = rem / (num_q_tiles * kBr);
        uint32_t rem2 = rem % (num_q_tiles * kBr);
        uint32_t q_tile_idx = rem2 / kBr;
        uint32_t row = rem2 % kBr;

        uint32_t q_start = cu_seqlens_q[batch_idx];
        uint32_t seq_len_q = cu_seqlens_q[batch_idx + 1] - q_start;
        uint32_t q_row = q_tile_idx * kBr + row;
        if (q_row >= seq_len_q) return;

        // Base index into partial arrays
        uint32_t base = ((batch_idx * num_q_heads + head_idx) * num_q_tiles + q_tile_idx)
                       * num_partitions * kBr + row;

        // Find global max across partitions
        float global_max = -1e30f;
        for (uint32_t p = 0; p < num_partitions; p++) {
            float m = partial_max[base + p * kBr];
            if (m > global_max) global_max = m;
        }

        // Merge: weighted sum of partial outputs
        simd<float, 16> merged[HD_CHUNKS];
        #pragma unroll
        for (int i = 0; i < HD_CHUNKS; i++) merged[i] = 0.0f;
        float total_sum = 0.0f;

        for (uint32_t p = 0; p < num_partitions; p++) {
            float pm = partial_max[base + p * kBr];
            float ps = partial_sum[base + p * kBr];
            if (ps == 0.0f) continue;  // empty partition

            float correction = sycl::exp(pm - global_max);
            float weighted_sum = ps * correction;
            total_sum += weighted_sum;

            float* po = partial_out + (base + p * kBr) * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < HD_CHUNKS; i++) {
                simd<float, 16> pv = block_load<float, 16>(po + i * 16);
                merged[i] = merged[i] + pv * correction;
            }
        }

        // Normalize and write final output (fp16)
        float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
        fp16* out_ptr = final_out + (q_start + q_row) * q_stride + head_idx * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < HD_CHUNKS; i++) {
            simd<float, 16> o = merged[i] * inv_sum;
            block_store<fp16, 16>(out_ptr + i * 16, simd<fp16, 16>(o));
        }
    }
};

// Host launcher with split-KV
inline void prefill_fmha_launch(
    fp16* query, fp16* key_cache, fp16* value_cache, fp16* output,
    int32_t* block_table, int32_t* cu_seqlens_q, int32_t* seqused_k,
    uint32_t max_seqlen_q, uint32_t max_seqlen_k,
    float sm_scale, bool is_causal,
    uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t block_size, uint32_t max_blocks_per_seq,
    uint32_t batch_size, sycl::queue& q) {

    uint32_t num_q_tiles = (max_seqlen_q + kBr - 1) / kBr;
    uint32_t num_partitions = (max_seqlen_k + PARTITION_SIZE - 1) / PARTITION_SIZE;

    // Allocate temp buffers
    uint32_t total_slots = batch_size * num_q_heads * num_q_tiles * num_partitions * kBr;
    float* partial_out = sycl::malloc_device<float>(total_slots * HEAD_DIM, q);
    float* partial_max = sycl::malloc_device<float>(total_slots, q);
    float* partial_sum = sycl::malloc_device<float>(total_slots, q);

    // Launch sub-kernels (all partitions in parallel)
    SubKernelArgs sub_args{query, key_cache, value_cache,
        partial_out, partial_max, partial_sum,
        block_table, cu_seqlens_q, seqused_k,
        sm_scale, num_q_heads, num_kv_heads, max_seqlen_q,
        block_size, max_blocks_per_seq, batch_size, num_partitions, is_causal};

    uint32_t total_sub_wgs = batch_size * num_q_heads * num_q_tiles * num_partitions;
    q.submit([&](sycl::handler& h) {
        PrefillSubKernel kernel{sub_args};
        h.parallel_for(sycl::nd_range<1>({total_sub_wgs}, {1}), kernel);
    }).wait();

    // Launch reduce kernel
    uint32_t total_reduce_wgs = batch_size * num_q_heads * num_q_tiles * kBr;
    uint32_t q_stride = num_q_heads * HEAD_DIM;
    ReduceKernel reduce_args{partial_out, partial_max, partial_sum, output,
        num_q_heads, max_seqlen_q, num_partitions, q_stride, cu_seqlens_q, batch_size};

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>({total_reduce_wgs}, {1}), reduce_args);
    }).wait();

    // Free temp buffers
    sycl::free(partial_out, q);
    sycl::free(partial_max, q);
    sycl::free(partial_sum, q);
}
