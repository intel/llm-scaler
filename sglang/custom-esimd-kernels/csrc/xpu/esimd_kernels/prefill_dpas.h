#pragma once
/* prefill_dpas.h — DPAS-based prefill SDPA kernel for HD=256 on Xe2/Xe3.
 *
 * Ported from frameworks.ai.client-ai.esimd-kernels Windows tree
 * (vllm-kernel-custom-windows/csrc/xpu/esimd_kernels/sdp_paged.h::
 * sdp_paged_prefill_dpas).
 *
 * Differences from the source:
 *   - KV cache split into separate `key_cache_ptr` / `value_cache_ptr`
 *     tensors (matching PTL pathB convention; SGL XPU also uses this).
 *     `kv_stride_split` parameter removed.
 *   - All other strides (block / pos / head) shared between K and V
 *     (the layout per-tensor is the same; only the base differs).
 *
 * KV layout (per cache, K and V independently):
 *   [num_blocks, block_size, num_kv_heads, head_dim] bf16/fp16 row-major.
 *
 * Workgroup geometry (kept from source):
 *   32 threads / WG, processes PF_WG_Q_ROWS=128 query rows per WG,
 *   tiles K/V cache in PF_KV_CHUNK=128 chunks, KV_PER_SG=16 cooperative.
 *
 * Author: ported 2026-05-21 (task #135).
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <type_traits>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#ifndef FP32_MIN
#define FP32_MIN (-3.402823466e+38f)
#endif

// Conditional bf16 vs fp16 type alias used by the templated kernel.
template<bool IS_BF16>
using half_t = std::conditional_t<IS_BF16, bf16, fp16>;

/* Constants for prefill DPAS kernel */
static constexpr uint32_t PF_HD = 256;
static constexpr uint32_t PF_HD_BLKS = 16;       // 256 / 16
static constexpr uint32_t PF_WG_Q_ROWS = 128;
static constexpr uint32_t PF_KV_CHUNK = 128;      // 8 sg_i × 16
static constexpr uint32_t PF_KV_PER_SG = 16;
static constexpr uint32_t PF_KV_BLKS = 8;
static constexpr uint32_t PF_Q_ROWS = 8;
static constexpr uint32_t PF_Q_PAIRS = 2;
static constexpr uint32_t PF_Q_TILES = 8;
static constexpr uint32_t PF_Q_GRPS = 4;
static constexpr uint32_t PF_D_BLKS_PER_SG = 2;

static constexpr uint32_t PF_Q_SLM_BASE   = 0x00000;  // 64 KB
static constexpr uint32_t PF_S_SLM_BASE   = 0x10000;  // 32 KB
static constexpr uint32_t PF_MAX_SLM_BASE = 0x18000;  // 4 KB
static constexpr uint32_t PF_SUM_SLM_BASE = 0x19000;  // 4 KB
static constexpr uint32_t PF_TOTAL_SLM    = 0x1A000;  // 104 KB total

template<bool CAUSAL, bool IS_BF16>
ESIMD_INLINE void sdp_paged_prefill_dpas(
    const unsigned short* __restrict__ query_ptr,
    const unsigned short* __restrict__ key_cache_ptr,
    const unsigned short* __restrict__ value_cache_ptr,
    unsigned short* __restrict__ output_ptr,
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    const int* __restrict__ query_start_loc_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_block, int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int num_tokens,
    int max_q_tiles_per_req,
    int batch,
    sycl::nd_item<1>& ndi)
{
    constexpr float LOG2E = sycl::ext::intel::esimd::detail::log2e;
    const float attnScoreMul = attn_scale * LOG2E;

    __ESIMD_NS::slm_init(PF_TOTAL_SLM);
    __esimd_nbarrier_init(1);

    int32_t tid = ndi.get_local_id(0);
    int32_t sg_i = tid & 7;
    int32_t sg_j = tid >> 3;
    int32_t wg_id = ndi.get_group(0);

    int32_t head_idx = wg_id % num_heads;
    int32_t temp = wg_id / num_heads;
    int32_t req_idx = temp / max_q_tiles_per_req;
    int32_t q_tile_idx = temp % max_q_tiles_per_req;

    if (req_idx >= batch) return;

    int32_t req_q_start = query_start_loc_ptr[req_idx];
    int32_t req_q_end = query_start_loc_ptr[req_idx + 1];
    int32_t req_query_len = req_q_end - req_q_start;
    int32_t q_offset = q_tile_idx * PF_WG_Q_ROWS;

    if (q_offset >= req_query_len) return;

    int32_t actual_q_rows = req_query_len - q_offset;
    if (actual_q_rows > (int)PF_WG_Q_ROWS) actual_q_rows = PF_WG_Q_ROWS;

    int32_t seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    int32_t group_size = num_heads / num_kv_heads;
    int32_t kv_head_idx = head_idx / group_size;

    // Convert int64 strides to uint32 derived values immediately — free int64 regs
    uint32_t kv_head_off_u32 = (uint32_t)((int64_t)kv_head_idx * kv_stride_head);
    uint32_t kv_row_bytes = (uint32_t)(kv_stride_pos * 2);
    const unsigned short* kv_v_base = value_cache_ptr;
    const unsigned short* kv_k_base = key_cache_ptr;

    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    // block_size is always power of 2 — use shift for division/modulo
    int32_t block_size_shift = __builtin_ctz(block_size);
    int32_t block_size_mask = block_size - 1;
    int32_t max_valid_blk_idx = (seq_len - 1) >> block_size_shift;

    // Physical rows per block in the 2D surface.
    // For contiguous [2, num_blocks, bs, nkvh, hd]: phys_rows_per_block = bs
    // For interleaved [num_blocks, 2, bs, nkvh, hd] presented as [2, ...]:
    //   phys_rows_per_block = stride(1)/stride(2) = 2*bs (K+V interleaved per block)
    int32_t phys_rows_per_block = (int32_t)(kv_stride_block / kv_stride_pos);
    int32_t phys_block_shift = __builtin_ctz(phys_rows_per_block);

// Experiment: bypass block table to measure overhead
#ifdef PAGED_BYPASS_BLOCK_TABLE
#define BLK_TABLE_LOAD(idx) (idx)
#else
#define BLK_TABLE_LOAD(idx) block_table_row[(idx)]
#endif

// Translate an absolute KV row index into the 2D-surface Y coordinate,
// re-resolving the phys block per row. Required when block_size <
// PF_KV_CHUNK (e.g. vllm production block_size=64 with PF_KV_CHUNK=128):
// a single PF_KV_CHUNK iteration spans multiple phys blocks that are
// scattered in physical memory (paged-attention free-list allocation),
// so we cannot use one v_Y_base + linear offset for the whole chunk.
//
// Inputs: kv_row (absolute KV position in this sequence).
// Returns: the Y coordinate to feed into payload.set_y(), accounting for
// the row's containing phys block.
#define KV_PHYS_Y(kv_row) \
    ((uint32_t)((BLK_TABLE_LOAD((int32_t)((kv_row) >> block_size_shift)) \
                 << phys_block_shift) \
                + ((kv_row) & block_size_mask)))

    int32_t q_global_start = req_q_start + q_offset;
    int32_t q_abs_base = (seq_len - req_query_len) + q_offset;

    // 2D surface parameters — FIXED BASE for entire KV cache
    uint32_t kv_surf_w = kv_row_bytes - 1;
    // Safe upper bound for surface height — covers any practical allocation
    uint32_t kv_surf_h = 0x3FFFFFU;
    uint32_t kv_x_k = kv_head_off_u32;

    // ============================================================
    // COOPERATIVE Q LOAD TO SLM
    // ============================================================
    {
        uint32_t widthInByteQ = num_heads * PF_HD * sizeof(bf16) - 1;
        uint32_t heightQ = num_tokens - 1;

        __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
            (uint32_t*)query_ptr, widthInByteQ, heightQ, widthInByteQ, 0, 0);

        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = tid * 4 + t;
            int d_blk = tile_id >> 3;
            int q_tile = tile_id & 7;

            payloadQ.set_x((head_idx * (int)(PF_HD / 2)) + d_blk * 8);
            payloadQ.set_y(q_global_start + q_tile * 16);

            simd<uint32_t, 128> qTile = __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);

            uint32_t slm_off = PF_Q_SLM_BASE + tile_id * 512;
            slm_block_store<uint32_t, 64>(slm_off, qTile.select<64, 1>(0));
            slm_block_store<uint32_t, 64>(slm_off + 256, qTile.select<64, 1>(64));
        }
    }

    barrier();

    // ============================================================
    // REGISTER DECLARATIONS
    // ============================================================
    simd<float, 1024> A_tile = 0;
    simd<float, 512> ST_tile;
    simd<float, 512> ST_next;
    simd<float, 32> fp32_max = FP32_MIN;
    simd<float, 32> fp32_sum = 0;
    simd<float, 32> delta;

    int32_t max_kv_end;
    if constexpr (CAUSAL) {
        max_kv_end = q_abs_base + actual_q_rows;
        if (max_kv_end > seq_len) max_kv_end = seq_len;
    } else {
        max_kv_end = seq_len;
    }
    int32_t kvOuterLoops = (max_kv_end + PF_KV_CHUNK - 1) / PF_KV_CHUNK;
    if (kvOuterLoops <= 0) kvOuterLoops = 1;

    // Pre-compute causal boundaries (CAUSAL only — noncausal skips entirely, saves 32 regs)
    simd<int32_t, 16> causal_bound_0, causal_bound_1;
    if constexpr (CAUSAL) {
        simd<int32_t, 16> q_idx_vec;
        #pragma unroll
        for (int qr = 0; qr < 16; qr++) q_idx_vec[qr] = qr;

        simd<int32_t, 16> q_local_0 = sg_j * 32 + q_idx_vec;
        simd<int32_t, 16> q_local_1 = sg_j * 32 + 16 + q_idx_vec;

        causal_bound_0 = q_abs_base + q_local_0;
        causal_bound_1 = q_abs_base + q_local_1;
        causal_bound_0.merge(-1, q_local_0 >= actual_q_rows);
        causal_bound_1.merge(-1, q_local_1 >= actual_q_rows);
    }

    // Fixed-base payloads — configured ONCE, only set_x/set_y per iteration
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
        (fp16*)kv_k_base, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, 0);

    __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadKpf(
        (uint32_t*)kv_k_base, kv_surf_w, kv_surf_h, kv_surf_w, (uint32_t)(kv_x_k / 2), 0);

    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
        (fp16*)(kv_v_base), kv_surf_w, kv_surf_h, kv_surf_w,
        (uint32_t)(kv_head_off_u32 + sg_i * 32), 0);

    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadVpf(
        (uint32_t*)(kv_v_base), kv_surf_w, kv_surf_h, kv_surf_w,
        (uint32_t)(kv_head_off_u32 / 2 + sg_i * 16), 0);

    // ============================================================
    // PROLOGUE K PREFETCH: prefetch K[0] + K[1]
    // Per-iteration block lookup: one block_table_row[] load per chunk.
    // ============================================================
    {
        int32_t pf0_phys = BLK_TABLE_LOAD(0);
        payloadKpf.set_y((uint32_t)((pf0_phys << phys_block_shift) + sg_i * (int32_t)PF_KV_PER_SG));
        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS; d++) {
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + d * 8));
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
        }
    }
    if (kvOuterLoops > 1) {
        int32_t pf1_logical = (int32_t)PF_KV_CHUNK >> block_size_shift;
        int32_t pf1_off = (int32_t)PF_KV_CHUNK & block_size_mask;
        int32_t pf1_phys = BLK_TABLE_LOAD(pf1_logical);
        payloadKpf.set_y((uint32_t)((pf1_phys << phys_block_shift) + pf1_off + sg_i * (int32_t)PF_KV_PER_SG));
        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS; d++) {
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + d * 8));
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
        }
    }

    // ============================================================
    // PROLOGUE: QK[0] -> ST_tile, with V[0] prefetch interleaved
    // D-loop peeled: main loop d=0..PF_HD_BLKS-2, then last D-block
    // ============================================================
    {
        // Per-row phys lookup: this sg_i covers KV rows [sg_i*16, sg_i*16+16)
        // (relative to outerIter=0 chunk start). With block_size=64 < KV_CHUNK=128
        // sg_i 0..3 and 4..7 land in different phys blocks.
        uint32_t Y_base_K = KV_PHYS_Y((uint32_t)(sg_i * PF_KV_PER_SG));
        // V prefetch is best-effort (cache miss, not correctness); use phys0.
        int32_t phys0 = BLK_TABLE_LOAD(0);
        uint32_t Y_base_V = (uint32_t)(phys0 << phys_block_shift);

        ST_tile = 0;

        payloadK.set_y(Y_base_K);
        simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS - 1; d++) {
            // V prefetch: half rate (every other D-block) — reduces memory pressure
            if ((d & 1) == 0) {
            payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
            payloadVpf.set_y(Y_base_V + (d >> 1) * 16);
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>(payloadVpf);
            }

            simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
            simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

            uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
            uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

            simd<uint32_t, 128> Q_raw0, Q_raw1;
            Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
            Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
            Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
            Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
            simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
            simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

            { auto acc = ST_tile.select<128, 1>(0);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(128);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
            { auto acc = ST_tile.select<128, 1>(256);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(384);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

            payloadK.set_x(kv_x_k + (d + 1) * 16);
            K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }

        // Last D-block (d = PF_HD_BLKS - 1): no next-K load
        {
            constexpr int d = (int)PF_HD_BLKS - 1;
            // V prefetch: half rate (every other D-block) — reduces memory pressure
            if ((d & 1) == 0) {
            payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
            payloadVpf.set_y(Y_base_V + (d >> 1) * 16);
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>(payloadVpf);
            }

            simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
            simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

            uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
            uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

            simd<uint32_t, 128> Q_raw0, Q_raw1;
            Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
            Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
            Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
            Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
            simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
            simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

            { auto acc = ST_tile.select<128, 1>(0);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(128);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
            { auto acc = ST_tile.select<128, 1>(256);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(384);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }
        }
    }

    // ============================================================
    // OUTER KV LOOP — single loop, ST_tile has QK[outerIter] scores on entry
    // ============================================================
    for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
        uint32_t kv_start = outerIter * PF_KV_CHUNK;

        // ========================================
        // SOFTMAX FIRST HALF: scale scores, apply masks
        // ========================================
        ST_tile *= attnScoreMul;

        // Masking: CAUSAL applies causal_bound every iteration;
        // noncausal only masks the last iteration for kv_pos >= seq_len and invalid Q rows.
        if constexpr (CAUSAL) {
            int32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
            #pragma unroll
            for (int kv = 0; kv < 16; kv++) {
                int32_t kv_pos = kv_base_sg + kv;
                simd<int32_t, 16> v_kv_pos(kv_pos);
                ST_tile.select<16, 1>(0 * 256 + kv * 16).merge(FP32_MIN, v_kv_pos > causal_bound_0);
                ST_tile.select<16, 1>(1 * 256 + kv * 16).merge(FP32_MIN, v_kv_pos > causal_bound_1);
            }
        } else {
            // Noncausal: only need masking in the last iteration
            if (outerIter == kvOuterLoops - 1) {
                int32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
                // Mask positions >= seq_len (partial last chunk)
                #pragma unroll
                for (int kv = 0; kv < 16; kv++) {
                    int32_t kv_pos = kv_base_sg + kv;
                    if (kv_pos >= seq_len) {
                        ST_tile.select<16, 1>(0 * 256 + kv * 16) = FP32_MIN;
                        ST_tile.select<16, 1>(1 * 256 + kv * 16) = FP32_MIN;
                    }
                }
                // Mask invalid Q rows (actual_q_rows < 128)
                if (actual_q_rows < (int)PF_WG_Q_ROWS) {
                    simd<int32_t, 16> q_idx_vec;
                    #pragma unroll
                    for (int qr = 0; qr < 16; qr++) q_idx_vec[qr] = qr;
                    simd<int32_t, 16> q_local_0 = sg_j * 32 + q_idx_vec;
                    simd<int32_t, 16> q_local_1 = sg_j * 32 + 16 + q_idx_vec;
                    #pragma unroll
                    for (int kv = 0; kv < 16; kv++) {
                        ST_tile.select<16, 1>(0 * 256 + kv * 16).merge(FP32_MIN, q_local_0 >= actual_q_rows);
                        ST_tile.select<16, 1>(1 * 256 + kv * 16).merge(FP32_MIN, q_local_1 >= actual_q_rows);
                    }
                }
            }
        }

        simd<float, 32> local_max;
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            local_max.select<16, 1>(qp * 16) = ST_tile.select<16, 1>(qp * 256);
            #pragma unroll
            for (int kv = 1; kv < 8; kv++)
                local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    local_max.select<16, 1>(qp * 16),
                    ST_tile.select<16, 1>(qp * 256 + kv * 16));
            #pragma unroll
            for (int kv = 0; kv < 8; kv++)
                local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    local_max.select<16, 1>(qp * 16),
                    ST_tile.select<16, 1>(qp * 256 + 128 + kv * 16));
        }

        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            slm_block_store<float, 16>(PF_MAX_SLM_BASE + (sg_i * 128 + q_base) * 4,
                local_max.select<16, 1>(qp * 16));
        }

        // ========================================
        // BARRIER A: arrive, QK[k+1] overlap, wait
        // ========================================
        __esimd_nbarrier_arrive(0, 0, 32, 32);

        if (outerIter < kvOuterLoops - 1) {
            uint32_t next_kv_start = (outerIter + 1) * PF_KV_CHUNK;
            // Per-row K lookup for sg_i's slice of the next chunk.
            uint32_t next_Y_base_K = KV_PHYS_Y(next_kv_start + (uint32_t)(sg_i * PF_KV_PER_SG));
            // V prefetch base — best-effort, use first row's phys.
            int32_t next_logical = next_kv_start >> block_size_shift;
            int32_t next_off = next_kv_start & block_size_mask;
            int32_t next_phys = BLK_TABLE_LOAD(next_logical);
            uint32_t next_Y_base_V = (uint32_t)((next_phys << phys_block_shift) + next_off);

            ST_next = 0;

            payloadK.set_y(next_Y_base_K);
            simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

            #pragma unroll
            for (int d = 0; d < (int)PF_HD_BLKS - 1; d++) {
                simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
                simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

                uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

                simd<uint32_t, 128> Q_raw0, Q_raw1;
                Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
                simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
                simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

                { auto acc = ST_next.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_next.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

                // V prefetch for outerIter+1
                // V prefetch: half rate (every other D-block) — reduces memory pressure
                if ((d & 1) == 0) {
                payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
                payloadVpf.set_y(next_Y_base_V + (d >> 1) * 16);
                __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>(payloadVpf);
                }

                payloadK.set_x(kv_x_k + (d + 1) * 16);
                K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
            }

            // Last D-block: no next-K load
            {
                constexpr int d = (int)PF_HD_BLKS - 1;
                simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
                simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

                uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

                simd<uint32_t, 128> Q_raw0, Q_raw1;
                Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
                simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
                simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

                { auto acc = ST_next.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_next.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

                // V prefetch last D-block
                // V prefetch: half rate (every other D-block) — reduces memory pressure
                if ((d & 1) == 0) {
                payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
                payloadVpf.set_y(next_Y_base_V + (d >> 1) * 16);
                __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>(payloadVpf);
                }
            }
        }

        __esimd_nbarrier(0, 0, 32);

        // ========================================
        // SOFTMAX SECOND HALF
        // ========================================
        simd<float, 32> global_max = FP32_MIN;
        #pragma unroll
        for (int si = 0; si < 8; si++) {
            #pragma unroll
            for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
                uint32_t q_base = sg_j * 32 + qp * 16;
                simd<float, 16> m = slm_block_load<float, 16>(PF_MAX_SLM_BASE + (si * 128 + q_base) * 4);
                global_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    global_max.select<16, 1>(qp * 16), m);
            }
        }
        global_max = __ESIMD_NS::max<float, 32, float>(global_max, fp32_max);

        delta = __ESIMD_NS::exp2<float, 32, float>(fp32_max - global_max);
        fp32_max = global_max;

        simd<float, 32> local_sum = 0;
        simd<unsigned short, 256> ST_bf16_0;
        simd<unsigned short, 256> ST_bf16_1;

        {
            simd<float, 16> gm = global_max.select<16, 1>(0);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_0.select<16, 1>(kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(0) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(128 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_0.select<16, 1>(128 + kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(0) += s;
            }
        }
        {
            simd<float, 16> gm = global_max.select<16, 1>(16);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_1.select<16, 1>(kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(16) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + 128 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_1.select<16, 1>(128 + kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(16) += s;
            }
        }

        // ========================================
        // S^T SCATTER TO SLM
        // ========================================
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            auto& ST_bf16 = (qp == 0) ? ST_bf16_0 : ST_bf16_1;
            simd<uint16_t, 256> ST_fp16_u16(ST_bf16);

            uint32_t tile_addr_qh0 = PF_S_SLM_BASE + (sg_i * 16 + sg_j * PF_Q_GRPS + qp * 2 + 0) * 256;
            uint32_t tile_addr_qh1 = PF_S_SLM_BASE + (sg_i * 16 + sg_j * PF_Q_GRPS + qp * 2 + 1) * 256;

            simd<uint32_t, 16> q_offsets;
            #pragma unroll
            for (int q = 0; q < 8; q++)
                q_offsets[q] = tile_addr_qh0 + q * 32;
            #pragma unroll
            for (int q = 0; q < 8; q++)
                q_offsets[8 + q] = tile_addr_qh1 + q * 32;

            #pragma unroll
            for (int kg = 0; kg < 4; kg++) {
                int kv_base = kg * 4;
                simd<uint32_t, 16> packed0 =
                    simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 0) * 16)) |
                    (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 1) * 16)) << 16);
                simd<uint32_t, 16> packed1 =
                    simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 2) * 16)) |
                    (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 3) * 16)) << 16);
                simd<uint32_t, 32> data;
                data.select<16, 1>(0) = packed0;
                data.select<16, 1>(16) = packed1;
                __ESIMD_ENS::lsc_slm_scatter<uint32_t, 2, __ESIMD_ENS::lsc_data_size::u32, 16>(
                    q_offsets + kv_base * 2, data);
            }
        }

        // ========================================
        // BARRIER B: arrive, V loads + compensation, wait
        // ========================================
        __esimd_nbarrier_arrive(0, 0, 32, 32);

        fp32_sum = fp32_sum * delta + local_sum;

        // V phase: re-resolve phys per KV_BLK (each block of 16 KV rows).
        // The chunk spans potentially-discontiguous phys blocks when
        // block_size < PF_KV_CHUNK; KV_PHYS_Y handles the lookup per row.
        uint32_t v_Y_blk0 = KV_PHYS_Y(kv_start);

        // V load kv_blk=0
        payloadV.set_x(kv_head_off_u32 + sg_i * 32);
        payloadV.set_y(v_Y_blk0);
        simd<fp16, 256> V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
        payloadV.set_x(kv_head_off_u32 + sg_i * 32 + 16);
        simd<fp16, 256> V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

        // Compensation
        #pragma unroll
        for (int qg = 0; qg < (int)PF_Q_GRPS; qg++) {
            #pragma unroll
            for (int db = 0; db < (int)PF_D_BLKS_PER_SG; db++) {
                #pragma unroll
                for (int q = 0; q < (int)PF_Q_ROWS; q++) {
                    float d_val = delta[qg * PF_Q_ROWS + q];
                    A_tile.select<16, 1>((qg * PF_D_BLKS_PER_SG + db) * 128 + q * 16) *= d_val;
                }
            }
        }

        __esimd_nbarrier(0, 0, 32);

        // ========================================
        // VS PHASE + K PREFETCH (remaining tiles)
        // ========================================

// Helper macro: load S from SLM, load V (skip for kv_blk 0), run 8 DPAS
// Per-KV_BLK phys lookup via KV_PHYS_Y so a chunk that spans multiple
// (scattered) phys blocks is loaded correctly.
#define VS_LOAD_AND_DPAS(KV_BLK)                                                     \
        do {                                                                          \
            if ((KV_BLK) > 0) {                                                       \
                uint32_t _vy = KV_PHYS_Y(kv_start + (uint32_t)((KV_BLK) * 16));       \
                payloadV.set_x(kv_head_off_u32 + sg_i * 32);               \
                payloadV.set_y(_vy);                                                  \
                V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,     \
                    __ESIMD_ENS::cache_hint::cached,                                  \
                    __ESIMD_ENS::cache_hint::cached>(payloadV);                       \
                payloadV.set_x(kv_head_off_u32 + sg_i * 32 + 16);          \
                V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,     \
                    __ESIMD_ENS::cache_hint::cached,                                  \
                    __ESIMD_ENS::cache_hint::cached>(payloadV);                       \
            }                                                                         \
            uint32_t sb = PF_S_SLM_BASE + ((KV_BLK) * 16 + sg_j * PF_Q_GRPS) * 256; \
            simd<half_t<IS_BF16>, 128> sA, sB, sC, sD;                                          \
            sA.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb);           \
            sB.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 256);     \
            sC.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 512);     \
            sD.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 768);     \
            auto Vb0 = V_vnni0.template bit_cast_view<half_t<IS_BF16>>();                        \
            auto Vb1 = V_vnni1.template bit_cast_view<half_t<IS_BF16>>();                        \
            { auto acc = A_tile.select<128, 1>(0*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(1*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(2*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(3*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(4*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(5*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(6*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sD); } \
            { auto acc = A_tile.select<128, 1>(7*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sD); } \
        } while(0)

#define K_PREFETCH_2(N)                                                                \
        do {                                                                           \
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + (2*(N)) * 8));                   \
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,            \
                __ESIMD_ENS::cache_hint::cached,                                       \
                __ESIMD_ENS::cache_hint::cached>(payloadKpf);                          \
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + (2*(N)+1) * 8));                 \
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,            \
                __ESIMD_ENS::cache_hint::cached,                                       \
                __ESIMD_ENS::cache_hint::cached>(payloadKpf);                          \
        } while(0)

        // K prefetch for outerIter+2: compute address
        {
            int32_t pf_kv_start = (outerIter + 2) * (int32_t)PF_KV_CHUNK;
            int32_t pf_logical = pf_kv_start >> block_size_shift;
            pf_logical = (pf_logical <= max_valid_blk_idx) ? pf_logical : max_valid_blk_idx;
            int32_t pf_off = pf_kv_start & block_size_mask;
            int32_t pf_phys = BLK_TABLE_LOAD(pf_logical);
            payloadKpf.set_y((uint32_t)((pf_phys << phys_block_shift) + pf_off + sg_i * (int32_t)PF_KV_PER_SG));
        }

        VS_LOAD_AND_DPAS(0);  K_PREFETCH_2(0);
        VS_LOAD_AND_DPAS(1);  K_PREFETCH_2(1);
        VS_LOAD_AND_DPAS(2);  K_PREFETCH_2(2);
        VS_LOAD_AND_DPAS(3);  K_PREFETCH_2(3);
        VS_LOAD_AND_DPAS(4);  K_PREFETCH_2(4);
        VS_LOAD_AND_DPAS(5);  K_PREFETCH_2(5);
        VS_LOAD_AND_DPAS(6);  K_PREFETCH_2(6);
        VS_LOAD_AND_DPAS(7);  K_PREFETCH_2(7);

#undef K_PREFETCH_2
#undef VS_LOAD_AND_DPAS

        ST_tile = ST_next;
    }

    // ============================================================
    // FINAL OUTPUT: normalize and store as bf16
    // ============================================================
    #pragma unroll
    for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
        uint32_t q_base = sg_j * 32 + qp * 16;
        slm_block_store<float, 16>(PF_SUM_SLM_BASE + (sg_i * 128 + q_base) * 4,
            fp32_sum.select<16, 1>(qp * 16));
    }

    barrier();

    simd<float, 32> total_sum = 0;
    #pragma unroll
    for (int si = 0; si < 8; si++) {
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            total_sum.select<16, 1>(qp * 16) += slm_block_load<float, 16>(
                PF_SUM_SLM_BASE + (si * 128 + q_base) * 4);
        }
    }

    simd<float, 32> inv_sum;
    inv_sum.select<16, 1>(0) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(0));
    inv_sum.select<16, 1>(16) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(16));

    uint32_t d_start = sg_i * PF_D_BLKS_PER_SG * 16;

    uint32_t outW = num_heads * PF_HD * sizeof(bf16) - 1;
    uint32_t outH = num_tokens - 1;
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 8, 1> payloadO(
        (fp16*)output_ptr, outW, outH, outW, 0, 0);

    #pragma unroll
    for (int qg = 0; qg < (int)PF_Q_GRPS; qg++) {
        #pragma unroll
        for (int db = 0; db < (int)PF_D_BLKS_PER_SG; db++) {
            simd<fp16, 128> fOut_fp16;

            #pragma unroll
            for (int q = 0; q < (int)PF_Q_ROWS; q++) {
                simd<float, 16> f32_out = A_tile.select<16, 1>((qg * PF_D_BLKS_PER_SG + db) * 128 + q * 16);
                float inv = inv_sum[qg * PF_Q_ROWS + q];
                f32_out *= inv;
                simd<half_t<IS_BF16>, 16> bf16_out = f32_out;
                fOut_fp16.select<16, 1>(q * 16) = bf16_out.template bit_cast_view<fp16>();
            }

            int q_row_in_tile = sg_j * 32 + qg * PF_Q_ROWS;
            int q_global_row = q_global_start + q_row_in_tile;

            if (q_row_in_tile + PF_Q_ROWS <= (uint32_t)actual_q_rows) {
                payloadO.set_x(head_idx * PF_HD + d_start + db * 16);
                payloadO.set_y(q_global_row);
                __ESIMD_ENS::lsc_store_2d<fp16, 16, 8, 1,
                    __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back>(payloadO, fOut_fp16);
            } else if (q_row_in_tile < actual_q_rows) {
                int valid_rows = actual_q_rows - q_row_in_tile;
                unsigned short* out_base = output_ptr +
                    (int64_t)q_global_row * num_heads * PF_HD +
                    (int64_t)head_idx * PF_HD + d_start + db * 16;
                int64_t row_stride = (int64_t)num_heads * PF_HD;
                for (int r = 0; r < valid_rows; r++) {
                    block_store<unsigned short, 16>(
                        out_base + (int64_t)r * row_stride,
                        fOut_fp16.select<16, 1>(r * 16).template bit_cast_view<unsigned short>());
                }
            }
        }
    }
#undef BLK_TABLE_LOAD
}

// ============================================================
// Host-side launcher (PTL pathB convention).
//
// Inputs:
//   q:           [total_q, n_heads, HD]                   bf16/fp16
//   k_cache:     [num_blocks, block_size, n_kv_heads, HD] bf16/fp16
//   v_cache:     [num_blocks, block_size, n_kv_heads, HD] bf16/fp16
//   out:         [total_q, n_heads, HD]                   bf16/fp16
//   block_table: [batch, max_blocks_per_seq]              i32
//   seq_lens:    [batch]                                  i32 (KV length per req)
//   query_start_loc: [batch+1]                            i32 (cu_seqlens_q)
//
// Workgroup: 32 threads, each WG processes PF_WG_Q_ROWS=128 query rows
// of one (req, head). Total WGs = batch × max_q_tiles_per_req × num_heads.
//
// Caller must ensure HD == 256.
// ============================================================
template<bool IS_BF16, bool CAUSAL>
inline void sdp_paged_prefill_dpas_host(
    const unsigned short* query_ptr,
    const unsigned short* key_cache_ptr,
    const unsigned short* value_cache_ptr,
    unsigned short* output_ptr,
    const int* block_table_ptr,
    const int* seq_lens_ptr,
    const int* query_start_loc_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_block, int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int num_tokens, int batch,
    sycl::queue& dpcpp_queue)
{
    // Each WG handles up to PF_WG_Q_ROWS query rows.
    int max_q_tiles_per_req = (num_tokens + (int)PF_WG_Q_ROWS - 1) / (int)PF_WG_Q_ROWS;
    int64_t total_wgs = (int64_t)batch * max_q_tiles_per_req * num_heads;
    sycl::nd_range<1> range(sycl::range<1>(total_wgs * 32), sycl::range<1>(32));

    dpcpp_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            sdp_paged_prefill_dpas<CAUSAL, IS_BF16>(
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                output_ptr,
                block_table_ptr,
                seq_lens_ptr,
                query_start_loc_ptr,
                num_heads, num_kv_heads, head_dim,
                block_size, max_blocks_per_seq,
                kv_stride_block, kv_stride_pos, kv_stride_head,
                attn_scale,
                num_tokens,
                max_q_tiles_per_req,
                batch,
                ndi);
        });
    });
}
