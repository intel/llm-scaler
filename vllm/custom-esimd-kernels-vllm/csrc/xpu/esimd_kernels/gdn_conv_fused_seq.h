/* gdn_conv_fused_seq.h — Fused Conv1d + GDN ESIMD kernel for SEQUENTIAL qkvz layout.
 *
 * Variant of gdn_conv_fused.h for models where the GEMV output is in
 * sequential [q_all | k_all | v_all | z_all] layout (e.g. Qwen3.5-35B-A3B),
 * rather than the GQA-interleaved layout used by Qwen3-Next-80B.
 *
 * Sequential qkvz layout (per TP rank):
 *   [q(H*K) | k(H*K) | v(HV*V) | z(HV*V)]
 *   e.g. for H=4, HV=8, K=V=128: [q(512) | k(512) | v(1024) | z(1024)] = 3072
 *
 * Interleaved ba layout (same as 80B, produced by gather index):
 *   [b_g0(HPG), a_g0(HPG), b_g1(HPG), a_g1(HPG), ...]
 *
 * Sequential ba layout (direct GEMV output):
 *   [b_all(HV) | a_all(HV)]
 *
 * The kernel reads qkvz/ba at correct offsets for sequential layout.
 * Everything else (conv1d, GDN, state update, z extraction) is identical.
 *
 * WG_SIZE is a template parameter (32 or 64), selected by the host
 * dispatcher based on H/HV. WG=32 suffices when 4*H <= 16 (i.e. H<=4,
 * TP>=4); WG=64 is used when H is larger (e.g. H=8 with TP=2).
 *
 * Thread→qkvz offset mapping for sequential layout (WG_SIZE threads):
 *   tid 0..(2*H-1):     q region (64 elem each)
 *   tid (2*H)..(4*H-1): k region (64 elem each)
 *   tid (4*H)..(WG_SIZE-1): v region
 *     non-double_v (HV <= v_slots/2): 64 elem each, 2 threads per v_head
 *     double_v     (HV >  v_slots/2): 128 elem each, 1 thread per v_head
 *
 * z is at offset: 2*H*K + HV*V + hv*V [+ half*64]
 *
 * ba sequential layout:
 *   b_col = hv (lane within b_all)
 *   a_col = HV + hv (lane within a_all)
 */

#include "utils.h"

namespace xmem = sycl::ext::intel::experimental::esimd;

/* ---- ESIMD scalar math helpers (same as gdn_conv_fused.h) ---- */
ESIMD_INLINE float esimd_expf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}
ESIMD_INLINE float esimd_logf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::log(v);
    return v[0];
}
ESIMD_INLINE float esimd_sqrtf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::sqrt(v);
    return v[0];
}

/* ---- LSC load/store helpers ---- */
ESIMD_INLINE simd<float, 64> lsc_load_state_64_seq(const fp16* ptr) {
    return xmem::lsc_block_load<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::cached>(ptr);
}

ESIMD_INLINE void lsc_store_state_64_seq(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}

/* ---- Dot product 128 (split lo/hi 64) ---- */
ESIMD_INLINE float gdn_dot128_seq(simd<float, 64> a_lo, simd<float, 64> a_hi,
                                   simd<float, 64> b_lo, simd<float, 64> b_hi) {
    simd<float, 64> p_lo = a_lo * b_lo;
    simd<float, 64> p_hi = a_hi * b_hi;
    p_lo += p_hi;
    p_lo.select<32,1>(0) += p_lo.select<32,1>(32);
    p_lo.select<16,1>(0) += p_lo.select<16,1>(16);
    p_lo.select<8,1>(0) += p_lo.select<8,1>(8);
    p_lo.select<4,1>(0) += p_lo.select<4,1>(4);
    p_lo.select<2,1>(0) += p_lo.select<2,1>(2);
    return p_lo[0] + p_lo[1];
}

ESIMD_INLINE float gdn_load_fp16_scalar_seq(const fp16* base, int64_t idx) {
    int64_t aligned = idx & ~15;
    int lane = (int)(idx & 15);
    simd<fp16, 16> chunk = block_load<fp16, 16>(base + aligned);
    simd<float, 16> chunk_f32 = chunk;
    return chunk_f32[lane];
}

/* ---- SLM layout per WG (byte offsets, same as original) ---- */
static constexpr int SLM_Q_LO_SEQ = 0;
static constexpr int SLM_Q_HI_SEQ = 256;
static constexpr int SLM_K_LO_SEQ = 512;
static constexpr int SLM_K_HI_SEQ = 768;
static constexpr int SLM_V_SEQ    = 1024;

/* ============================================================
 * KERNEL: reads from SEQUENTIAL qkvz layout [q|k|v|z].
 * Template parameter WG_SIZE (32 or 64) controls thread count.
 * ============================================================ */
template<int WG_SIZE>
ESIMD_INLINE void gdn_conv_fused_seq_kernel(
    const fp16* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* __restrict__ conv_state_ptr,
    const fp16* __restrict__ conv_weight_ptr,
    const fp16* __restrict__ conv_bias_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    const fp16* __restrict__ A_log_ptr,
    const fp16* __restrict__ dt_bias_ptr,
    const fp16* __restrict__ ba_ptr,
    int64_t ba_stride0,
    fp16* __restrict__ ssm_state_ptr,
    const int* __restrict__ ssm_state_indices_ptr,
    fp16* __restrict__ output_ptr,
    fp16* __restrict__ z_out_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    float attn_scale, int64_t conv_stride0, int64_t ssm_stride0,
    int inline_conv_shift,   // 1 = do conv_state shift inline (safe when N*HV<=32)
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);  // 0..WG_SIZE-1

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;

    const int num_v_threads = WG_SIZE - 4 * H;
    // double_v: v-threads handle 128 elements each instead of 64
    const bool double_v = (HV > num_v_threads / 2);

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    const int ssm_idx = ssm_state_indices_ptr[seq_idx];

    // ---- Sequential layout base offsets ----
    const int dim = 2 * H * gdn_K + HV * gdn_V;  // conv_state row width
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;

    // ---- Compute qkvz read offset and conv_state chunk_start ----
    int qkvz_offset = 0;
    int chunk_start = 0;
    int qkvz_offset_hi = 0;
    int chunk_start_hi = 0;

    if (tid < 2 * H) {
        // q region: tid 0..(2*H-1), 64 elements each
        int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        // k region: tid (2*H)..(4*H-1), 64 elements each
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        // v region (double): tid (4*H)..(WG_SIZE-1), 128 elements each (one full v_head)
        int v_tid = tid - 4 * H;
        int v_hv = v_tid;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        // v region (original): tid (4*H)..(WG_SIZE-1), 64 elements each (half v_head)
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }

    // When HV < available v-thread slots, surplus v-threads have OOB offsets.
    // Clamp to valid range; their results are never stored.
    const bool v_oob = (tid >= 4 * H) &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));
    if (v_oob) {
        qkvz_offset = v_base;
        qkvz_offset_hi = v_base + 64;
        chunk_start = v_base;
        chunk_start_hi = v_base + 64;
    }

    // ---- Phase 1: Conv1d ----
    const fp16* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    fp16* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    // -- lo chunk (all threads) --
    simd<fp16, 64> x_fp16 = block_load<fp16, 64>(qkvz_row + qkvz_offset);
    simd<float, 64> x_f32 = x_fp16;

    simd<float, 64> s0 = block_load<fp16, 64>(cstate_base + 0 * dim + chunk_start);
    simd<float, 64> s1 = block_load<fp16, 64>(cstate_base + 1 * dim + chunk_start);
    simd<float, 64> s2 = block_load<fp16, 64>(cstate_base + 2 * dim + chunk_start);

    simd<fp16, 256> w_raw = block_load<fp16, 256>(conv_weight_ptr + (int64_t)chunk_start * 4);
    simd<float, 64> conv_result =
        s0 * w_raw.select<64, 4>(0) + s1 * w_raw.select<64, 4>(1) +
        s2 * w_raw.select<64, 4>(2) + x_f32 * w_raw.select<64, 4>(3) +
        (simd<float, 64>)block_load<fp16, 64>(conv_bias_ptr + chunk_start);

    // SiLU
    {
        simd<float, 64> exp_neg = sycl::ext::intel::esimd::exp(-conv_result);
        conv_result = conv_result / (1.0f + exp_neg);
    }

    // -- hi chunk (v-threads only, when double_v) --
    simd<fp16, 64> x_fp16_hi;
    simd<float, 64> s0_hi, s1_hi, s2_hi, conv_result_hi;

    if (double_v && tid >= 4 * H && !v_oob) {
        x_fp16_hi = block_load<fp16, 64>(qkvz_row + qkvz_offset_hi);
        simd<float, 64> x_f32_hi = x_fp16_hi;

        s0_hi = block_load<fp16, 64>(cstate_base + 0 * dim + chunk_start_hi);
        s1_hi = block_load<fp16, 64>(cstate_base + 1 * dim + chunk_start_hi);
        s2_hi = block_load<fp16, 64>(cstate_base + 2 * dim + chunk_start_hi);

        simd<fp16, 256> w_raw_hi = block_load<fp16, 256>(
            conv_weight_ptr + (int64_t)chunk_start_hi * 4);
        conv_result_hi =
            s0_hi * w_raw_hi.select<64, 4>(0) + s1_hi * w_raw_hi.select<64, 4>(1) +
            s2_hi * w_raw_hi.select<64, 4>(2) + x_f32_hi * w_raw_hi.select<64, 4>(3) +
            (simd<float, 64>)block_load<fp16, 64>(conv_bias_ptr + chunk_start_hi);

        // SiLU
        simd<float, 64> exp_neg_hi = sycl::ext::intel::esimd::exp(-conv_result_hi);
        conv_result_hi = conv_result_hi / (1.0f + exp_neg_hi);
    }

    // ---- Store q/k/v to SLM ----
    {
        const int q_tid_lo = 2 * i_h;
        if (tid == q_tid_lo)     slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
        if (tid == q_tid_lo + 1) slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);

        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo)     slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
        if (tid == k_tid_lo + 1) slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);

        if (double_v) {
            if (tid == 4 * H + hv) {
                slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
                slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result_hi);
            }
        } else {
            const int v_tid_lo = 4 * H + 2 * hv;
            if (tid == v_tid_lo)     slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
            if (tid == v_tid_lo + 1) slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result);
        }
    }

    barrier();

    // ---- Phase 2: GDN (all WG_SIZE threads) ----
    // VPT (V elements per thread): WG=32→4, WG=64→2
    constexpr int VPT = 128 / WG_SIZE;

    if (ssm_idx >= 0) {
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);

        float q_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
        float k_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
        q_lo *= q_inv * attn_scale; q_hi *= q_inv * attn_scale;
        k_lo *= k_inv; k_hi *= k_inv;

        const int vi0 = tid * VPT;
        simd<float, VPT> v_f32 = slm_block_load<float, VPT>(SLM_V_SEQ + vi0 * (int)sizeof(float));

        const float A_log_val = gdn_load_fp16_scalar_seq(A_log_ptr, hv);
        const float dt_bias_val = gdn_load_fp16_scalar_seq(dt_bias_ptr, hv);
        const float neg_exp_A = -esimd_expf_seq(A_log_val);

        // ---- ba: SEQUENTIAL layout [b_all(HV) | a_all(HV)] ----
        const int b_col = hv;
        const int a_col = HV + hv;
        float a_val = gdn_load_fp16_scalar_seq(ba_ptr, (int64_t)seq_idx * ba_stride0 + a_col);
        float b_val = gdn_load_fp16_scalar_seq(ba_ptr, (int64_t)seq_idx * ba_stride0 + b_col);
        float x_gate = a_val + dt_bias_val;
        float sp = (x_gate > 20.0f) ? x_gate : esimd_logf_seq(1.0f + esimd_expf_seq(x_gate));
        float g = neg_exp_A * sp;
        float exp_g = esimd_expf_seq(g);
        float beta = 1.0f / (1.0f + esimd_expf_seq(-b_val));

        fp16* sstate_base = ssm_state_ptr +
            (int64_t)ssm_idx * ssm_stride0 + (int64_t)hv * gdn_V * gdn_K;

        // Manual unroll to avoid ESIMD compiler codegen issues with
        // #pragma unroll on lsc load/store heavy loops.
        simd<float, VPT> o_acc;

        if constexpr (VPT == 4) {
            fp16* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
            fp16* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;
            fp16* sr2 = sstate_base + (int64_t)(vi0 + 2) * gdn_K;
            fp16* sr3 = sstate_base + (int64_t)(vi0 + 3) * gdn_K;

            simd<float, 64> h0_lo = lsc_load_state_64_seq(sr0);
            simd<float, 64> h0_hi = lsc_load_state_64_seq(sr0 + 64);
            simd<float, 64> h1_lo = lsc_load_state_64_seq(sr1);
            simd<float, 64> h1_hi = lsc_load_state_64_seq(sr1 + 64);
            simd<float, 64> h2_lo = lsc_load_state_64_seq(sr2);
            simd<float, 64> h2_hi = lsc_load_state_64_seq(sr2 + 64);
            simd<float, 64> h3_lo = lsc_load_state_64_seq(sr3);
            simd<float, 64> h3_hi = lsc_load_state_64_seq(sr3 + 64);

            h0_lo *= exp_g; h0_hi *= exp_g;
            h1_lo *= exp_g; h1_hi *= exp_g;
            h2_lo *= exp_g; h2_hi *= exp_g;
            h3_lo *= exp_g; h3_hi *= exp_g;

            float kv0 = gdn_dot128_seq(h0_lo, h0_hi, k_lo, k_hi);
            float kv1 = gdn_dot128_seq(h1_lo, h1_hi, k_lo, k_hi);
            float kv2 = gdn_dot128_seq(h2_lo, h2_hi, k_lo, k_hi);
            float kv3 = gdn_dot128_seq(h3_lo, h3_hi, k_lo, k_hi);

            float d0 = (v_f32[0] - kv0) * beta;
            float d1 = (v_f32[1] - kv1) * beta;
            float d2 = (v_f32[2] - kv2) * beta;
            float d3 = (v_f32[3] - kv3) * beta;

            h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
            h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;
            h2_lo += d2 * k_lo; h2_hi += d2 * k_hi;
            h3_lo += d3 * k_lo; h3_hi += d3 * k_hi;

            o_acc[0] = gdn_dot128_seq(h0_lo, h0_hi, q_lo, q_hi);
            o_acc[1] = gdn_dot128_seq(h1_lo, h1_hi, q_lo, q_hi);
            o_acc[2] = gdn_dot128_seq(h2_lo, h2_hi, q_lo, q_hi);
            o_acc[3] = gdn_dot128_seq(h3_lo, h3_hi, q_lo, q_hi);

            lsc_store_state_64_seq(sr0, h0_lo);
            lsc_store_state_64_seq(sr0 + 64, h0_hi);
            lsc_store_state_64_seq(sr1, h1_lo);
            lsc_store_state_64_seq(sr1 + 64, h1_hi);
            lsc_store_state_64_seq(sr2, h2_lo);
            lsc_store_state_64_seq(sr2 + 64, h2_hi);
            lsc_store_state_64_seq(sr3, h3_lo);
            lsc_store_state_64_seq(sr3 + 64, h3_hi);
        } else {
            // VPT == 2 (WG=64)
            fp16* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
            fp16* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;

            simd<float, 64> h0_lo = lsc_load_state_64_seq(sr0);
            simd<float, 64> h0_hi = lsc_load_state_64_seq(sr0 + 64);
            simd<float, 64> h1_lo = lsc_load_state_64_seq(sr1);
            simd<float, 64> h1_hi = lsc_load_state_64_seq(sr1 + 64);

            h0_lo *= exp_g; h0_hi *= exp_g;
            h1_lo *= exp_g; h1_hi *= exp_g;

            float kv0 = gdn_dot128_seq(h0_lo, h0_hi, k_lo, k_hi);
            float kv1 = gdn_dot128_seq(h1_lo, h1_hi, k_lo, k_hi);

            float d0 = (v_f32[0] - kv0) * beta;
            float d1 = (v_f32[1] - kv1) * beta;

            h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
            h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;

            o_acc[0] = gdn_dot128_seq(h0_lo, h0_hi, q_lo, q_hi);
            o_acc[1] = gdn_dot128_seq(h1_lo, h1_hi, q_lo, q_hi);

            lsc_store_state_64_seq(sr0, h0_lo);
            lsc_store_state_64_seq(sr0 + 64, h0_hi);
            lsc_store_state_64_seq(sr1, h1_lo);
            lsc_store_state_64_seq(sr1 + 64, h1_hi);
        }

        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        if constexpr (VPT >= 4) {
            xmem::lsc_block_store<fp16, VPT,
                xmem::lsc_data_size::default_size,
                xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
                out, simd<fp16, VPT>(o_acc));
        } else {
            block_store<fp16, VPT>(out, simd<fp16, VPT>(o_acc));
        }
    } else {
        int vi0_z = tid * VPT;
        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0_z;
        if constexpr (VPT >= 4) {
            xmem::lsc_block_store<fp16, VPT,
                xmem::lsc_data_size::default_size,
                xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
                out, simd<fp16, VPT>(0.0f));
        } else {
            block_store<fp16, VPT>(out, simd<fp16, VPT>(0.0f));
        }
    }

    // ---- Phase 3: conv_state shift (inline path, only when N*HV <= WG_SIZE) ----
    // When N*HV > WG_SIZE, the shift is done by a separate kernel to avoid a
    // cross-WG race: one WG's shift writes could land before another WG's
    // Phase 1 reads for the same seq_idx.
    // Uses register-cached s1, s2, x_fp16 from Phase 1 (not re-read from memory).
    if (inline_conv_shift && conv_idx >= 0 && hv == 0 && !v_oob) {
        // lo chunk (all threads)
        block_store<fp16, 64>(cstate_base + 0 * dim + chunk_start, simd<fp16, 64>(s1));
        block_store<fp16, 64>(cstate_base + 1 * dim + chunk_start, simd<fp16, 64>(s2));
        block_store<fp16, 64>(cstate_base + 2 * dim + chunk_start, x_fp16);

        // hi chunk (v-threads only, when double_v)
        if (double_v && tid >= 4 * H) {
            block_store<fp16, 64>(cstate_base + 0 * dim + chunk_start_hi, simd<fp16, 64>(s1_hi));
            block_store<fp16, 64>(cstate_base + 1 * dim + chunk_start_hi, simd<fp16, 64>(s2_hi));
            block_store<fp16, 64>(cstate_base + 2 * dim + chunk_start_hi, x_fp16_hi);
        }
    }

    // ---- z extraction: v-threads copy z from SEQUENTIAL qkvz to z_out ----
    if (tid >= 4 * H && !v_oob) {
        int v_tid = tid - 4 * H;
        if (double_v) {
            // One thread per v_head, two 64-element loads
            int v_hv = v_tid;
            int z_off = z_base + v_hv * gdn_V;

            simd<fp16, 64> z_lo = block_load<fp16, 64>(qkvz_row + z_off);
            simd<fp16, 64> z_hi = block_load<fp16, 64>(qkvz_row + z_off + 64);

            fp16* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V;
            block_store<fp16, 64>(z_dst, z_lo);
            block_store<fp16, 64>(z_dst + 64, z_hi);
        } else {
            // Two threads per v_head (original)
            int v_hv = v_tid / 2;
            int v_half = v_tid & 1;
            int z_qkvz_offset = z_base + v_hv * gdn_V + v_half * 64;

            simd<fp16, 64> z_data = block_load<fp16, 64>(qkvz_row + z_qkvz_offset);

            fp16* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V + v_half * 64;
            block_store<fp16, 64>(z_dst, z_data);
        }
    }
}

/* ============================================================
 * Conv-state shift kernel (sequential layout) — runs AFTER the
 * main fused kernel to avoid cross-WG race on conv_state.
 *
 * Grid: N × 1 × WG_SIZE  (one WG per sequence)
 * Each thread shifts its chunk: row0←row1, row1←row2, row2←x_new.
 * conv_state dim = 2*H*K + HV*V (computed, not hardcoded).
 * ============================================================ */
template<int WG_SIZE>
ESIMD_INLINE void conv_state_shift_seq_kernel(
    const fp16* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* __restrict__ conv_state_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    int64_t conv_stride0,
    nd_item<3>& ndi)
{
    const int seq_idx = ndi.get_group(0);
    const int tid = ndi.get_local_id(2);

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    if (conv_idx < 0) return;

    // Sequential layout offsets (same as main kernel)
    const int dim = 2 * H * gdn_K + HV * gdn_V;
    const int num_v_threads_s = WG_SIZE - 4 * H;
    const bool double_v = (HV > num_v_threads_s / 2);
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;

    int qkvz_offset = 0;
    int chunk_start = 0;
    int qkvz_offset_hi = 0;
    int chunk_start_hi = 0;

    if (tid < 2 * H) {
        int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }

    const bool v_oob_s = (tid >= 4 * H) &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));
    if (v_oob_s) return;

    const fp16* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    fp16* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    // lo chunk (all threads)
    simd<float, 64> s1_val = block_load<fp16, 64>(cstate_base + 1 * dim + chunk_start);
    simd<float, 64> s2_val = block_load<fp16, 64>(cstate_base + 2 * dim + chunk_start);
    simd<fp16, 64> x_new = block_load<fp16, 64>(qkvz_row + qkvz_offset);

    block_store<fp16, 64>(cstate_base + 0 * dim + chunk_start, simd<fp16, 64>(s1_val));
    block_store<fp16, 64>(cstate_base + 1 * dim + chunk_start, simd<fp16, 64>(s2_val));
    block_store<fp16, 64>(cstate_base + 2 * dim + chunk_start, x_new);

    // hi chunk (v-threads only, when double_v)
    if (double_v && tid >= 4 * H) {
        simd<float, 64> s1_hi = block_load<fp16, 64>(cstate_base + 1 * dim + chunk_start_hi);
        simd<float, 64> s2_hi = block_load<fp16, 64>(cstate_base + 2 * dim + chunk_start_hi);
        simd<fp16, 64> x_hi = block_load<fp16, 64>(qkvz_row + qkvz_offset_hi);

        block_store<fp16, 64>(cstate_base + 0 * dim + chunk_start_hi, simd<fp16, 64>(s1_hi));
        block_store<fp16, 64>(cstate_base + 1 * dim + chunk_start_hi, simd<fp16, 64>(s2_hi));
        block_store<fp16, 64>(cstate_base + 2 * dim + chunk_start_hi, x_hi);
    }
}

/* ============================================================
 * Templated dispatch helper — launches kernels for a given WG_SIZE.
 * ============================================================ */
template<int WG_SIZE>
inline void gdn_conv_fused_seq_dispatch(
    const fp16* qkvz_ptr, int64_t qkvz_stride0,
    fp16* conv_state_ptr, const fp16* conv_weight_ptr,
    const fp16* conv_bias_ptr, const int* conv_state_indices_ptr,
    const fp16* A_log_ptr, const fp16* dt_bias_ptr,
    const fp16* ba_ptr, int64_t ba_stride0,
    fp16* ssm_state_ptr, const int* ssm_state_indices_ptr,
    fp16* output_ptr, fp16* z_out_ptr,
    int N, int H, int HV, int K, int V, float scale,
    int64_t conv_stride0, int64_t ssm_stride0,
    sycl::queue& q)
{
    const int total_wgs = N * HV;
    const int inline_shift = (total_wgs <= WG_SIZE) ? 1 : 0;

    sycl::nd_range<3> Range(
        sycl::range<3>(N, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_kernel<WG_SIZE>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, ssm_state_indices_ptr,
                output_ptr, z_out_ptr,
                N, H, HV, K, V, scale, conv_stride0, ssm_stride0,
                inline_shift, ndi);
        });
    });

    if (!inline_shift) {
        sycl::nd_range<3> ShiftRange(
            sycl::range<3>(N, 1, WG_SIZE),
            sycl::range<3>(1, 1, WG_SIZE));

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(ShiftRange, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                conv_state_shift_seq_kernel<WG_SIZE>(
                    qkvz_ptr, qkvz_stride0, conv_state_ptr,
                    conv_state_indices_ptr,
                    N, H, HV, K, V,
                    conv_stride0, ndi);
            });
        });
    }
}

/* ============================================================
 * Host Dispatcher — selects WG_SIZE=32 or 64 based on H/HV.
 * ============================================================ */
inline void gdn_conv_fused_seq_host(
    const fp16* qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* conv_state_ptr,
    const fp16* conv_weight_ptr,
    const fp16* conv_bias_ptr,
    const int* conv_state_indices_ptr,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_stride0,
    fp16* ssm_state_ptr,
    const int* ssm_state_indices_ptr,
    fp16* output_ptr,
    fp16* z_out_ptr,
    int N, int H, int HV, int K, int V,
    float scale,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    sycl::queue& q)
{
    TORCH_CHECK(HV > 0 && HV % H == 0,
        "gdn_conv_fused_seq: HV (", HV, ") must be a positive multiple of H (", H, ")");
    TORCH_CHECK(K == 128 && V == 128,
        "gdn_conv_fused_seq: only K=128, V=128 supported, got K=", K, " V=", V);

    // Pick smallest WG_SIZE that has enough v-thread slots for HV.
    // WG=32: v_slots = 32-4*H, WG=64: v_slots = 64-4*H.
    const int v_slots_32 = 32 - 4 * H;
    if (v_slots_32 > 0 && HV <= v_slots_32) {
        gdn_conv_fused_seq_dispatch<32>(
            qkvz_ptr, qkvz_stride0, conv_state_ptr,
            conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
            A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
            ssm_state_ptr, ssm_state_indices_ptr,
            output_ptr, z_out_ptr,
            N, H, HV, K, V, scale, conv_stride0, ssm_stride0, q);
    } else {
        const int v_slots_64 = 64 - 4 * H;
        TORCH_CHECK(v_slots_64 > 0 && HV <= v_slots_64,
            "gdn_conv_fused_seq: HV (", HV, ") exceeds max v-thread slots even with WG_SIZE=64 (",
            v_slots_64, "). H=", H);
        gdn_conv_fused_seq_dispatch<64>(
            qkvz_ptr, qkvz_stride0, conv_state_ptr,
            conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
            A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
            ssm_state_ptr, ssm_state_indices_ptr,
            output_ptr, z_out_ptr,
            N, H, HV, K, V, scale, conv_stride0, ssm_stride0, q);
    }
}
