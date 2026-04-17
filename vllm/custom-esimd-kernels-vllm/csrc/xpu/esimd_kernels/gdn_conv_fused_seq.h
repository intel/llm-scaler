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
 * Thread→qkvz offset mapping for sequential layout (64 elements each):
 *   tid 0..7:   q region → head=tid/2, offset = head*K + (tid%2)*64
 *   tid 8..15:  k region → head=(tid-8)/2, offset = H*K + head*K + ((tid-8)%2)*64
 *   tid 16..31: v region → vhv=(tid-16)/2, offset = 2*H*K + vhv*V + ((tid-16)%2)*64
 *
 * z is at offset: 2*H*K + HV*V + hv*V + half*64
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
 * ============================================================ */
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
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);  // 0..31

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    const int ssm_idx = ssm_state_indices_ptr[seq_idx];

    // ---- Sequential layout base offsets ----
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;

    // ---- Compute qkvz read offset for this thread (SEQUENTIAL layout) ----
    int qkvz_offset;
    if (tid < 2 * H) {
        // q region: tid 0..(2*H-1)
        int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
    } else if (tid < 4 * H) {
        // k region: tid (2*H)..(4*H-1)
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
    } else {
        // v region: tid (4*H)..(32-1)
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
    }

    // conv_state is in mixed_qkv dim order, NOT interleaved.
    // For sequential layout, the thread's conv_state chunk matches qkvz_offset.
    const int chunk_start = qkvz_offset;

    // ---- Phase 1: Conv1d (all 32 threads, 64 dims each) ----
    const fp16* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    fp16* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    // Read x from qkvz at mapped offset (contiguous 64 fp16)
    simd<fp16, 64> x_fp16 = block_load<fp16, 64>(qkvz_row + qkvz_offset);
    simd<float, 64> x_f32 = x_fp16;

    // State and weight are in sequential dim order (chunk_start)
    simd<float, 64> s0 = block_load<fp16, 64>(cstate_base + 0 * 2048 + chunk_start);
    simd<float, 64> s1 = block_load<fp16, 64>(cstate_base + 1 * 2048 + chunk_start);
    simd<float, 64> s2 = block_load<fp16, 64>(cstate_base + 2 * 2048 + chunk_start);

    simd<fp16, 256> w_raw = block_load<fp16, 256>(conv_weight_ptr + (int64_t)chunk_start * 4);
    simd<float, 64> w0 = w_raw.select<64, 4>(0);
    simd<float, 64> w1 = w_raw.select<64, 4>(1);
    simd<float, 64> w2 = w_raw.select<64, 4>(2);
    simd<float, 64> w3 = w_raw.select<64, 4>(3);

    simd<float, 64> bias = block_load<fp16, 64>(conv_bias_ptr + chunk_start);
    simd<float, 64> conv_result = s0 * w0 + s1 * w1 + s2 * w2 + x_f32 * w3 + bias;

    // SiLU
    {
        simd<float, 64> neg_r = -conv_result;
        simd<float, 64> exp_neg = sycl::ext::intel::esimd::exp(neg_r);
        simd<float, 64> sigmoid_val = 1.0f / (1.0f + exp_neg);
        conv_result = conv_result * sigmoid_val;
    }

    // Store q/k/v to SLM (only the relevant threads)
    {
        const int q_tid_lo = 2 * i_h;
        if (tid == q_tid_lo)     slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
        if (tid == q_tid_lo + 1) slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);

        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo)     slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
        if (tid == k_tid_lo + 1) slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);

        const int v_tid_lo = 4 * H + 2 * hv;
        if (tid == v_tid_lo)     slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
        if (tid == v_tid_lo + 1) slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result);
    }

    barrier();

    // ---- Phase 2: GDN (all 32 threads, V_PER_THREAD=4) ----
    if (ssm_idx >= 0) {
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);

        float q_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
        float k_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
        q_lo *= q_inv * attn_scale; q_hi *= q_inv * attn_scale;
        k_lo *= k_inv; k_hi *= k_inv;

        const int vi0 = tid * 4;
        simd<float, 4> v_f32 = slm_block_load<float, 4>(SLM_V_SEQ + vi0 * (int)sizeof(float));

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

        simd<float, 4> o_acc;
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

        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        xmem::lsc_block_store<fp16, 4,
            xmem::lsc_data_size::default_size,
            xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
            out, simd<fp16, 4>(o_acc));
    } else {
        int vi0_z = tid * 4;
        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0_z;
        xmem::lsc_block_store<fp16, 4,
            xmem::lsc_data_size::default_size,
            xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
            out, simd<fp16, 4>(0.0f));
    }

    // ---- Phase 3: conv_state shift + z extraction ----
    if (conv_idx >= 0) {
        simd<float, 64> s1_copy = block_load<fp16, 64>(cstate_base + 1 * 2048 + chunk_start);
        simd<float, 64> s2_copy = block_load<fp16, 64>(cstate_base + 2 * 2048 + chunk_start);
        block_store<fp16, 64>(cstate_base + 0 * 2048 + chunk_start, simd<fp16, 64>(s1_copy));
        block_store<fp16, 64>(cstate_base + 1 * 2048 + chunk_start, simd<fp16, 64>(s2_copy));
        block_store<fp16, 64>(cstate_base + 2 * 2048 + chunk_start, x_fp16);
    }

    // ---- z extraction: v-threads copy z from SEQUENTIAL qkvz to z_out ----
    if (tid >= 4 * H) {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        int v_half = v_tid & 1;

        // z in sequential layout: z_base + hv*V + half*64
        int z_qkvz_offset = z_base + v_hv * gdn_V + v_half * 64;

        simd<fp16, 64> z_data = block_load<fp16, 64>(qkvz_row + z_qkvz_offset);

        fp16* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                    + (int64_t)v_hv * gdn_V + v_half * 64;
        block_store<fp16, 64>(z_dst, z_data);
    }
}

/* ============================================================
 * Host Dispatcher
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
    constexpr int WG_SIZE = 32;

    sycl::nd_range<3> Range(
        sycl::range<3>(N, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_kernel(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, ssm_state_indices_ptr,
                output_ptr, z_out_ptr,
                N, H, HV, K, V, scale, conv_stride0, ssm_stride0, ndi);
        });
    });
}
