/* gdn_conv_fused.h — Fused Conv1d + GDN ESIMD kernel for Qwen3-Next-80B-A3B decode.
 *
 * v9: Multi-WG, single submit, reads directly from projected_states_qkvz.
 *   - Grid: N × HV WGs, WG_SIZE=32 (one WG per sequence × value_head)
 *   - Phase 1: All 32 threads compute conv1d
 *     - q/k threads (tid 0..4*H-1): 64 dims each
 *     - v threads (tid 4*H..31): 64 or 128 dims each (double_v mode for HV>8)
 *     - Reads x from qkvz at mapped offsets (no rearrange needed)
 *     - Store q/k/v to SLM
 *   - barrier()
 *   - Phase 2: First V/4 threads do GDN (V_PER_THREAD=4)
 *   - Phase 3: conv_state shift + z extraction (hv==0 only for shift)
 *
 * Eliminates ALL host-side tensor ops: no cat, no reshape, no contiguous.
 * Inputs: projected_states_qkvz [N, qkvz_dim] + projected_states_ba [N, 2*HV]
 * Outputs: core_attn_out [N, HV, V] + z_out [N, HV, V]
 */

#include "utils.h"

namespace xmem = sycl::ext::intel::experimental::esimd;

/* ---- ESIMD scalar math helpers ---- */
ESIMD_INLINE float esimd_expf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}
ESIMD_INLINE float esimd_logf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::log(v);
    return v[0];
}
ESIMD_INLINE float esimd_sqrtf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::sqrt(v);
    return v[0];
}

/* ---- LSC load/store helpers ---- */

ESIMD_INLINE simd<float, 64> lsc_load_state_64(const fp16* ptr) {
    return xmem::lsc_block_load<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::cached>(ptr);
}

ESIMD_INLINE void lsc_store_state_64(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}

/* ---- Dot product 128 (split lo/hi 64) ---- */

ESIMD_INLINE float gdn_dot128(simd<float, 64> a_lo, simd<float, 64> a_hi,
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

ESIMD_INLINE float gdn_load_fp16_scalar(const fp16* base, int64_t idx) {
    int64_t aligned = idx & ~15;
    int lane = (int)(idx & 15);
    simd<fp16, 16> chunk = block_load<fp16, 16>(base + aligned);
    simd<float, 16> chunk_f32 = chunk;
    return chunk_f32[lane];
}

/* ---- SLM layout per WG (byte offsets) ---- */
static constexpr int SLM_Q_LO = 0;       // 64 floats = 256 bytes
static constexpr int SLM_Q_HI = 256;
static constexpr int SLM_K_LO = 512;
static constexpr int SLM_K_HI = 768;
static constexpr int SLM_V    = 1024;    // 128 floats = 512 bytes
// Total: 1536 bytes, allocate 2048

/* ============================================================
 * v9 KERNEL: reads from projected_states_qkvz directly.
 * WG_SIZE=32 always. When HV>8, v-threads handle 128 elements each.
 * ============================================================ */
ESIMD_INLINE void gdn_conv_fused_kernel_v9(
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
    const int dim = 2 * H * gdn_K + HV * gdn_V;

    // double_v: v-threads handle 128 elements each instead of 64
    const bool double_v = (HV > (32 - 4 * H) / 2);  // true when HV > 8

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    const int ssm_idx = ssm_state_indices_ptr[seq_idx];

    // ---- Compute qkvz read offset and conv_state chunk_start ----
    const int group_dim = gdn_K + gdn_K + heads_per_group * gdn_V * 2;

    int qkvz_offset = 0;
    int chunk_start = 0;
    // hi-half offsets (only used by v-threads when double_v)
    int qkvz_offset_hi = 0;
    int chunk_start_hi = 0;

    if (tid < 2 * H) {
        // q region: tid 0..(2*H-1), 64 elements each
        int q_head = tid / 2;
        qkvz_offset = q_head * group_dim + (tid & 1) * 64;
        chunk_start = tid * 64;
    } else if (tid < 4 * H) {
        // k region: tid (2*H)..(4*H-1), 64 elements each
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_head * group_dim + gdn_K + (k_tid & 1) * 64;
        chunk_start = tid * 64;
    } else if (double_v) {
        // v region (double): tid (4*H)..31, 128 elements each (one full v_head)
        int v_tid = tid - 4 * H;
        int v_hv = v_tid;  // one thread per v_head
        int v_group = v_hv / heads_per_group;
        int v_lane = v_hv % heads_per_group;
        int base = v_group * group_dim + 2 * gdn_K + v_lane * gdn_V;
        qkvz_offset = base;
        qkvz_offset_hi = base + 64;
        chunk_start = 4 * H * 64 + v_tid * 128;
        chunk_start_hi = chunk_start + 64;
    } else {
        // v region (original): tid (4*H)..31, 64 elements each (half v_head)
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        int v_group = v_hv / heads_per_group;
        int v_lane = v_hv % heads_per_group;
        qkvz_offset = v_group * group_dim + 2 * gdn_K
                     + v_lane * gdn_V + (v_tid & 1) * 64;
        chunk_start = tid * 64;
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

    if (double_v && tid >= 4 * H) {
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
        if (tid == q_tid_lo)     slm_block_store<float, 64>(SLM_Q_LO, conv_result);
        if (tid == q_tid_lo + 1) slm_block_store<float, 64>(SLM_Q_HI, conv_result);

        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo)     slm_block_store<float, 64>(SLM_K_LO, conv_result);
        if (tid == k_tid_lo + 1) slm_block_store<float, 64>(SLM_K_HI, conv_result);

        if (double_v) {
            // One thread per v_head: tid == 4*H + hv writes both halves
            if (tid == 4 * H + hv) {
                slm_block_store<float, 64>(SLM_V, conv_result);
                slm_block_store<float, 64>(SLM_V + 256, conv_result_hi);
            }
        } else {
            // Two threads per v_head (original)
            const int v_tid_lo = 4 * H + 2 * hv;
            if (tid == v_tid_lo)     slm_block_store<float, 64>(SLM_V, conv_result);
            if (tid == v_tid_lo + 1) slm_block_store<float, 64>(SLM_V + 256, conv_result);
        }
    }

    barrier();

    // ---- Phase 2: GDN (V/4 threads, V_PER_THREAD=4) ----
    if (ssm_idx >= 0 && tid * 4 < gdn_V) {
        // Load q, k from SLM
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI);

        // L2 normalize q, k
        float q_inv = 1.0f / esimd_sqrtf(gdn_dot128(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
        float k_inv = 1.0f / esimd_sqrtf(gdn_dot128(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
        q_lo *= q_inv * attn_scale; q_hi *= q_inv * attn_scale;
        k_lo *= k_inv; k_hi *= k_inv;

        // Load v (4 values from SLM)
        const int vi0 = tid * 4;
        simd<float, 4> v_f32 = slm_block_load<float, 4>(SLM_V + vi0 * (int)sizeof(float));

        // Gating from projected_states_ba
        const float A_log_val = gdn_load_fp16_scalar(A_log_ptr, hv);
        const float dt_bias_val = gdn_load_fp16_scalar(dt_bias_ptr, hv);
        const float neg_exp_A = -esimd_expf(A_log_val);

        // ba layout: [b_grp0, a_grp0, b_grp1, a_grp1, ...]
        const int hpg = heads_per_group;
        const int lane = hv - i_h * hpg;
        const int b_col = i_h * 2 * hpg + lane;
        const int a_col = b_col + hpg;
        float a_val = gdn_load_fp16_scalar(ba_ptr, (int64_t)seq_idx * ba_stride0 + a_col);
        float b_val = gdn_load_fp16_scalar(ba_ptr, (int64_t)seq_idx * ba_stride0 + b_col);
        float x_gate = a_val + dt_bias_val;
        float sp = (x_gate > 20.0f) ? x_gate : esimd_logf(1.0f + esimd_expf(x_gate));
        float g = neg_exp_A * sp;
        float exp_g = esimd_expf(g);
        float beta = 1.0f / (1.0f + esimd_expf(-b_val));

        // State base for this head
        fp16* sstate_base = ssm_state_ptr +
            (int64_t)ssm_idx * ssm_stride0 + (int64_t)hv * gdn_V * gdn_K;

        // Process 4 V-rows in one batch
        fp16* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
        fp16* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;
        fp16* sr2 = sstate_base + (int64_t)(vi0 + 2) * gdn_K;
        fp16* sr3 = sstate_base + (int64_t)(vi0 + 3) * gdn_K;

        simd<float, 64> h0_lo = lsc_load_state_64(sr0);
        simd<float, 64> h0_hi = lsc_load_state_64(sr0 + 64);
        simd<float, 64> h1_lo = lsc_load_state_64(sr1);
        simd<float, 64> h1_hi = lsc_load_state_64(sr1 + 64);
        simd<float, 64> h2_lo = lsc_load_state_64(sr2);
        simd<float, 64> h2_hi = lsc_load_state_64(sr2 + 64);
        simd<float, 64> h3_lo = lsc_load_state_64(sr3);
        simd<float, 64> h3_hi = lsc_load_state_64(sr3 + 64);

        h0_lo *= exp_g; h0_hi *= exp_g;
        h1_lo *= exp_g; h1_hi *= exp_g;
        h2_lo *= exp_g; h2_hi *= exp_g;
        h3_lo *= exp_g; h3_hi *= exp_g;

        float kv0 = gdn_dot128(h0_lo, h0_hi, k_lo, k_hi);
        float kv1 = gdn_dot128(h1_lo, h1_hi, k_lo, k_hi);
        float kv2 = gdn_dot128(h2_lo, h2_hi, k_lo, k_hi);
        float kv3 = gdn_dot128(h3_lo, h3_hi, k_lo, k_hi);

        float d0 = (v_f32[0] - kv0) * beta;
        float d1 = (v_f32[1] - kv1) * beta;
        float d2 = (v_f32[2] - kv2) * beta;
        float d3 = (v_f32[3] - kv3) * beta;

        h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
        h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;
        h2_lo += d2 * k_lo; h2_hi += d2 * k_hi;
        h3_lo += d3 * k_lo; h3_hi += d3 * k_hi;

        simd<float, 4> o_acc;
        o_acc[0] = gdn_dot128(h0_lo, h0_hi, q_lo, q_hi);
        o_acc[1] = gdn_dot128(h1_lo, h1_hi, q_lo, q_hi);
        o_acc[2] = gdn_dot128(h2_lo, h2_hi, q_lo, q_hi);
        o_acc[3] = gdn_dot128(h3_lo, h3_hi, q_lo, q_hi);

        lsc_store_state_64(sr0, h0_lo);
        lsc_store_state_64(sr0 + 64, h0_hi);
        lsc_store_state_64(sr1, h1_lo);
        lsc_store_state_64(sr1 + 64, h1_hi);
        lsc_store_state_64(sr2, h2_lo);
        lsc_store_state_64(sr2 + 64, h2_hi);
        lsc_store_state_64(sr3, h3_lo);
        lsc_store_state_64(sr3 + 64, h3_hi);

        // Store GDN output
        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        xmem::lsc_block_store<fp16, 4,
            xmem::lsc_data_size::default_size,
            xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
            out, simd<fp16, 4>(o_acc));
    } else {
        // ssm_idx < 0: write zeros (eliminates need for caller .zero_())
        int vi0_z = tid * 4;
        fp16* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0_z;
        xmem::lsc_block_store<fp16, 4,
            xmem::lsc_data_size::default_size,
            xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
            out, simd<fp16, 4>(0.0f));
    }

    // ---- Phase 3: conv_state shift (hv==0 only to avoid cross-WG race) ----
    if (conv_idx >= 0 && hv == 0) {
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

    // ---- z extraction: v-threads copy z from qkvz to z_out ----
    if (tid >= 4 * H) {
        int v_tid = tid - 4 * H;
        if (double_v) {
            // One thread per v_head, two 64-element loads
            int v_hv = v_tid;
            int v_group = v_hv / heads_per_group;
            int v_lane = v_hv % heads_per_group;

            int z_base = v_group * group_dim + 2 * gdn_K
                        + heads_per_group * gdn_V  // skip v portion
                        + v_lane * gdn_V;

            simd<fp16, 64> z_lo = block_load<fp16, 64>(qkvz_row + z_base);
            simd<fp16, 64> z_hi = block_load<fp16, 64>(qkvz_row + z_base + 64);

            fp16* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V;
            block_store<fp16, 64>(z_dst, z_lo);
            block_store<fp16, 64>(z_dst + 64, z_hi);
        } else {
            // Two threads per v_head (original)
            int v_hv = v_tid / 2;
            int v_half = v_tid & 1;
            int v_group = v_hv / heads_per_group;
            int v_lane = v_hv % heads_per_group;

            int z_qkvz_offset = v_group * group_dim + 2 * gdn_K
                               + heads_per_group * gdn_V
                               + v_lane * gdn_V + v_half * 64;

            simd<fp16, 64> z_data = block_load<fp16, 64>(qkvz_row + z_qkvz_offset);

            fp16* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V + v_half * 64;
            block_store<fp16, 64>(z_dst, z_data);
        }
    }
}

/* ============================================================
 * Host Dispatcher
 * ============================================================ */
inline void gdn_conv_fused_host(
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
            gdn_conv_fused_kernel_v9(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, ssm_state_indices_ptr,
                output_ptr, z_out_ptr,
                N, H, HV, K, V, scale, conv_stride0, ssm_stride0, ndi);
        });
    });
}
