#pragma once

/*
 * Speculative variant of the sequential Qwen3.5/Qwen3.6 GDN kernel.
 *
 * The ordinary fused kernel maps one work-group to one token and cannot model
 * speculative rollback: every draft token has its own conv/SSM cache slot and
 * the initial slot depends on the number of tokens accepted in the previous
 * step. This variant maps one work-group to one speculative sequence and
 * walks all of its tokens in order, keeping the same state semantics as the
 * vllm-xpu speculative GDN kernels while avoiding the intermediate q/k/v/b/a
 * buffers and the separate conv and delta-rule launches.
 *
 * The implementation uses a fixed WG_SIZE=64 and K=V=128. It supports the
 * existing Qwen3.5/3.6 TP=2 geometries (H=8, HV=16/24) and, through the
 * versioned host entry point, the Qwen3.8 TP=8 geometry (H=2, HV=6).
 * Unsupported geometries are rejected rather than silently selecting a slower
 * or incorrect layout.
 */

template <int WG_SIZE>
ESIMD_INLINE simd<float, 2> gdn_spec_update_seq(
    const simd<float, 64>& q_lo,
    const simd<float, 64>& q_hi,
    const simd<float, 64>& k_lo,
    const simd<float, 64>& k_hi,
    const simd<float, 2>& v_f32,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_offset,
    fp16* ssm_state_ptr,
    int64_t ssm_stride0,
    int prev_state_idx,
    int save_state_idx,
    int tid,
    int hv,
    int HV,
    int gdn_K,
    int gdn_V,
    float attn_scale)
{
    float q_inv =
        1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
    float k_inv =
        1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
    simd<float, 64> qn_lo = q_lo * (q_inv * attn_scale);
    simd<float, 64> qn_hi = q_hi * (q_inv * attn_scale);
    simd<float, 64> kn_lo = k_lo * k_inv;
    simd<float, 64> kn_hi = k_hi * k_inv;

    const float A_log_val = gdn_load_fp16_scalar_seq(A_log_ptr, hv);
    const float dt_bias_val = gdn_load_fp16_scalar_seq(dt_bias_ptr, hv);
    const float neg_exp_A = -esimd_expf_seq(A_log_val);
    const int b_col = hv;
    const int a_col = HV + hv;
    const float b_val = gdn_load_fp16_scalar_seq(ba_ptr, ba_offset + b_col);
    const float a_val = gdn_load_fp16_scalar_seq(
        ba_ptr, ba_offset + a_col);
    const float x_gate = a_val + dt_bias_val;
    const float sp =
        (x_gate > 20.0f) ? x_gate : esimd_logf_seq(1.0f + esimd_expf_seq(x_gate));
    const float exp_g = esimd_expf_seq(neg_exp_A * sp);
    const float beta = 1.0f / (1.0f + esimd_expf_seq(-b_val));

    const int vi0 = tid * 2;
    fp16* state_base = nullptr;
    if (prev_state_idx >= 0) {
        state_base = ssm_state_ptr +
            (int64_t)prev_state_idx * ssm_stride0 +
            (int64_t)hv * gdn_V * gdn_K;
    }
    fp16* save_base = nullptr;
    if (save_state_idx >= 0) {
        save_base = ssm_state_ptr +
            (int64_t)save_state_idx * ssm_stride0 +
            (int64_t)hv * gdn_V * gdn_K;
    }

    simd<float, 64> h0_lo(0.0f), h0_hi(0.0f);
    simd<float, 64> h1_lo(0.0f), h1_hi(0.0f);
    if (state_base != nullptr) {
        fp16* sr0 = state_base + (int64_t)(vi0 + 0) * gdn_K;
        fp16* sr1 = state_base + (int64_t)(vi0 + 1) * gdn_K;
        h0_lo = lsc_load_state_64_seq(sr0);
        h0_hi = lsc_load_state_64_seq(sr0 + 64);
        h1_lo = lsc_load_state_64_seq(sr1);
        h1_hi = lsc_load_state_64_seq(sr1 + 64);
    }

    h0_lo *= exp_g;
    h0_hi *= exp_g;
    h1_lo *= exp_g;
    h1_hi *= exp_g;

    const float kv0 = gdn_dot128_seq(h0_lo, h0_hi, kn_lo, kn_hi);
    const float kv1 = gdn_dot128_seq(h1_lo, h1_hi, kn_lo, kn_hi);
    const float d0 = (v_f32[0] - kv0) * beta;
    const float d1 = (v_f32[1] - kv1) * beta;
    h0_lo += d0 * kn_lo;
    h0_hi += d0 * kn_hi;
    h1_lo += d1 * kn_lo;
    h1_hi += d1 * kn_hi;

    simd<float, 2> result;
    result[0] = gdn_dot128_seq(h0_lo, h0_hi, qn_lo, qn_hi);
    result[1] = gdn_dot128_seq(h1_lo, h1_hi, qn_lo, qn_hi);

    if (save_base != nullptr) {
        fp16* sr0 = save_base + (int64_t)(vi0 + 0) * gdn_K;
        fp16* sr1 = save_base + (int64_t)(vi0 + 1) * gdn_K;
        lsc_store_state_64_seq(sr0, h0_lo);
        lsc_store_state_64_seq(sr0 + 64, h0_hi);
        lsc_store_state_64_seq(sr1, h1_lo);
        lsc_store_state_64_seq(sr1 + 64, h1_hi);
    }
    return result;
}

template <int WG_SIZE>
ESIMD_INLINE void gdn_conv_fused_seq_spec_kernel(
    const fp16* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* __restrict__ conv_state_ptr,
    const fp16* __restrict__ conv_weight_ptr,
    const fp16* __restrict__ conv_bias_ptr,
    const int* __restrict__ spec_state_indices_ptr,
    const fp16* __restrict__ A_log_ptr,
    const fp16* __restrict__ dt_bias_ptr,
    const fp16* __restrict__ ba_ptr,
    int64_t ba_stride0,
    fp16* __restrict__ ssm_state_ptr,
    fp16* __restrict__ output_ptr,
    fp16* __restrict__ z_out_ptr,
    const int* __restrict__ token_indx_ptr,
    const int* __restrict__ num_accepted_tokens_ptr,
    int num_spec_decodes,
    int num_spec_tokens,
    int H,
    int HV,
    int gdn_K,
    int gdn_V,
    float attn_scale,
    int conv_state_len,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);
    if (seq_idx >= num_spec_decodes) {
        return;
    }

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;
    const int num_v_threads = WG_SIZE - 4 * H;
    const bool double_v = HV > num_v_threads / 2;
    const bool v_oob = tid >= 4 * H &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));

    const int dim = 2 * H * gdn_K + HV * gdn_V;
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;
    const int state_row = seq_idx * num_spec_tokens;

    int qkvz_offset = 0;
    int qkvz_offset_hi = 0;
    int chunk_start = 0;
    int chunk_start_hi = 0;
    if (tid < 2 * H) {
        const int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        const int k_tid = tid - 2 * H;
        const int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        const int v_hv = tid - 4 * H;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        const int v_tid = tid - 4 * H;
        const int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }
    if (v_oob) {
        qkvz_offset = v_base;
        qkvz_offset_hi = v_base + 64;
        chunk_start = v_base;
        chunk_start_hi = v_base + 64;
    }

    const bool packed_conv_state =
        conv_state_len >= num_spec_tokens + 2;
    const int accepted_prev = num_accepted_tokens_ptr[seq_idx] - 1;
    const int init_col = accepted_prev > 0 ? accepted_prev : 0;
    const int init_state_idx = spec_state_indices_ptr[
        state_row + (packed_conv_state ? 0 : init_col)];
    fp16* init_conv_state = nullptr;
    if (init_state_idx >= 0) {
        init_conv_state =
            conv_state_ptr + (int64_t)init_state_idx * conv_stride0 +
            (packed_conv_state ? (int64_t)init_col * dim : 0);
    }
    simd<float, 64> s0(0.0f), s1(0.0f), s2(0.0f);
    if (init_conv_state != nullptr) {
        s0 = block_load<fp16, 64>(init_conv_state + 0 * dim + chunk_start);
        s1 = block_load<fp16, 64>(init_conv_state + 1 * dim + chunk_start);
        s2 = block_load<fp16, 64>(init_conv_state + 2 * dim + chunk_start);
    }
    simd<float, 64> s0_hi(0.0f), s1_hi(0.0f), s2_hi(0.0f);
    if (double_v && tid >= 4 * H && !v_oob &&
        init_conv_state != nullptr) {
        s0_hi = block_load<fp16, 64>(
            init_conv_state + 0 * dim + chunk_start_hi);
        s1_hi = block_load<fp16, 64>(
            init_conv_state + 1 * dim + chunk_start_hi);
        s2_hi = block_load<fp16, 64>(
            init_conv_state + 2 * dim + chunk_start_hi);
    }

    for (int t = 0; t < num_spec_tokens; ++t) {
        const int global_t = token_indx_ptr[state_row + t];
        const int prev_col = t == 0
            ? init_col
            : t - 1;
        const int prev_state_idx =
            spec_state_indices_ptr[state_row + prev_col];
        const int save_state_idx = spec_state_indices_ptr[state_row + t];

        const fp16* qkvz_row =
            qkvz_ptr + (int64_t)global_t * qkvz_stride0;
        simd<fp16, 64> x_fp16 = block_load<fp16, 64>(
            qkvz_row + qkvz_offset);
        simd<float, 64> x_f32 = x_fp16;
        simd<fp16, 256> w_raw = block_load<fp16, 256>(
            conv_weight_ptr + (int64_t)chunk_start * 4);
        simd<float, 64> conv_result =
            s0 * w_raw.select<64, 4>(0) + s1 * w_raw.select<64, 4>(1) +
            s2 * w_raw.select<64, 4>(2) + x_f32 * w_raw.select<64, 4>(3) +
            (simd<float, 64>)block_load<fp16, 64>(
                conv_bias_ptr + chunk_start);
        conv_result = conv_result /
            (1.0f + sycl::ext::intel::esimd::exp(-conv_result));

        simd<fp16, 64> x_fp16_hi;
        simd<float, 64> conv_result_hi(0.0f);
        if (double_v && tid >= 4 * H && !v_oob) {
            x_fp16_hi = block_load<fp16, 64>(qkvz_row + qkvz_offset_hi);
            simd<float, 64> x_f32_hi = x_fp16_hi;
            simd<fp16, 256> w_raw_hi = block_load<fp16, 256>(
                conv_weight_ptr + (int64_t)chunk_start_hi * 4);
            conv_result_hi =
                s0_hi * w_raw_hi.select<64, 4>(0) +
                s1_hi * w_raw_hi.select<64, 4>(1) +
                s2_hi * w_raw_hi.select<64, 4>(2) +
                x_f32_hi * w_raw_hi.select<64, 4>(3) +
                (simd<float, 64>)block_load<fp16, 64>(
                    conv_bias_ptr + chunk_start_hi);
            conv_result_hi = conv_result_hi /
                (1.0f + sycl::ext::intel::esimd::exp(-conv_result_hi));
        }

        // v0.26 stores all speculative conv checkpoints in one wide cache
        // block. Older layouts use one three-row block per token.
        const int conv_save_state_idx = packed_conv_state
            ? spec_state_indices_ptr[state_row]
            : save_state_idx;
        if (conv_save_state_idx >= 0) {
            fp16* save_state =
                conv_state_ptr +
                (int64_t)conv_save_state_idx * conv_stride0;
            if (hv == 0 && tid < 4 * H) {
                if (!packed_conv_state || t == 0) {
                    block_store<fp16, 64>(
                        save_state + 0 * dim + chunk_start,
                        simd<fp16, 64>(s1));
                    block_store<fp16, 64>(
                        save_state + 1 * dim + chunk_start,
                        simd<fp16, 64>(s2));
                }
                block_store<fp16, 64>(
                    save_state +
                        (packed_conv_state ? 2 + t : 2) * dim +
                        chunk_start,
                    x_fp16);
            }
            if (!v_oob && tid >= 4 * H) {
                const int v_tid = tid - 4 * H;
                if (double_v && v_tid == hv) {
                    if (!packed_conv_state || t == 0) {
                        block_store<fp16, 64>(
                            save_state + 0 * dim + chunk_start,
                            simd<fp16, 64>(s1));
                        block_store<fp16, 64>(
                            save_state + 1 * dim + chunk_start,
                            simd<fp16, 64>(s2));
                        block_store<fp16, 64>(
                            save_state + 0 * dim + chunk_start_hi,
                            simd<fp16, 64>(s1_hi));
                        block_store<fp16, 64>(
                            save_state + 1 * dim + chunk_start_hi,
                            simd<fp16, 64>(s2_hi));
                    }
                    block_store<fp16, 64>(
                        save_state +
                            (packed_conv_state ? 2 + t : 2) * dim +
                            chunk_start,
                        x_fp16);
                    block_store<fp16, 64>(
                        save_state +
                            (packed_conv_state ? 2 + t : 2) * dim +
                            chunk_start_hi,
                        x_fp16_hi);
                } else if (!double_v && v_tid / 2 == hv) {
                    if (!packed_conv_state || t == 0) {
                        block_store<fp16, 64>(
                            save_state + 0 * dim + chunk_start,
                            simd<fp16, 64>(s1));
                        block_store<fp16, 64>(
                            save_state + 1 * dim + chunk_start,
                            simd<fp16, 64>(s2));
                    }
                    block_store<fp16, 64>(
                        save_state +
                            (packed_conv_state ? 2 + t : 2) * dim +
                            chunk_start,
                        x_fp16);
                }
            }
        }

        const int q_tid_lo = 2 * i_h;
        if (tid == q_tid_lo) {
            slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
        }
        if (tid == q_tid_lo + 1) {
            slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);
        }
        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo) {
            slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
        }
        if (tid == k_tid_lo + 1) {
            slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);
        }
        if (!v_oob && tid >= 4 * H) {
            const int v_tid = tid - 4 * H;
            if (double_v) {
                if (v_tid == hv) {
                    slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
                    slm_block_store<float, 64>(
                        SLM_V_SEQ + 256, conv_result_hi);
                }
            } else {
                const int v_hv = v_tid / 2;
                if (v_hv == hv) {
                    slm_block_store<float, 64>(
                        SLM_V_SEQ + (v_tid & 1) * 256, conv_result);
                }
            }
        }

        barrier();

        const int vi0 = tid * 2;
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);
        simd<float, 2> v_f32 =
            slm_block_load<float, 2>(SLM_V_SEQ + vi0 * (int)sizeof(float));

        const int64_t ba_offset = (int64_t)global_t * ba_stride0;
        simd<float, 2> o_acc = gdn_spec_update_seq<WG_SIZE>(
            q_lo, q_hi, k_lo, k_hi, v_f32, A_log_ptr, dt_bias_ptr,
            ba_ptr, ba_offset, ssm_state_ptr, ssm_stride0,
            prev_state_idx, save_state_idx, tid, hv, HV, gdn_K, gdn_V,
            attn_scale);

        fp16* out = output_ptr + (int64_t)global_t * HV * gdn_V +
            (int64_t)hv * gdn_V + vi0;
        block_store<fp16, 2>(out, simd<fp16, 2>(o_acc));

        if (tid < 2) {
            const int z_off = z_base + hv * gdn_V + tid * 64;
            simd<fp16, 64> z_data =
                block_load<fp16, 64>(qkvz_row + z_off);
            fp16* z_dst = z_out_ptr + (int64_t)global_t * HV * gdn_V +
                (int64_t)hv * gdn_V + tid * 64;
            block_store<fp16, 64>(z_dst, z_data);
        }

        s0 = s1;
        s1 = s2;
        s2 = x_f32;
        if (double_v && tid >= 4 * H && !v_oob) {
            s0_hi = s1_hi;
            s1_hi = s2_hi;
            s2_hi = x_fp16_hi;
        }

        barrier();
    }
}

inline void gdn_conv_fused_seq_spec_host(
    const fp16* qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* conv_state_ptr,
    const fp16* conv_weight_ptr,
    const fp16* conv_bias_ptr,
    const int* spec_state_indices_ptr,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_stride0,
    fp16* ssm_state_ptr,
    fp16* output_ptr,
    fp16* z_out_ptr,
    const int* token_indx_ptr,
    const int* num_accepted_tokens_ptr,
    int num_spec_decodes,
    int num_spec_tokens,
    int H,
    int HV,
    int K,
    int V,
    float scale,
    int conv_state_len,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    sycl::queue& q,
    bool allow_qwen38_tp8 = false)
{
    const bool legacy_geometry =
        H == 8 && (HV == 16 || HV == 24) && K == 128 && V == 128;
    const bool qwen38_tp8_geometry =
        H == 2 && HV == 6 && K == 128 && V == 128;
    if (allow_qwen38_tp8) {
        TORCH_CHECK(
            legacy_geometry || qwen38_tp8_geometry,
            "gdn_conv_fused_seq_spec_v2 supports H=8, HV=16/24 or H=2, "
            "HV=6, K=V=128; got H=", H, " HV=", HV, " K=", K,
            " V=", V);
    } else {
        TORCH_CHECK(
            legacy_geometry,
            "gdn_conv_fused_seq_spec supports H=8, HV=16/24, K=V=128; "
            "got H=", H, " HV=", HV, " K=", K, " V=", V);
    }
    TORCH_CHECK(num_spec_decodes > 0 && num_spec_tokens > 0,
        "speculative GDN dimensions must be positive");
    TORCH_CHECK(
        conv_state_len == 3 || conv_state_len >= num_spec_tokens + 2,
        "speculative GDN conv state must have 3 rows or at least ",
        num_spec_tokens + 2, "; got ", conv_state_len);

    constexpr int WG_SIZE = 64;
    sycl::nd_range<3> range(
        sycl::range<3>(num_spec_decodes, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_spec_kernel<WG_SIZE>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, spec_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, output_ptr, z_out_ptr, token_indx_ptr,
                num_accepted_tokens_ptr, num_spec_decodes, num_spec_tokens,
                H, HV, K, V, scale, conv_state_len, conv_stride0,
                ssm_stride0, ndi);
        });
    });
}

// V2 owns each convolution feature exactly once. No recurrent work-group
// reads mutable conv history: the next submission consumes FP16 q/k/v only.
inline sycl::event gdn_spec_v2_conv_host(
    const fp16* qkvz, int64_t qkvz_stride, fp16* conv,
    const fp16* weight, const fp16* bias, const int* indices,
    const int* tokens, const int* accepted, fp16* qkv, fp16* z,
    int sequences, int M, int H, int HV, int conv_len,
    int64_t conv_stride, sycl::queue& queue)
{
    const int dim = (2 * H + HV) * 128;
    const int chunks = dim / 64;
    const int groups = (chunks + 15) / 16;
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(sequences, groups * 16),
                              sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                const int seq = item.get_global_id(0);
                const int chunk = item.get_global_id(1);
                if (chunk >= chunks) return;
                const int feature = chunk * 64;
                const int row = seq * M;
                const int initial_col = accepted[seq] - 1;
                if (initial_col < 0 || initial_col >= M) return;
                const bool packed = conv_len != 3;
                const int initial_idx = indices[row + (packed ? 0 : initial_col)];
                // Standard Triton uses NULL_BLOCK_ID=0, not the old wheel's -1.
                if (initial_idx <= 0) return;
                fp16* source = conv + (int64_t)initial_idx * conv_stride + feature;
                const int offset = packed ? initial_col : 0;
                simd<float, 64> s0 = block_load<fp16, 64>(source + offset * dim);
                simd<float, 64> s1 = block_load<fp16, 64>(source + (offset + 1) * dim);
                simd<float, 64> s2 = block_load<fp16, 64>(source + (offset + 2) * dim);
                simd<fp16, 256> w = block_load<fp16, 256>(weight + feature * 4);
                const simd<float, 64> w0 = w.select<64, 4>(0);
                const simd<float, 64> w1 = w.select<64, 4>(1);
                const simd<float, 64> w2 = w.select<64, 4>(2);
                const simd<float, 64> w3 = w.select<64, 4>(3);
                const simd<float, 64> bias_value = block_load<fp16, 64>(bias + feature);
                // Triton caps effective state_len at M+2 even when the physical
                // cache has more rows; the unused suffix must remain untouched.
                const int retained = 2;
                if (packed) {
                    // Forward copy is safe: every source row is strictly after
                    // its destination, and this work-item is the sole owner.
                    for (int j = 0; j < retained; ++j) {
                        const auto history = block_load<fp16, 64>(
                            source + (offset + 1 + j) * dim);
                        block_store<fp16, 64>(source + j * dim, history);
                    }
                }
                for (int t = 0; t < M; ++t) {
                    const int global_t = tokens[row + t];
                    const fp16* input = qkvz + (int64_t)global_t * qkvz_stride;
                    const simd<fp16, 64> x16 = block_load<fp16, 64>(input + feature);
                    const simd<float, 64> x = x16;
                    simd<float, 64> acc = bias_value;
                    // Standard Triton multiplies FP16 operands in FP16 before
                    // extending each product into the FP32 accumulator.
                    acc += simd<float, 64>(simd<fp16, 64>(s0 * w0));
                    acc += simd<float, 64>(simd<fp16, 64>(s1 * w1));
                    acc += simd<float, 64>(simd<fp16, 64>(s2 * w2));
                    acc += simd<float, 64>(simd<fp16, 64>(x * w3));
                    acc = acc / (1.0f + sycl::ext::intel::esimd::exp(-acc));
                    block_store<fp16, 64>(
                        qkv + (int64_t)(row + t) * dim + feature,
                        simd<fp16, 64>(acc));
                    const int save_idx = packed ? initial_idx : indices[row + t];
                    if (save_idx > 0) {
                        fp16* dest = conv + (int64_t)save_idx * conv_stride + feature;
                        if (!packed) {
                            block_store<fp16, 64>(dest, simd<fp16, 64>(s1));
                            block_store<fp16, 64>(dest + dim, simd<fp16, 64>(s2));
                        }
                        block_store<fp16, 64>(
                            dest + (packed ? retained + t : 2) * dim, x16);
                    }
                    if (feature >= 2 * H * 128) {
                        const int z_feature = feature - 2 * H * 128;
                        block_store<fp16, 64>(
                            z + (int64_t)global_t * HV * 128 + z_feature,
                            block_load<fp16, 64>(input + dim + z_feature));
                    }
                    s0 = s1;
                    s1 = s2;
                    s2 = x;
                }
            });
    });
}

template <typename AType>
inline void gdn_spec_v2_recurrent_host(
    const fp16* qkv, const AType* A_log, const fp16* dt_bias,
    const fp16* ba, int64_t ba_stride, fp16* state, int64_t state_stride,
    fp16* output, const int* indices, const int* tokens, const int* accepted,
    int sequences, int M, int H, int HV, float scale,
    const sycl::event& conv_ready, sycl::queue& queue)
{
    const int dim = (2 * H + HV) * 128;
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(conv_ready);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(sequences, HV, 64),
                              sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
                const int seq = item.get_group(0);
                const int hv = item.get_group(1);
                const int vi = item.get_local_id(2) * 2;
                const int kh = hv / (HV / H);
                const int row = seq * M;
                const int initial_col = accepted[seq] - 1;
                if (initial_col < 0 || initial_col >= M) return;
                const int initial_idx = indices[row + initial_col];
                if (initial_idx <= 0) return;
                const int64_t head_offset = (int64_t)hv * 128 * 128 + vi * 128;
                const fp16* initial = state + (int64_t)initial_idx * state_stride + head_offset;
                // Keep these FP32 registers alive for the entire sequence.
                simd<float, 64> h0_lo = lsc_load_state_64_seq(initial);
                simd<float, 64> h0_hi = lsc_load_state_64_seq(initial + 64);
                simd<float, 64> h1_lo = lsc_load_state_64_seq(initial + 128);
                simd<float, 64> h1_hi = lsc_load_state_64_seq(initial + 192);
                const float neg_A = -esimd_expf_seq((float)A_log[hv]);
                const float bias = (float)dt_bias[hv];
                for (int t = 0; t < M; ++t) {
                    const int global_t = tokens[row + t];
                    const fp16* input = qkv + (int64_t)(row + t) * dim;
                    simd<float, 64> q0 = block_load<fp16, 64>(input + kh * 128);
                    simd<float, 64> q1 = block_load<fp16, 64>(input + kh * 128 + 64);
                    simd<float, 64> k0 = block_load<fp16, 64>(input + (H + kh) * 128);
                    simd<float, 64> k1 = block_load<fp16, 64>(input + (H + kh) * 128 + 64);
                    const float q_inv = 1.0f / esimd_sqrtf_seq(
                        gdn_dot128_seq(q0, q1, q0, q1) + 1e-6f);
                    const float k_inv = 1.0f / esimd_sqrtf_seq(
                        gdn_dot128_seq(k0, k1, k0, k1) + 1e-6f);
                    q0 *= q_inv * scale;
                    q1 *= q_inv * scale;
                    k0 *= k_inv;
                    k1 *= k_inv;
                    const simd<float, 2> v = block_load<fp16, 2>(
                        input + (2 * H + hv) * 128 + vi);
                    const float b = (float)ba[(int64_t)global_t * ba_stride + hv];
                    const float x = (float)ba[(int64_t)global_t * ba_stride + HV + hv] + bias;
                    const float softplus = x > 20.0f ? x : esimd_logf_seq(1.0f + esimd_expf_seq(x));
                    const float decay = esimd_expf_seq(neg_A * softplus);
                    const float beta = 1.0f / (1.0f + esimd_expf_seq(-b));
                    h0_lo *= decay;
                    h0_hi *= decay;
                    h1_lo *= decay;
                    h1_hi *= decay;
                    const float d0 = (v[0] - gdn_dot128_seq(h0_lo, h0_hi, k0, k1)) * beta;
                    const float d1 = (v[1] - gdn_dot128_seq(h1_lo, h1_hi, k0, k1)) * beta;
                    h0_lo += d0 * k0;
                    h0_hi += d0 * k1;
                    h1_lo += d1 * k0;
                    h1_hi += d1 * k1;
                    simd<float, 2> result;
                    result[0] = gdn_dot128_seq(h0_lo, h0_hi, q0, q1);
                    result[1] = gdn_dot128_seq(h1_lo, h1_hi, q0, q1);
                    block_store<fp16, 2>(
                        output + (int64_t)global_t * HV * 128 + hv * 128 + vi,
                        simd<fp16, 2>(result));
                    const int save_idx = indices[row + t];
                    if (save_idx > 0) {
                        fp16* dest = state + (int64_t)save_idx * state_stride + head_offset;
                        lsc_store_state_64_seq(dest, h0_lo);
                        lsc_store_state_64_seq(dest + 64, h0_hi);
                        lsc_store_state_64_seq(dest + 128, h1_lo);
                        lsc_store_state_64_seq(dest + 192, h1_hi);
                    }
                }
            });
    });
}
