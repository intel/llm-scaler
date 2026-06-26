/* chunk_gated_delta_rule_extend — bf16 variant.
 *
 * See extend.kernels.fp16.h for the math and layout description. This
 * file is a near-mechanical dtype swap from fp16 to bf16 for q/k/v/beta/
 * state/o. g stays fp32 (log-space decay).
 */

ESIMD_INLINE void chunkGatedDeltaRuleExtendBf16(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* gState,        // fp32
  uint8_t* betaState,     // fp32
  uint8_t* stateBuf,      // bf16 in/out
  uint8_t* oState,        // bf16 out
  uint32_t* cuSeqlens,
  uint32_t headQk,        // H_k: headV must be a multiple of this (GQA on GDN)
  uint32_t headV,
  uint32_t headDim,
  float    qScale,
  sycl::nd_item<2>& ndi)
{
  constexpr uint32_t SLM_V_PRED_OFFSET = 0;
  constexpr uint32_t SLM_O_REDUCE_OFFSET = 128 * sizeof(float);
  constexpr uint32_t SLM_SIZE = 128 * sizeof(float)
                              + 16 * 128 * sizeof(float);
  __ESIMD_NS::slm_init(SLM_SIZE);

  const uint32_t headIdx = ndi.get_group(0);
  const uint32_t seqIdx  = ndi.get_group(1);
  const uint32_t hh      = ndi.get_local_id(0);

  const uint32_t tStart = cuSeqlens[seqIdx];
  const uint32_t tEnd   = cuSeqlens[seqIdx + 1];
  const uint32_t nTokSeq = tEnd - tStart;
  if (nTokSeq == 0) return;

  const uint32_t qkTokStride = headQk * headDim;
  const uint32_t vTokStride  = headV  * headDim;
  const uint32_t gTokStride  = headV;
  const uint32_t stateHeadElems = headDim * headDim;
  const uint32_t stateSeqElems  = headV * stateHeadElems;
  // GQA on GDN: v-heads share k-heads (repeat = H_v / H_k).
  const uint32_t kHeadIdx = headIdx / (headV / headQk);

  bf16* qPtr     = (bf16*)qState;
  bf16* kPtr     = (bf16*)kState;
  bf16* vPtr     = (bf16*)vState;
  float* gPtr    = (float*)gState;
  float* betaPtr = (float*)betaState;
  float* sPtr    = (float*)stateBuf;  // fp32 state
  bf16* oPtr     = (bf16*)oState;

  float* sMyRows = sPtr
                 + seqIdx * stateSeqElems
                 + headIdx * stateHeadElems
                 + (hh * 8) * headDim;

  // Load fp32 state (8 × 128 floats = 4 KB).
  simd<float, 128 * 8> fp32InS_persistent;
  #pragma unroll
  for (int r = 0; r < 8; r++) {
    fp32InS_persistent.select<128, 1>(r * 128) =
        block_load<float, 128>(sMyRows + r * headDim);
  }

  namespace xens = sycl::ext::intel::experimental::esimd;
  for (uint32_t tRel = 0; tRel < nTokSeq; tRel++) {
    const uint32_t t = tStart + tRel;

    simd<bf16, 128> q_bf16 = block_load<bf16, 128>(qPtr + t * qkTokStride + kHeadIdx * headDim);
    simd<bf16, 128> k_bf16 = block_load<bf16, 128>(kPtr + t * qkTokStride + kHeadIdx * headDim);
    simd<bf16, 128> v_bf16 = block_load<bf16, 128>(vPtr + t * vTokStride  + headIdx  * headDim);

    // Prefetch next token's q, k, v while we process current token.
    // L2 miss on PTL is in the low hundreds of cycles; the decay + v_pred +
    // state update + output block below is ~2000-3000 cycles so the miss is
    // fully hidden.
    if (tRel + 1 < nTokSeq) {
      const uint32_t t_next = t + 1;
      xens::lsc_prefetch<bf16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(qPtr + t_next * qkTokStride + kHeadIdx * headDim);
      xens::lsc_prefetch<bf16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(kPtr + t_next * qkTokStride + kHeadIdx * headDim);
      xens::lsc_prefetch<bf16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(vPtr + t_next * vTokStride + headIdx * headDim);
    }

    simd<float, 128> q_f32 = q_bf16;
    simd<float, 128> k_f32 = k_bf16;
    float q_sumsq = sycl::ext::intel::esimd::detail::sum<float, float, 128>(q_f32 * q_f32);
    float k_sumsq = sycl::ext::intel::esimd::detail::sum<float, float, 128>(k_f32 * k_f32);
    float q_inv = 1.0f / __ESIMD_NS::sqrt(q_sumsq + 1e-6f);
    float k_inv = 1.0f / __ESIMD_NS::sqrt(k_sumsq + 1e-6f);
    q_f32 = q_f32 * (q_inv * qScale);
    k_f32 = k_f32 * k_inv;

    float g_f32    = gPtr[t * gTokStride + headIdx];
    float beta_f32 = betaPtr[t * gTokStride + headIdx];

    simd<float, 1> g_vec(g_f32);
    float decay = __ESIMD_NS::exp(g_vec)[0];

    // Apply decay in place (state lives in fp32 registers).
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      fp32InS_persistent.select<128, 1>(r * 128) =
          fp32InS_persistent.select<128, 1>(r * 128) * decay;
    }

    simd<float, 8> my_v_pred_slab;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      my_v_pred_slab[r] = sycl::ext::intel::esimd::detail::sum<float, float, 128>(row * k_f32);
    }
    slm_block_store<float, 8>(SLM_V_PRED_OFFSET + hh * 8 * sizeof(float), my_v_pred_slab);
    barrier();

    simd<float, 128> v_pred = slm_block_load<float, 128>(SLM_V_PRED_OFFSET);

    simd<float, 128> v_err;
    {
      simd<float, 128> v_f32 = v_bf16;
      v_err = (v_f32 - v_pred) * beta_f32;
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t globalRow = hh * 8 + r;
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      float v_err_scalar = v_err[globalRow];
      row = row + k_f32 * v_err_scalar;
      fp32InS_persistent.select<128, 1>(r * 128) = row;
    }

    simd<float, 8> o_scalars;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      o_scalars[r] = sycl::ext::intel::esimd::detail::sum<float, float, 128>(row * q_f32);
    }
    simd<bf16, 8> o_bf16 = o_scalars;
    block_store<bf16, 8>(oPtr + t * vTokStride + headIdx * headDim + hh * 8, o_bf16);
  }

  // Write fp32 state back as last_state.
  #pragma unroll
  for (int r = 0; r < 8; r++) {
    block_store<float, 128>(sMyRows + r * headDim,
                            fp32InS_persistent.select<128, 1>(r * 128));
  }
}
