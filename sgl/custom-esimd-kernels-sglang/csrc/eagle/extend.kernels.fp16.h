/* chunk_gated_delta_rule_extend — ESIMD kernel for GDN prefill extend.
 *
 * Replaces chunk_torch_xpu.chunk_gated_delta_rule_torch for dense GDN
 * (H_k == H_v, head_k_dim == head_v_dim == 128). Qwen3.5-0.8B fits.
 *
 * Math (per token) — matches chunk_gated_delta_rule_torch:
 *     q_hat = l2norm(q) * scale      (scale = 1/sqrt(K))
 *     k_hat = l2norm(k)
 *     S *= exp(g)
 *     v_err = v - S @ k_hat
 *     S += β * outer(v_err, k_hat)
 *     o  = S @ q_hat
 *
 * Grid/thread layout:
 *     global_range = (H_v * threads_per_head, n_seqs)
 *     local_range  = (threads_per_head, 1)
 *     group(0) = headIdx ∈ [0, H_v)
 *     group(1) = seqIdx
 *     local_id(0) = hh ∈ [0, threads_per_head)
 *
 * With threads_per_head = 16 and V = K = 128, each thread owns 8 contiguous
 * rows of the state matrix S[V, K]:
 *     rows [hh*8, hh*8 + 8) × cols [0, K)
 *
 * Tensor shapes (contiguous bf16/fp16 unless noted):
 *     q, k       : [T_total, H_k, K]        fp16
 *     v          : [T_total, H_v, V]        fp16
 *     g          : [T_total, H_v]           fp32
 *     beta       : [T_total, H_v]           fp16
 *     initial_state / last_state : [n_seqs, H_v, V, K] fp16 (same buffer)
 *     o          : [T_total, H_v, V]        fp16
 *     cu_seqlens : [n_seqs + 1]             int32
 */

// StateT = persisted SSM state dtype (float | sycl::half). The recurrence
// math always runs in fp32; only the load/store of the persistent state
// buffer is in StateT (matches mamba_ssm_dtype: fp32 or fp16).
template <typename StateT>
ESIMD_INLINE void chunkGatedDeltaRuleExtendFp16(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* gState,        // fp32 log-space decay
  uint8_t* betaState,     // fp32 post-sigmoid
  uint8_t* stateBuf,      // StateT in/out: initial_state on entry, last_state on exit
  uint8_t* oState,
  uint32_t* cuSeqlens,
  uint32_t headQk,        // H_k: headV must be a multiple of this (GQA on GDN)
  uint32_t headV,
  uint32_t headDim,       // = 128
  float    qScale,
  sycl::nd_item<2>& ndi)
{
  // SLM budget: 16 threads * 128 floats for output reduction +
  //             128 floats for v_pred buffer.
  constexpr uint32_t SLM_V_PRED_OFFSET = 0;                           // 128 floats
  constexpr uint32_t SLM_O_REDUCE_OFFSET = 128 * sizeof(float);       // 16 * 128 floats
  constexpr uint32_t SLM_SIZE = 128 * sizeof(float)
                              + 16 * 128 * sizeof(float);             // = 8704 B
  __ESIMD_NS::slm_init(SLM_SIZE);

  const uint32_t headIdx = ndi.get_group(0);
  const uint32_t seqIdx  = ndi.get_group(1);
  const uint32_t hh      = ndi.get_local_id(0);

  const uint32_t tStart = cuSeqlens[seqIdx];
  const uint32_t tEnd   = cuSeqlens[seqIdx + 1];
  const uint32_t nTokSeq = tEnd - tStart;
  if (nTokSeq == 0) return;

  // --- Pointer / offset bookkeeping ------------------------------------
  // Element (not byte) strides.
  const uint32_t qkTokStride = headQk * headDim;  // per-token stride on q/k (fp16 elements)
  const uint32_t vTokStride  = headV  * headDim;
  const uint32_t gTokStride  = headV;             // per-token stride on g (fp32 elements) / beta (fp16 elements)
  const uint32_t stateHeadElems = headDim * headDim;                      // per head, in StateT elements
  const uint32_t stateSeqElems  = headV * stateHeadElems;                 // per seq, in StateT elements

  // GQA on GDN: multiple v-heads share one k-head (repeat = H_v / H_k).
  // H_k == H_v is the common case (repeat = 1, "dense GDN").
  const uint32_t kHeadIdx = headIdx / (headV / headQk);

  fp16* qPtr     = (fp16*)qState;
  fp16* kPtr     = (fp16*)kState;
  fp16* vPtr     = (fp16*)vState;
  float* gPtr    = (float*)gState;
  float* betaPtr = (float*)betaState;
  StateT* sPtr   = (StateT*)stateBuf;
  fp16* oPtr     = (fp16*)oState;

  // Base pointer to this thread's 8-row slab of the (seq, head) state:
  // state[seq, head][hh*8 : hh*8 + 8, :] contiguous = headDim * 8 StateT elements.
  // Element counts are dtype-independent; only the element size differs.
  StateT* sMyRows = sPtr
                 + seqIdx * stateSeqElems
                 + headIdx * stateHeadElems
                 + (hh * 8) * headDim;

  // Load our 8 state rows (128 elems × 8 rows). State always held in fp32
  // registers; StateT=half upconverts to float on assign, StateT=float no-op.
  // block_load max per call is 128 elements; loop 8 times.
  simd<float, 128 * 8> fp32InS_persistent;
  #pragma unroll
  for (int r = 0; r < 8; r++) {
    simd<StateT, 128> raw = block_load<StateT, 128>(sMyRows + r * headDim);
    fp32InS_persistent.select<128, 1>(r * 128) = raw;
  }

  // ---- Per-token loop --------------------------------------------------
  namespace xens = sycl::ext::intel::experimental::esimd;
  for (uint32_t tRel = 0; tRel < nTokSeq; tRel++) {
    const uint32_t t = tStart + tRel;

    // --- Load q, k, v for the token's head -----------------------------
    simd<fp16, 128> q_fp16 = block_load<fp16, 128>(qPtr + t * qkTokStride + kHeadIdx * headDim);
    simd<fp16, 128> k_fp16 = block_load<fp16, 128>(kPtr + t * qkTokStride + kHeadIdx * headDim);
    simd<fp16, 128> v_fp16 = block_load<fp16, 128>(vPtr + t * vTokStride  + headIdx  * headDim);

    // Prefetch next token's q, k, v while we process current token.
    if (tRel + 1 < nTokSeq) {
      const uint32_t t_next = t + 1;
      xens::lsc_prefetch<fp16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(qPtr + t_next * qkTokStride + kHeadIdx * headDim);
      xens::lsc_prefetch<fp16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(kPtr + t_next * qkTokStride + kHeadIdx * headDim);
      xens::lsc_prefetch<fp16, 128,
        xens::lsc_data_size::default_size,
        xens::cache_hint::cached,
        xens::cache_hint::cached>(vPtr + t_next * vTokStride + headIdx * headDim);
    }

    // --- l2norm q and k (in fp32) --------------------------------------
    simd<float, 128> q_f32 = q_fp16;
    simd<float, 128> k_f32 = k_fp16;
    float q_sumsq = sycl::ext::intel::esimd::detail::sum<float, float, 128>(q_f32 * q_f32);
    float k_sumsq = sycl::ext::intel::esimd::detail::sum<float, float, 128>(k_f32 * k_f32);
    float q_inv = 1.0f / __ESIMD_NS::sqrt(q_sumsq + 1e-6f);
    float k_inv = 1.0f / __ESIMD_NS::sqrt(k_sumsq + 1e-6f);
    q_f32 = q_f32 * (q_inv * qScale);
    k_f32 = k_f32 * k_inv;

    // --- Load per-head scalars g, beta (both fp32 from fused_gdn_gating) ---
    float g_f32    = gPtr[t * gTokStride + headIdx];
    float beta_f32 = betaPtr[t * gTokStride + headIdx];

    // decay = exp(g)
    simd<float, 1> g_vec(g_f32);
    float decay = __ESIMD_NS::exp(g_vec)[0];

    // --- Apply decay to our 8 rows in place ----------------------------
    // State lives fully in fp32 registers across tokens now.
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      fp32InS_persistent.select<128, 1>(r * 128) =
          fp32InS_persistent.select<128, 1>(r * 128) * decay;
    }

    // --- Compute v_pred: each thread contributes v_pred[hh*8 : hh*8+8] --
    // v_pred[row] = Σ_c fp32InS[row, c] * k_hat[c]
    simd<float, 8> my_v_pred_slab;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      my_v_pred_slab[r] = sycl::ext::intel::esimd::detail::sum<float, float, 128>(row * k_f32);
    }
    slm_block_store<float, 8>(SLM_V_PRED_OFFSET + hh * 8 * sizeof(float), my_v_pred_slab);
    barrier();

    // All threads read the full v_pred [128 floats].
    simd<float, 128> v_pred = slm_block_load<float, 128>(SLM_V_PRED_OFFSET);

    // --- v_err = (v - v_pred) * beta -----------------------------------
    simd<float, 128> v_err;
    {
      simd<float, 128> v_f32 = v_fp16;
      v_err = (v_f32 - v_pred) * beta_f32;
    }

    // --- State update: state[r, :] += v_err[r] * k_hat ------------------
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      const uint32_t globalRow = hh * 8 + r;
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      float v_err_scalar = v_err[globalRow];
      row = row + k_f32 * v_err_scalar;
      fp32InS_persistent.select<128, 1>(r * 128) = row;
    }

    // --- Output: o[v] = Σ_c state[v, c] * q_hat[c] ----------------------
    // Correct interpretation: o = state @ q_hat, where state[V, K] and
    // q_hat[K] — so o[v] is a dot product over K.  Each thread owns
    // 8 rows of V, so it computes 8 scalars of o directly.  No cross-
    // thread reduction needed.
    simd<float, 8> o_scalars;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
      simd<float, 128> row = fp32InS_persistent.select<128, 1>(r * 128);
      o_scalars[r] = sycl::ext::intel::esimd::detail::sum<float, float, 128>(row * q_f32);
    }
    // Write this thread's 8 output scalars to the correct slice of o[V=128].
    // Output position: o[head_base + hh*8 .. hh*8+7].
    simd<fp16, 8> o_fp16 = o_scalars;
    block_store<fp16, 8>(oPtr + t * vTokStride + headIdx * headDim + hh * 8, o_fp16);

    // state stays in fp32 registers (fp32InS_persistent) across iterations.
  }

  // ---- Phase 3: write final state back to stateBuf as last_state (StateT) --
  // fp32 registers downconvert to StateT on store (no-op when StateT=float).
  #pragma unroll
  for (int r = 0; r < 8; r++) {
    simd<StateT, 128> ov = fp32InS_persistent.select<128, 1>(r * 128);
    block_store<StateT, 128>(sMyRows + r * headDim, ov);
  }
}
