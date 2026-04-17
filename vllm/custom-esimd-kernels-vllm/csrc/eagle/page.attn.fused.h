// page.attn.fused.h — Fused single-kernel paged attention decode
// GQA ratio = 4, head_dim = 256, fp16.
//
// Grid:  (kvHead, batches)      ← 2D, not 3D
// WG:    4 threads
//
// Register peak: 91 / 128 GRF (K phase), 67 / 128 GRF (V phase)

#define FUSED_FP32_MIN (-1e+38f)

ESIMD_INLINE void sdpaDecodeGqa4Fused(
  uint8_t* qState,
  uint8_t* kvCache,
  uint8_t* out,
  uint32_t* pageTable,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
  uint32_t kvCacheBatchStride0,
  uint32_t kvCacheBatchStride1,
  uint32_t pageTableBatchStride,
  uint32_t pageTableSizeLog2,
  uint32_t headKv,
  sycl::nd_item<2>& ndi) {

  constexpr uint32_t baseOffsetInc16[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  constexpr float scaleCoeff = 0.0625f;
  constexpr uint32_t gqaRatio = 4;
  constexpr uint32_t headDim = 256;

  // SLM: 4 threads × 4 heads × 16 scores = 1024 bytes
  constexpr uint32_t slmSize = 4 * gqaRatio * 16 * sizeof(float);
  __ESIMD_NS::slm_init(slmSize);

  int hh         = ndi.get_local_id(0);  // [0..3]
  int kvHeadIdx  = ndi.get_group(0);
  int batchIdx   = ndi.get_group(1);

  uint32_t headQ         = headKv * gqaRatio;
  uint32_t pageTableSize = 1u << pageTableSizeLog2;
  uint32_t pageMask      = pageTableSize - 1;
  uint32_t kvSeqLen      = batchKvSeqLen[batchIdx];
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;

  if (kvSeqLen == 0) return;

  simd<uint32_t, 16> baseInc16(baseOffsetInc16);
  uint32_t qHeadIdx = kvHeadIdx * gqaRatio;

  // ══════ Load Q (identical to Phase 1) ══════
  // Each thread: 4 heads × 64 dims (two 32-dim chunks: hh*32 and 128+hh*32)
  simd<fp16, 64 * 4> qqFp16;
  simd<float, 64 * 4> qqFp32;  // 16 GRF — constant

  uint32_t offsetQ = (qHeadIdx * headDim + hh * 32) * sizeof(fp16)
                   + batchIdx * headQ * headDim * sizeof(fp16);

#pragma unroll
  for (int qn = 0; qn < 4; qn++) {
#pragma unroll
    for (int qk = 0; qk < 2; qk++) {
      qqFp16.template bit_cast_view<uint8_t>().select<64, 1>(128 * qn + 64 * qk) =
        __ESIMD_ENS::lsc_block_load<uint8_t, 64,
          __ESIMD_ENS::lsc_data_size::default_size,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(
            (uint8_t*)qState + offsetQ
            + qn * headDim * sizeof(fp16)
            + qk * 4 * 32 * sizeof(fp16));
    }
  }
  qqFp32 = qqFp16;

  // ══════ Output accumulator — constant across loop ══════
  simd<float, 4 * 64> outputAccum = 0.0f;  // 16 GRF
  simd<float, 4> runningMax;                // 1 GRF
  runningMax = FUSED_FP32_MIN;
  simd<float, 4> softmaxSum = 0.0f;        // 1 GRF
  // Total constant: 34 GRF


  // ══════ K offset base (same as Phase 1) ══════
  uint32_t offsetBaseK = hh * 32 * sizeof(fp16) + headDim * kvHeadIdx * sizeof(fp16);

  // ══════ V cache ══════
  uint32_t vWidthBytes = headKv * headDim * sizeof(fp16) - 1;
  fp16* vPtrBase = (fp16*)kvCache + kvCacheBatchStride0;
  uint32_t totalPages = (kvSeqLen + pageTableSize - 1) >> pageTableSizeLog2;
  uint32_t lastPageH  = (kvSeqLen - 1) & pageMask;

  // ══════ Main loop: 16 tokens per chunk ══════
  uint32_t numChunks = (kvSeqLen + 15) / 16;

  for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    uint32_t kvSeqOffset = chunkIdx * 16;

    // Page table lookup
    uint32_t ptIdx    = kvSeqOffset >> pageTableSizeLog2;
    uint32_t ptOffset = kvSeqOffset & pageMask;
    uint32_t pageIdx  = pageTable[pageTableBase + ptIdx];

    // Validity mask
    simd<uint16_t, 16> validMask   = (baseInc16 + kvSeqOffset) < kvSeqLen;
    simd<uint16_t, 16> invalidMask = (baseInc16 + kvSeqOffset) >= kvSeqLen;

    // ─── QK^T: copied from Phase 1, kkCacheHalf per kk ───
    simd<float, 4 * 16> partialScores = 0.0f;  // 4 GRF

    simd<uint32_t, 16> simdOffsetsK = baseInc16;
    simdOffsetsK =
      simdOffsetsK * headKv * headDim * sizeof(fp16)
      + offsetBaseK
      + pageIdx * kvCacheBatchStride1 * sizeof(fp16)
      + ptOffset * headKv * headDim * sizeof(fp16);

#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      // 16 GRF: 512 fp16 = 16 tokens × 32 dims (one kk half)
      simd<fp16, 16 * 32> kkCacheHalf;
      // 32 GRF: 512 float (deinterleaved)
      simd<float, 16 * 32> kkFp32;

      // 4 gathers: 16 lanes × 4 uint32 = 16 tokens × 8 fp16 each
#pragma unroll
      for (int kkk = 0; kkk < 4; kkk++) {
        kkCacheHalf.template bit_cast_view<uint32_t>().select<64, 1>(64 * kkk) =
          __ESIMD_ENS::lsc_gather<uint32_t, 4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16, uint32_t>(
              (uint32_t*)kvCache, simdOffsetsK, validMask);
        simdOffsetsK = simdOffsetsK + 8 * sizeof(fp16);
      }

      // Skip to next 32-dim block (other 3 threads' slices)
      simdOffsetsK = simdOffsetsK + 3 * 32 * sizeof(fp16);

      // Deinterleave fp16 pairs within uint32 → fp32
      // Same pattern as original but with offset 0 instead of 512*kk
#pragma unroll
      for (int kkk = 0; kkk < 16; kkk++) {
        kkFp32.select<16, 1>(32 * kkk + 16 * 0) = kkCacheHalf.select<16, 2>(32 * kkk + 0);
        kkFp32.select<16, 1>(32 * kkk + 16 * 1) = kkCacheHalf.select<16, 2>(32 * kkk + 1);
      }

      // MAD: partialScores[head, token] += K[dim, token] * Q[head, dim]
#pragma unroll
      for (int kkk = 0; kkk < 32; kkk++) {
#pragma unroll
        for (int pn = 0; pn < 4; pn++) {
          partialScores.select<16, 1>(16 * pn) =
            partialScores.select<16, 1>(16 * pn)
            + kkFp32.select<16, 1>(16 * kkk) * qqFp32[64 * pn + 32 * kk + kkk];
        }
      }
    } // kk
    // After this scope: kkCacheHalf(16) + kkFp32(32) = 48 GRF freed

    // Mask invalid tokens
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      partialScores.select<16, 1>(16 * pn).merge(FUSED_FP32_MIN, invalidMask);
    }

    // ─── SLM reduce: 4 threads → full 4×16 scores ───
    // Each thread writes its partial scores for all 4 heads
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      slm_block_store<float, 16>(
        hh * gqaRatio * 16 * sizeof(float) + pn * 16 * sizeof(float),
        partialScores.select<16, 1>(16 * pn));
    }
    barrier();

    // Each thread reads and sums across 4 threads
    simd<float, 4 * 16> fullScores;  // 4 GRF (reuses partialScores reg space)
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      simd<float, 16> s =
        slm_block_load<float, 16>(0 * gqaRatio * 16 * sizeof(float) + pn * 16 * sizeof(float));
      s = s + slm_block_load<float, 16>(1 * gqaRatio * 16 * sizeof(float) + pn * 16 * sizeof(float));
      s = s + slm_block_load<float, 16>(2 * gqaRatio * 16 * sizeof(float) + pn * 16 * sizeof(float));
      s = s + slm_block_load<float, 16>(3 * gqaRatio * 16 * sizeof(float) + pn * 16 * sizeof(float));
      fullScores.select<16, 1>(16 * pn) = s;
    }
    barrier();  // protect SLM for next iteration

    // ─── Online softmax ───
    fullScores = fullScores * scaleCoeff;

    simd<float, 4> chunkMax;
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      simd<float, 16> s = fullScores.select<16, 1>(16 * pn);
      simd<float, 8> m8 = __ESIMD_NS::max<float, 8, float>(s.select<8,1>(0), s.select<8,1>(8));
      simd<float, 4> m4 = __ESIMD_NS::max<float, 4, float>(m8.select<4,1>(0), m8.select<4,1>(4));
      simd<float, 2> m2 = __ESIMD_NS::max<float, 2, float>(m4.select<2,1>(0), m4.select<2,1>(2));
      chunkMax[pn] = __ESIMD_NS::max<float>(m2[0], m2[1]);
    }

    simd<float, 4> newMax = __ESIMD_NS::max<float, 4, float>(chunkMax, runningMax);
    simd<float, 4> compOld = exp(runningMax - newMax);
    simd<float, 4> compNew = exp(chunkMax - newMax);

    // Rescale accumulators
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      outputAccum.select<64, 1>(64 * pn) = outputAccum.select<64, 1>(64 * pn) * compOld[pn];
      softmaxSum[pn] = softmaxSum[pn] * compOld[pn];
    }

    // exp(score - newMax) and accumulate softmax sum
    simd<float, 4 * 16> expScores;  // 4 GRF
#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      expScores.select<16, 1>(16 * pn) = exp(fullScores.select<16, 1>(16 * pn) - newMax[pn]);
      softmaxSum[pn] = softmaxSum[pn]
        + __ESIMD_DNS::sum<float, float, 16>(expScores.select<16, 1>(16 * pn));
    }

    runningMax = newMax;

    // ─── V accumulation ───
    // This thread owns 64 output dims (two 32-dim chunks).
    // Process in 4 sub-passes of 16 dims: lsc_load_2d(16 tokens × 16 dims)
    //
    // V sub-pass dim offsets within headDim=256:
    //   pass 0: hh*32 + 0
    //   pass 1: hh*32 + 16
    //   pass 2: 128 + hh*32 + 0
    //   pass 3: 128 + hh*32 + 16

    uint32_t pageOffset = pageIdx * kvCacheBatchStride1;
    fp16* currPtrV = vPtrBase + pageOffset;
    uint32_t heightV = (ptIdx + 1 < totalPages) ? (pageTableSize - 1) : lastPageH;

    uint32_t vDimBase[4];
    vDimBase[0] = kvHeadIdx * headDim + hh * 32;
    vDimBase[1] = kvHeadIdx * headDim + hh * 32 + 16;
    vDimBase[2] = kvHeadIdx * headDim + 128 + hh * 32;
    vDimBase[3] = kvHeadIdx * headDim + 128 + hh * 32 + 16;

#pragma unroll
    for (int vpass = 0; vpass < 4; vpass++) {
      simd<fp16, 16 * 16> vvCache;  // 8 GRF
      vvCache =
        __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached>(
            currPtrV, vWidthBytes, heightV, vWidthBytes,
            vDimBase[vpass],  // x = dim offset
            ptOffset);        // y = token offset within page

      simd<float, 16 * 16> vvFp32;  // 16 GRF
      vvFp32 = vvCache;

      // output[head, vpass*16 + 0..15] += sum_t( expScore[head, t] * V[t, 0..15] )
      uint32_t dimOff = vpass * 16;
#pragma unroll
      for (int pn = 0; pn < 4; pn++) {
#pragma unroll
        for (int t = 0; t < 16; t++) {
          outputAccum.select<16, 1>(64 * pn + dimOff) =
            outputAccum.select<16, 1>(64 * pn + dimOff)
            + vvFp32.select<16, 1>(16 * t) * expScores[16 * pn + t];
        }
      }
    }
  } // end main loop

  // ══════ Finalize: normalize and store ══════
  uint32_t outBase = (batchIdx * headQ + qHeadIdx) * headDim;

#pragma unroll
  for (int pn = 0; pn < 4; pn++) {
    float ss = softmaxSum[pn];
    float invSum = (ss > 0.0f) ? (1.0f / ss) : 0.0f;
    outputAccum.select<64, 1>(64 * pn) = outputAccum.select<64, 1>(64 * pn) * invSum;
  }

  // Store: 4 heads × 2 chunks of 32 fp16
#pragma unroll
  for (int pn = 0; pn < 4; pn++) {
    simd<fp16, 64> outFp16 = outputAccum.select<64, 1>(64 * pn);
    // First 32 dims: offset hh*32
    block_store<fp16, 32>(
      (fp16*)out + outBase + pn * headDim + hh * 32,
      outFp16.select<32, 1>(0));
    // Second 32 dims: offset 128+hh*32
    block_store<fp16, 32>(
      (fp16*)out + outBase + pn * headDim + 128 + hh * 32,
      outFp16.select<32, 1>(32));
  }
}
