// page.attn.splitk.h — Split-K paged attention decode for GQA4 / headDim=256
//
// Reuses Phase 1 (sdpaDecodeGqa4Phase1) from page.attn.h unchanged.
// Phase 2 is split along the KV-sequence dimension.
// numPartitions is FIXED at launch time (e.g. 8). Each WG computes its
// own token range from the per-batch seq_lens[batchIdx] on device, so
// the grid size and buffer size are independent of max_model_len.

// ────────────────────────────────────────────────────────────────────────
// Phase 2 with split-K.
//
// Grid:  (headDim/16 * 2,  kvHead,  batches * numPartitions)
// Local: (2, 1, 1)
//
// Each WG computes:
//   kvSeqLen = batchKvSeqLen[batchIdx]
//   totalChunks = ceil(kvSeqLen / 64)
//   myStartChunk = partitionIdx * ceil(totalChunks / numPartitions)
//   myEndChunk   = min((partitionIdx+1) * ceil(totalChunks / numPartitions), totalChunks)
//
// This means the partition boundaries adapt to actual sequence length,
// not max_model_len.
// ────────────────────────────────────────────────────────────────────────
ESIMD_INLINE void sdpaDecodeGqa4Phase2SplitK(
  uint8_t* pState,
  float* pMax,
  uint8_t* vState,
  uint8_t* tmpOut,
  float* expSums,
  float* maxLogitsOut,
  uint32_t* pageTable,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
  uint32_t kvCacheBatchStride0,
  uint32_t kvCacheBatchStride1,
  uint32_t pageTableBatchStride,
  uint32_t pageTableSizeLog2,
  uint32_t pStride,
  uint32_t maxStride,
  uint32_t headKv,
  uint32_t numPartitions,
  sycl::nd_item<3>& ndi) {

  constexpr uint32_t gqaRatio = 4;
  constexpr uint32_t headDim = 256;
  constexpr uint32_t outputPerGroup = 64;
  constexpr float matMulQuantCoeff = 0.0625f;
  constexpr uint32_t slmSizeOut = 2 * gqaRatio * 16 * sizeof(float);
  constexpr uint32_t slmSizeSoftmaxSum = 2 * gqaRatio * sizeof(float);
  constexpr uint32_t slmSize = slmSizeOut + slmSizeSoftmaxSum;
  __ESIMD_NS::slm_init(slmSize);

  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0); // [0, headDim / 16)
  int globalLinearId1 = ndi.get_group(1); // [0, kvHead)
  int globalLinearId2 = ndi.get_group(2); // [0, batches * numPartitions)
  int hh = localLinearId & 0x1;

  int batchIdx = globalLinearId2 / numPartitions;
  int partitionIdx = globalLinearId2 % numPartitions;

  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  uint32_t headQ = headKv * gqaRatio;

  // ── Compute this partition's chunk range from actual kvSeqLen ──
  uint32_t totalChunks = (kvSeqLen + outputPerGroup - 1) / outputPerGroup;
  uint32_t chunksPerPart = (totalChunks + numPartitions - 1) / numPartitions;
  uint32_t partStartChunk = partitionIdx * chunksPerPart;
  uint32_t partEndChunk = partStartChunk + chunksPerPart;
  if (partEndChunk > totalChunks) partEndChunk = totalChunks;

  // Early exit: no work for this partition
  if (partStartChunk >= totalChunks) {
    if (localLinearId == 0 && globalLinearId0 == 0) {
      uint32_t metaBase = (batchIdx * numPartitions + partitionIdx) * headQ;
      for (uint32_t h = 0; h < headQ; h++) {
        expSums[metaBase + h]   = 0.0f;
        maxLogitsOut[metaBase + h] = FP32_MIN * matMulQuantCoeff;
      }
    }
    return;
  }

  int kvSeqOutLoopCount = (int)(partEndChunk - partStartChunk);

  uint32_t pageTableSize     = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = pageTableSize - 1;
  uint32_t pageTableBase     = pageTableBatchStride * batchIdx;

  simd<uint32_t, 32> simdOffsetsVs;
  simd<float, 4 * 32> ppFp32;
  simd<float, 4> historicMax;
  simd<fp16, 32 * 16> vvCache;
  simd<float, 16 * 16> vvFp32;
  simd<float, 4 * 16> softmaxSum;
  simd<float, 4 * 16> outputFp32;

  // tmpOut layout:  [batches, numPartitions, headQ, headDim]  fp16
  uint32_t tmpOutOffset =
      ((batchIdx * numPartitions + partitionIdx) * headQ
        + globalLinearId1 * gqaRatio) * headDim
      + globalLinearId0 * 16
      + hh * 2 * headDim;

  // pState / pMax pointers — offset to this partition's first chunk
  uint32_t offsetP = globalLinearId1 * gqaRatio * pStride
                   + hh * 32
                   + batchIdx * headQ * pStride
                   + partStartChunk * outputPerGroup;

  uint32_t offsetMax = globalLinearId1 * gqaRatio * maxStride
                     + batchIdx * headQ * maxStride
                     + partStartChunk;

  // V cache 2D-load parameters
  uint32_t widthV  = headKv * headDim * sizeof(fp16) - 1;
  uint32_t heightV = pageTableSize - 1;
  uint32_t vX      = globalLinearId0 * 16 + headDim * globalLinearId1;
  uint32_t vY;
  uint32_t totalPages    = (kvSeqLen + pageTableSize - 1) >> pageTableSizeLog2;
  uint32_t lastPageHeight = (kvSeqLen > 0) ? ((kvSeqLen - 1) & (pageTableSize - 1)) : 0;
  fp16* vPtrBase = (fp16*)vState + kvCacheBatchStride0;

  historicMax = FP32_MIN * matMulQuantCoeff;
  softmaxSum  = 0;
  outputFp32  = 0;

  // ── Main loop ──
  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    simd<float, 4> loopMax;
    simd<float, 4> currMax;
    simd<float, 4> compensationP;
    simd<float, 4> compensationO;
    uint32_t kvSeqOffset    = (partStartChunk + loopIdx) * outputPerGroup;
    uint32_t pageTableIdx   = kvSeqOffset >> pageTableSizeLog2;
    uint32_t pageTableOffset = kvSeqOffset & pageTableLoopMask;
    uint32_t pageIdx  = pageTable[pageTableBase + pageTableIdx];
    uint32_t pageOff  = pageIdx * kvCacheBatchStride1;
    fp16* currPtrV = vPtrBase + pageOff;

    heightV = (pageTableIdx + 1 < totalPages) ? (pageTableSize - 1) : lastPageHeight;
    vY = pageTableOffset + 32 * hh;

#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      ppFp32.select<32, 1>(32 * pn) =
        block_load<float, 32>((float*)pState + offsetP + pStride * pn);
      currMax[pn] = pMax[offsetMax + maxStride * pn];
    }

    vvCache.select<256, 1>(0) =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>(currPtrV, widthV, heightV, widthV, vX, vY);
    vY += 16;
    vvCache.select<256, 1>(256) =
      __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>(currPtrV, widthV, heightV, widthV, vX, vY);

    loopMax = __ESIMD_NS::max<float, 4, float>(currMax, historicMax);
    compensationO = historicMax - loopMax;
    compensationP = currMax - loopMax;
    compensationO = exp(compensationO);
    compensationP = exp(compensationP);

#pragma unroll
    for (int oc = 0; oc < 4; oc++)
      ppFp32.select<32, 1>(32 * oc) = ppFp32.select<32, 1>(32 * oc) * compensationP[oc];

#pragma unroll
    for (int oc = 0; oc < 4; oc++) {
      outputFp32.select<16, 1>(16 * oc) = outputFp32.select<16, 1>(16 * oc) * compensationO[oc];
      softmaxSum.select<16, 1>(16 * oc) = softmaxSum.select<16, 1>(16 * oc) * compensationO[oc];
    }

#pragma unroll
    for (int oc = 0; oc < 4; oc++) {
#pragma unroll
      for (int pk = 0; pk < 2; pk++)
        softmaxSum.select<16, 1>(16 * oc) += ppFp32.select<16, 1>(32 * oc + 16 * pk);
    }

#pragma unroll
    for (int vk = 0; vk < 2; vk++) {
      vvFp32 = vvCache.select<256, 1>(256 * vk);
#pragma unroll
      for (int oc = 0; oc < 4; oc++) {
#pragma unroll
        for (int pk = 0; pk < 16; pk++)
          outputFp32.select<16, 1>(16 * oc) += vvFp32.select<16, 1>(16 * pk) * ppFp32[32 * oc + 16 * vk + pk];
      }
    }

    historicMax = loopMax;
    offsetP   += outputPerGroup;
    offsetMax += 1;
  }

  // ── SLM reduction across 2 threads ──

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++)
    slm_block_store<float, 16>(hh * 16 * sizeof(float) + 32 * kk * sizeof(float),
                               outputFp32.select<16, 1>(16 * kk));

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    float s = __ESIMD_DNS::sum<float, float, 16>(softmaxSum.select<16, 1>(16 * kk));
    slm_block_store<float, 1>(slmSizeOut + hh * sizeof(float) + 2 * kk * sizeof(float), s);
  }

  barrier();

  {
    simd<float, 4> dividor;
    simd<float, 64> outputTempFp32;

    dividor = slm_block_load<float, 4>(slmSizeOut + hh * 4 * sizeof(float));
    dividor.select<2, 2>(0) = dividor.select<2, 2>(0) + dividor.select<2, 2>(1);

    float expSum0 = dividor[0];
    float expSum1 = dividor[2];

    dividor = 1.0f / dividor;

    outputTempFp32 = slm_block_load<float, 64>(hh * 4 * 16 * sizeof(float));

#pragma unroll
    for (int oc = 0; oc < 2; oc++) {
      outputFp32.select<16, 1>(16 * oc) =
        outputTempFp32.select<16, 1>(32 * oc) + outputTempFp32.select<16, 1>(32 * oc + 16);
      outputFp32.select<16, 1>(16 * oc) = outputFp32.select<16, 1>(16 * oc) * dividor[2 * oc];
    }

    simd<fp16, 32> outputTemp = outputFp32.select<32, 1>(0);
#pragma unroll
    for (int oc = 0; oc < 2; oc++)
      block_store<fp16, 16>((fp16*)tmpOut + tmpOutOffset + oc * headDim,
                            outputTemp.select<16, 1>(16 * oc));

    if (globalLinearId0 == 0) {
      uint32_t metaBase = (batchIdx * numPartitions + partitionIdx) * headQ
                        + globalLinearId1 * gqaRatio
                        + hh * 2;
      expSums[metaBase + 0]   = expSum0;
      expSums[metaBase + 1]   = expSum1;
      maxLogitsOut[metaBase + 0] = historicMax[hh * 2 + 0];
      maxLogitsOut[metaBase + 1] = historicMax[hh * 2 + 1];
    }
  }
}

// ────────────────────────────────────────────────────────────────────────
// Reduce kernel.
//
// Grid:  (headDim / 16,  headQ,  batches)
// Local: (1, 1, 1)
// ────────────────────────────────────────────────────────────────────────
ESIMD_INLINE void sdpaDecodeGqa4Reduce(
  uint8_t* tmpOut,
  float*   expSums,
  float*   maxLogitsArr,
  uint8_t* out,
  uint32_t* batchKvSeqLen,
  uint32_t numPartitions,
  uint32_t headQ,
  sycl::nd_item<3>& ndi) {

  constexpr uint32_t headDim = 256;
  constexpr float matMulQuantCoeff = 0.0625f;

  int dimChunk  = ndi.get_group(0);
  int headIdx   = ndi.get_group(1);
  int batchIdx  = ndi.get_group(2);

  // Find how many partitions actually did work for this batch.
  // A partition with expSum==0 and maxLogit==sentinel had no tokens.
  float sentinel = FP32_MIN * matMulQuantCoeff;

  // Fast path: single partition — just copy.
  if (numPartitions == 1) {
    uint32_t srcOff = (batchIdx * numPartitions * headQ + headIdx) * headDim + dimChunk * 16;
    uint32_t dstOff = (batchIdx * headQ + headIdx) * headDim + dimChunk * 16;
    simd<fp16, 16> v = block_load<fp16, 16>((fp16*)tmpOut + srcOff);
    block_store<fp16, 16>((fp16*)out + dstOff, v);
    return;
  }

  // 1. Find global max across active partitions
  float globalMax = sentinel;
  for (uint32_t p = 0; p < numPartitions; p++) {
    uint32_t idx = (batchIdx * numPartitions + p) * headQ + headIdx;
    float m = maxLogitsArr[idx];
    if (m > globalMax) globalMax = m;
  }

  // 2. Accumulate rescaled partial outputs.
  simd<float, 16> accum = 0.0f;
  float globalExpSum = 0.0f;

  for (uint32_t p = 0; p < numPartitions; p++) {
    uint32_t metaIdx = (batchIdx * numPartitions + p) * headQ + headIdx;
    float localMax    = maxLogitsArr[metaIdx];
    float localExpSum = expSums[metaIdx];

    // Skip empty partitions
    if (localExpSum == 0.0f) continue;

    float rescale     = exp(localMax - globalMax);
    float rescaledExp = localExpSum * rescale;
    globalExpSum += rescaledExp;

    uint32_t srcOff = ((batchIdx * numPartitions + p) * headQ + headIdx) * headDim
                    + dimChunk * 16;
    simd<fp16, 16> partialFp16 = block_load<fp16, 16>((fp16*)tmpOut + srcOff);
    simd<float, 16> partialFp32 = partialFp16;
    accum = accum + partialFp32 * rescaledExp;
  }

  // 3. Normalise and store.
  float invSum = (globalExpSum > 0.0f) ? (1.0f / globalExpSum) : 0.0f;
  accum = accum * invSum;

  uint32_t dstOff = (batchIdx * headQ + headIdx) * headDim + dimChunk * 16;
  simd<fp16, 16> result = accum;
  block_store<fp16, 16>((fp16*)out + dstOff, result);
}
