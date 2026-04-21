#define FP32_MIN (-1e+38)
// kv cache shape [2 (k/v), reserved size, page_size, head_num, head_dim]
// kv cache strid: [2 * page_size * head_num * head_dim, page_size * head_num * head_dim, head_dim, head_dim, 1]
ESIMD_INLINE void sdpaDecodeGqa4Phase1(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* pState,
  float* pGroupMax,
  float* pGlobalMax,
  uint32_t* pPollP,
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
  uint32_t gqaRatio,
  sycl::nd_item<3>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr float matMulQuantCoeff = 0.0625f;
  constexpr uint32_t headDim = 256;
  constexpr uint32_t maxPGroupOut = 128;
  constexpr uint32_t outputPerGroup = 64;
  constexpr uint32_t loopCount = outputPerGroup / 16;
  constexpr uint32_t slmSizePhase1 = 4 * maxPGroupOut * 4 * sizeof(float);
  __ESIMD_NS::slm_init<slmSizePhase1>();
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0); // [0, kvHead * gqaRatio/4)
  int globalLinearId1 = ndi.get_group(1); // [0, keSeqLen / 64)
  int globalLinearId2 = ndi.get_group(2); // [0, bs)
  int hh = localLinearId & 0x3;
  int vv = localLinearId >> 2;
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 4;
  int kvHeadIdx = globalLinearId0 / gqaGroups;
  int qGroupIdx = globalLinearId0 % gqaGroups;
  int qHeadIdx = kvHeadIdx * gqaRatio + qGroupIdx * 4;
  uint32_t pageTableSize = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = pageTableSize - 1;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;
  simd<uint32_t, 16> simdOffsetsK;

  simd<uint32_t, 16> baseOffsetInc16AsSimd(baseOffsetInc16);
  simd<float, 64 * 4> qqFp32;
  simd<fp16, 64 * 4> qqFp16;
  simd<float, 16 * 4> ppFp32;
  simd<fp16, 16 * 64> kkCache;
  simd<float, 1> ppMax;

  uint32_t headQ = headKv * gqaRatio;
  uint64_t outputOffset = qHeadIdx * pStride + globalLinearId1 * outputPerGroup + hh * pStride + batchIdx * headQ * pStride;
  uint32_t outputMaxOffset = qHeadIdx * maxStride + globalLinearId1 + hh * maxStride + batchIdx * headQ * maxStride;
  uint32_t offsetQ = qHeadIdx * headDim * sizeof(fp16) + hh * 32 * sizeof(fp16) + batchIdx * headQ * headDim * sizeof(fp16);
  uint32_t baseCoordK = globalLinearId1 * outputPerGroup;
  uint32_t offsetBaseK = hh * 32 * sizeof(fp16) + headDim * kvHeadIdx * sizeof(fp16);
  uint32_t kvSeqOffset = baseCoordK;

  if (baseCoordK >= kvSeqLen) {
    return;
  }

#pragma unroll
  for (int qn = 0; qn < 4; qn++) {
#pragma unroll
    for (int qk = 0; qk < 2; qk++) {
      qqFp16.template bit_cast_view<uint8_t>().select<64, 1>(128 * qn + 64 * qk) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQ + qn * headDim * sizeof(fp16) + qk * 4 * 32 * sizeof(fp16));
    }
  }

  qqFp32 = qqFp16;

#pragma unroll
  for (int loopIdx = 0; loopIdx < loopCount; loopIdx++) {
    ppFp32 = 0.0f;
    if (kvSeqOffset < kvSeqLen) {
      uint32_t pageTableIdx = kvSeqOffset >> pageTableSizeLog2;
      uint32_t pageTableOffset = kvSeqOffset & pageTableLoopMask;
      uint32_t pageIdx = pageTable[pageTableBase + pageTableIdx];
      simd<uint16_t, 16> mask;
      simd<uint16_t, 16> maskNeg;
      simd<float, 16 * 32> kkFp32;
      simd<uint32_t, 16> logicSimdOffsetK;
      logicSimdOffsetK = baseOffsetInc16AsSimd + kvSeqOffset;
      mask = logicSimdOffsetK < kvSeqLen;
      maskNeg = logicSimdOffsetK >= kvSeqLen;
      simdOffsetsK = baseOffsetInc16AsSimd;

      simdOffsetsK = 
        simdOffsetsK * headKv * headDim * sizeof(fp16) +
        offsetBaseK +
        pageIdx * kvCacheBatchStride1 * sizeof(fp16) +
        pageTableOffset * headKv * headDim * sizeof(fp16)
        ;

#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
#pragma unroll
        for (int kkk = 0; kkk < 4; kkk++) {
          kkCache.template bit_cast_view<uint32_t>().select<64, 1>(256 * kk + 64 * kkk) =
            __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t
            >((uint32_t*)kState, simdOffsetsK, mask);
          simdOffsetsK += 8 * sizeof(fp16);
        }

        simdOffsetsK += 3 * 32 * sizeof(fp16);

#pragma unroll
        for (int kkk = 0; kkk < 16; kkk++) {
          kkFp32.select<16, 1>(32 * kkk + 16 * 0) = kkCache.select<16, 2>(512 * kk + 32 * kkk + 0);
          kkFp32.select<16, 1>(32 * kkk + 16 * 1) = kkCache.select<16, 2>(512 * kk + 32 * kkk + 1);
        }

#pragma unroll
        for (int kkk = 0; kkk < 32; kkk++) {
#pragma unroll
          for (int pn = 0; pn < 4; pn++) {
            ppFp32.select<16, 1>(16 * pn) = ppFp32.select<16, 1>(16 * pn) + kkFp32.select<16, 1>(16 * kkk) * qqFp32[64 * pn + 32 * kk + kkk];
          }
        }
      }

#pragma unroll
      for (int pn = 0; pn < 4; pn++) {
        ppFp32.select<16, 1>(pn * 16).merge(FP32_MIN, maskNeg);
      }
    }
    else {
      ppFp32 = FP32_MIN;
    }

#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      slm_block_store<float, 16>(
        loopIdx * 4 * 64 * sizeof(float) +
        pn * 64 * sizeof(float) +
        hh * 16 * sizeof(float),
        ppFp32.select<16, 1>(16 * pn));
    }
    kvSeqOffset += 16;
  }

  barrier();

  {
    simd<float, 16 * 4> ppOut;
    simd<float, 16> maxTemp;
    simd<float, 64> ppTemp;
#pragma unroll
    for (int pk = 0; pk < 4; pk++) {
      ppTemp = slm_block_load<float, 64>(pk * 64 * 4 * sizeof(float) + hh * 64 * sizeof(float));
#pragma unroll
      for (int pn = 0; pn < 1; pn++) {
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppTemp.select<16, 1>(64 * pn) + ppTemp.select<16, 1>(64 * pn + 16);
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) + ppTemp.select<16, 1>(64 * pn + 32);
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) + ppTemp.select<16, 1>(64 * pn + 48);
      }
    }

    ppOut = ppOut * matMulQuantCoeff;
    maxTemp = ppOut.select<16, 1>(0);
#pragma unroll
    for (int pk = 1; pk < 4; pk++) {
      maxTemp = __ESIMD_NS::max<float, 16, float>(maxTemp, ppOut.select<16, 1>(16 * pk));
    }

    maxTemp.select<8, 1>(0) = __ESIMD_NS::max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
    maxTemp.select<4, 1>(0) = __ESIMD_NS::max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));
    maxTemp.select<2, 1>(0) = __ESIMD_NS::max<float, 2, float>(maxTemp.select<2, 1>(0), maxTemp.select<2, 1>(2));
    ppMax[0] = __ESIMD_NS::max<float>(maxTemp[0], maxTemp[1]);
    ppOut.select<64, 1>(0) = ppOut.select<64, 1>(0) - ppMax[0];
    ppOut = __ESIMD_NS::pow<float, 64, float>(2.718281828459f, ppOut);

    {
      block_store<float, 64>((float*)pState + outputOffset, ppOut.select<64, 1>(0));
      simd<float, 1> zeros = 0.0f;
      pGroupMax[outputMaxOffset] = ppMax[0];
      uint32_t atomicOffset = (batchIdx * gqaRatio * headKv + qHeadIdx + hh) * sizeof(float);
      uint32_t arrivalId =
        atomic_update<
        __ESIMD_NS::atomic_op::inc,
        uint32_t,
        1,
        uint32_t>(pPollP, atomicOffset);

      if (0 == arrivalId) {
        atomic_update<
          __ESIMD_NS::atomic_op::fcmpxchg,
          float,
          1,
          uint32_t>(pGlobalMax, atomicOffset, ppMax, zeros);
      }
      else {
        atomic_update<
          __ESIMD_NS::atomic_op::fmax,
          float,
          1,
          uint32_t>(pGlobalMax, atomicOffset, ppMax);
      }

      outputOffset += pStride;
      outputMaxOffset += maxStride;
    }
  }
}

ESIMD_INLINE void sdpaDecodeGqa4Phase2(
  uint8_t* pState,
  float* pGroupMax,
  uint8_t* vState,
  float* pGlobalMax,
  float* pTempOut,
  float* pSoftmaxSum,
  uint8_t* out,
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
  uint32_t longestBatch,
  uint32_t flag,
  uint32_t gqaRatio,
  sycl::nd_item<3>& ndi) {
  constexpr uint32_t headDim = 256;
  constexpr uint32_t maxPGroupOut = 128;
  constexpr uint32_t outputPerGroup = 64;
  constexpr float matMulQuantCoeff = 0.0625f;
  constexpr uint32_t slmSizeOut = 2 * 4 * 16 * sizeof(float);
  constexpr uint32_t slmSizeSoftmaxSum = 2 * 4 * sizeof(float);
  constexpr uint32_t slmSize = slmSizeOut + slmSizeSoftmaxSum;
  __ESIMD_NS::slm_init<slmSize>();
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0); // [0, head dim / 16)
  int globalLinearId1 = ndi.get_group(1); // [0, kvHead * gqaRatio/4)
  int globalLinearId2 = ndi.get_group(2); // [0, bs)
  int hh = localLinearId & 0x1;
  int vv = localLinearId >> 1;
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 4;
  int kvHeadIdx = globalLinearId1 / gqaGroups;
  int qGroupIdx = globalLinearId1 % gqaGroups;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  int kvSeqOutGroup = (kvSeqLen + 1023) >> 10;
  uint32_t pageTableSize = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = (1 << pageTableSizeLog2) - 1;
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;
  uint32_t channelOffset = globalLinearId0 & 0xf;
  uint32_t groupIdx = globalLinearId0 >> 4;

  simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
  simd<uint32_t, 16> pageTableIndice;
  simd<uint32_t, 16> pageTableOffsets;

  simd<uint32_t, 16> pageAlignedOffsets;
  simd<uint32_t, 16> pageOffsetsInner;
  simd<uint32_t, 16> pageNumbers;

  simd<float, 4 * 32> ppFp32;
  simd<float, 4> historicMax;
  simd<fp16, 32 * 16> vvCache;
  simd<float, 16 * 16> vvFp32;
  simd<float, 4 * 16> softmaxSum;
  simd<float, 4 * 16> outputFp32;
  simd<float, 32> vvScale;

  if (groupIdx >= kvSeqOutGroup) {
    return;
  }
  uint32_t reduceCount = (longestBatch + 1023) >> 10;
  uint32_t headQ = headKv * gqaRatio;
  uint32_t qBaseIdx = kvHeadIdx * gqaRatio + qGroupIdx * 4;
  uint32_t outputOffset = qBaseIdx * headDim + channelOffset * 16 + hh * 2 * headDim + batchIdx * headQ * headDim;
  uint32_t outTempOffset = qBaseIdx * headDim + channelOffset * 16 + hh * 2 * headDim + groupIdx * headQ * headDim + batchIdx * reduceCount * headQ * headDim;
  uint32_t outOffsetSoftmaxSum = qBaseIdx + hh * 2 + groupIdx * headQ + batchIdx * reduceCount * headQ;
  uint32_t offsetP = qBaseIdx * pStride + hh * 32 + groupIdx * 1024 + batchIdx * headQ * pStride;
  uint32_t offsetMax = qBaseIdx * maxStride + batchIdx * headQ * maxStride + groupIdx * 16;
  uint32_t offsetGlobalMax = batchIdx * headQ + qBaseIdx;
  uint32_t widthV = headKv * headDim * sizeof(fp16) - 1;
  uint32_t heightV = pageTableSize - 1;
  uint32_t vX = channelOffset * 16 + headDim * kvHeadIdx;
  uint32_t vY = 32 * hh;
  uint32_t totalPages = (kvSeqLen + pageTableSize - 1) >> pageTableSizeLog2;
  uint32_t lastPageHeight = (kvSeqLen - 1) & (pageTableSize - 1);
  if (kvSeqLen == 0) {
    lastPageHeight = 0;
  }
  fp16* vPtrBase = (fp16*)vState + kvCacheBatchStride0;

  historicMax = FP32_MIN * matMulQuantCoeff;
  softmaxSum = 0;
  outputFp32 = 0;

  simd<float, 4> globalMax;
  simd<float, 4> currMax;
  simd<float, 4> compensationP;
  simd<uint16_t, 16> mask;
  simd<uint16_t, 16> maskNeg;

  uint32_t kvSeqOffset = groupIdx * 1024;
  simdOffsets = simdOffsets * 64 + kvSeqOffset;
  mask = simdOffsets < kvSeqLen;
  maskNeg = simdOffsets >= kvSeqLen;
  pageTableIndice = (simdOffsets >> pageTableSizeLog2);
  pageTableOffsets = pageTableIndice * sizeof(uint32_t) + pageTableBase * sizeof(uint32_t);
  pageOffsetsInner = simdOffsets & pageTableLoopMask;
  pageNumbers =
    gather<
    uint32_t,
    16,
    1,
    uint32_t>(pageTable, pageTableOffsets, mask);

  pageAlignedOffsets = pageNumbers * kvCacheBatchStride1;
  pageAlignedOffsets.merge(0, maskNeg);
  globalMax = block_load<float, 4>(pGlobalMax + offsetGlobalMax);

#pragma unroll
  for (uint32_t loop = 0; loop < 16; loop++) {
    if (kvSeqOffset + loop * 64 < kvSeqLen) {
      fp16* currPtrV = vPtrBase + pageAlignedOffsets[loop];
      if (pageTableIndice[loop] + 1 < totalPages) {
        heightV = pageTableSize - 1;
      }
      else {
        heightV = lastPageHeight;
      }
      vY = pageOffsetsInner[loop] + 32 * hh;

#pragma unroll
      for (int pn = 0; pn < 4; pn++) {
        ppFp32.select<32, 1>(32 * pn) = block_load<float, 32>((float*)pState + offsetP + pStride * pn);
        currMax[pn] = pGroupMax[offsetMax + maxStride * pn];
      }

      vvCache.select<256, 1>(0 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        fp16,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
        >(currPtrV, widthV, heightV, widthV, vX, vY);
      vY += 16;

      vvCache.select<256, 1>(1 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        fp16,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>
        (currPtrV, widthV, heightV, widthV, vX, vY);

      compensationP = currMax - globalMax;
      compensationP = exp(compensationP);

#pragma unroll
      for (int oc = 0; oc < 4; oc++) {
        ppFp32.select<32, 1>(32 * oc) = ppFp32.select<32, 1>(32 * oc) * compensationP[oc];
      }

#pragma unroll
      for (int oc = 0; oc < 4; oc++) {
#pragma unroll
        for (int pk = 0; pk < 2; pk++) {
          softmaxSum.select<16, 1>(16 * oc) = softmaxSum.select<16, 1>(16 * oc) + ppFp32.select<16, 1>(32 * oc + 16 * pk);
        }
      }

#pragma unroll
      for (int vk = 0; vk < 2; vk++) {
        vvFp32 = vvCache.select<256, 1>(256 * vk);
#pragma unroll
        for (int oc = 0; oc < 4; oc++) {
#pragma unroll
          for (int pk = 0; pk < 16; pk++) {
            outputFp32.select<16, 1>(16 * oc) += vvFp32.select<16, 1>(16 * pk) * ppFp32[32 * oc + 16 * vk + pk];
          }
        }
      }

      offsetP += 64;
      offsetMax += 1;
    }
  }

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    slm_block_store<float, 16>(hh * 16 * sizeof(float) + 32 * kk * sizeof(float), outputFp32.select<16, 1>(16 * kk));
  }

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    float slmTempSoftmaxSum;
    slmTempSoftmaxSum = __ESIMD_DNS::sum<float, float, 16>(softmaxSum.select<16, 1>(16 * kk));
    slm_block_store<float, 1>(slmSizeOut + hh * sizeof(float) + 2 * kk * sizeof(float), slmTempSoftmaxSum);
  }

  barrier();

  {
    simd<float, 4> dividor;
    simd<float, 2> softmaxMul;
    simd<float, 64> outputTempFp32;
    dividor = slm_block_load<float, 4>(slmSizeOut + hh * 4 * sizeof(float));
    softmaxMul = dividor.select<2, 2>(0) + dividor.select<2, 2>(1);
    outputTempFp32 = slm_block_load<float, 64>(hh * 4 * 16 * sizeof(float));

#pragma unroll
    for (int oc = 0; oc < 2; oc++) {
      outputFp32.select<16, 1>(16 * oc) = outputTempFp32.select<16, 1>(32 * oc) + outputTempFp32.select<16, 1>(32 * oc + 16);
    }

    if ((flag & 0x1) == 1) {
      softmaxMul = 1.0f / softmaxMul;
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        outputFp32.select<16, 1>(16 * oc) = outputFp32.select<16, 1>(16 * oc) * softmaxMul[oc];
      }
      simd<fp16, 32> outputTemp = outputFp32.select<32, 1>(0);
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        block_store<fp16, 16>((fp16*)out + outputOffset + oc * headDim, outputTemp.select<16, 1>(16 * oc));
      }
    }
    else {
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        block_store<float, 16>((float*)pTempOut + outTempOffset + oc * headDim, outputFp32.select<16, 1>(16 * oc));
      }

      block_store<float, 2>((float*)pSoftmaxSum + outOffsetSoftmaxSum, softmaxMul);
    }
  }
}

ESIMD_INLINE void sdpaDecodeGqa4Phase3(
  float* pTempOut,
  float* pSoftmaxSum,
  uint8_t* out,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
  uint32_t headKv,
  uint32_t longestBatch,
  uint32_t gqaRatio,
  sycl::nd_item<3>& ndi) {
  constexpr uint32_t headDim = 256;
  constexpr uint32_t maxPGroupOut = 128;
  constexpr uint32_t outputPerGroup = 64;
  constexpr float matMulQuantCoeff = 0.0625f;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int globalLinearId0 = ndi.get_group(0); // [0, head dim / 16)
  int globalLinearId1 = ndi.get_group(1); // [0, qHead)
  int globalLinearId2 = ndi.get_group(2); // [0, bs)
  uint32_t batchIdx = globalLinearId2;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  uint32_t headQ = headKv * gqaRatio;
  uint32_t reduceCount = (longestBatch + 1023) >> 10;
  uint32_t effectiveReduceCount = (kvSeqLen + 1023) >> 10;
  uint32_t channelOffset = globalLinearId0;
  uint32_t outDim = headQ * headDim * sizeof(float);
  uint32_t widthT = headQ * headDim * sizeof(float) - 1;
  uint32_t heightT = effectiveReduceCount - 1;
  float* batchPTempOut = pTempOut + batchIdx * reduceCount * headQ * headDim;
  uint32_t vX = channelOffset * 16 + headDim * globalLinearId1;
  uint32_t vY = 0;
  uint32_t loopCount = (effectiveReduceCount + 31) >> 5;
  uint32_t outOffset = channelOffset * 16 + headDim * globalLinearId1 + batchIdx * headQ * headDim;
  uint32_t offsetBaseSoftmaxSum = globalLinearId1 * sizeof(float) + batchIdx * reduceCount * headQ * sizeof(float);
  simd<float, 32 * 16> ttFp32;
  simd<float, 16> smFp32;
  simd<float, 16> smSumFp32 = 0.0f;
  simd<float, 16> output = 0.0f;
  simd<uint32_t, 16> coordReduce(baseOffsetInc16);
  simd<uint32_t, 16> simdOffsetSoftmaxSum;
  simd_mask<16> mask;
  simd_mask<16> negMask;

  for (uint32_t loop = 0; loop < loopCount; loop++) {
#pragma unroll
    for (uint32_t kk = 0; kk < 2; kk++) {
      simdOffsetSoftmaxSum = coordReduce * headQ * sizeof(float) + offsetBaseSoftmaxSum;
      mask = coordReduce < effectiveReduceCount;
      negMask = coordReduce >= effectiveReduceCount;
      smFp32 = 
        gather<
        float,
        16,
        1,
        uint32_t>(pSoftmaxSum, simdOffsetSoftmaxSum, mask);

      ttFp32.select<256, 1>(kk * 256) =
        __ESIMD_ENS::lsc_load_2d<
        float,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
        >(batchPTempOut, widthT, heightT, widthT, vX, vY);

#pragma unroll
      for (uint32_t kkk = 0; kkk < 16; kkk++) {
        ttFp32.select<16, 1>(256 * kk + 16 * kkk).merge(0.0f, coordReduce.replicate_w<16, 1>(kkk) >= effectiveReduceCount);
        output = output + ttFp32.select<16, 1>(256 * kk + 16 * kkk);
      }

      smFp32.merge(0.0f, negMask);
      smSumFp32 = smSumFp32 + smFp32;
      vY += 16;
      coordReduce = coordReduce + 16;
    }
  }

  float softmaxMul = __ESIMD_DNS::sum<float, float, 16>(smSumFp32.select<16, 1>(0));
  softmaxMul = 1.0f / softmaxMul;
  output.select<16, 1>(0) = output.select<16, 1>(0)* softmaxMul;
  simd<fp16, 16> outputTemp = output.select<16, 1>(0);
  block_store<fp16, 16>((fp16*)out + outOffset, outputTemp);
}