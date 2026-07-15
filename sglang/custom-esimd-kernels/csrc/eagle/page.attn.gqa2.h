#ifndef PAGE_ATTN_GQA2_H
#define PAGE_ATTN_GQA2_H
// GQA ratio = 2 variant of page.attn.h
// Processes 2 Q heads per work-group (2 threads, each handling 1 Q head).
// K and V caches are passed as two independent tensors, each shaped
// [reserved_size, page_size, head_num, head_dim] with page stride =
// kvCacheBatchStride1.
//
// Unlike GQA4 which has 4 threads each computing 64 dims of the 256-dim dot
// product (then reducing via SLM barrier), GQA2 has 2 threads each computing
// the full 256-dim dot product for their own Q head independently.
//
// Phase2 in GQA4 uses local_size=2 to split each 64-token P block across
// 2 threads (each handling 32 tokens). In GQA2, each thread handles one
// full Q head independently: loads all 64 P values and all 64 V rows
// per outer-loop iteration (4 x 16-row V loads instead of 2 x 16-row).
//
// Kernels templated on storage dtype T (fp16 or bf16). Compute is fp32.

template <typename T>
ESIMD_INLINE void sdpaDecodeGqa2Phase1(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* pState,
  float* pGroupMax,
  float* pGlobalMax,
  uint32_t* pPollP,
  uint32_t* pageTable,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
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
  // No SLM needed -- each thread independently computes the full dot product
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0); // [0, kvHead * gqaGroups)
  int globalLinearId1 = ndi.get_group(1); // [0, kvSeqLen / 64)
  int globalLinearId2 = ndi.get_group(2); // [0, bs)
  int hh = localLinearId & 0x1; // thread index within work-group (0 or 1)
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 2;
  int kvHeadIdx = globalLinearId0 / gqaGroups;
  int qGroupIdx = globalLinearId0 % gqaGroups;
  int qHeadIdx = kvHeadIdx * gqaRatio + qGroupIdx * 2 + hh;
  uint32_t pageTableSize = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = pageTableSize - 1;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;
  simd<uint32_t, 16> simdOffsetsK;

  simd<uint32_t, 16> baseOffsetInc16AsSimd(baseOffsetInc16);
  // Each thread loads one full Q head (256 dims) in 4 chunks of 64 dims each,
  // matching the GQA4 dim-partitioning pattern for K-loading compatibility.
  simd<float, 64> qqFp32_0; // dims [0..31] and [128..159]
  simd<float, 64> qqFp32_1; // dims [32..63] and [160..191]
  simd<float, 64> qqFp32_2; // dims [64..95] and [192..223]
  simd<float, 64> qqFp32_3; // dims [96..127] and [224..255]
  simd<T, 64> qqT_chunk;
  simd<float, 16> ppFp32;
  simd<T, 16 * 64> kkCache;
  simd<float, 1> ppMax;

  uint32_t headQ = headKv * gqaRatio;
  uint64_t outputOffset = qHeadIdx * pStride + globalLinearId1 * outputPerGroup + batchIdx * headQ * pStride;
  uint32_t outputMaxOffset = qHeadIdx * maxStride + globalLinearId1 + batchIdx * headQ * maxStride;
  uint32_t offsetQBase = qHeadIdx * headDim * sizeof(T) + batchIdx * headQ * headDim * sizeof(T);
  uint32_t baseCoordK = globalLinearId1 * outputPerGroup;
  uint32_t offsetBaseK = headDim * kvHeadIdx * sizeof(T);
  uint32_t kvSeqOffset = baseCoordK;

  if (baseCoordK >= kvSeqLen) {
    return;
  }

  // Load Q in 4 chunks matching the K dim-partition pattern.
  // Each chunk covers dims [hPart*32..hPart*32+31, 128+hPart*32..128+hPart*32+31]
#pragma unroll
  for (int qk = 0; qk < 2; qk++) {
    qqT_chunk.template bit_cast_view<uint8_t>().template select<64, 1>(64 * qk) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      64,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQBase + 0 * 32 * sizeof(T) + qk * 4 * 32 * sizeof(T));
  }
  qqFp32_0 = qqT_chunk;

#pragma unroll
  for (int qk = 0; qk < 2; qk++) {
    qqT_chunk.template bit_cast_view<uint8_t>().template select<64, 1>(64 * qk) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      64,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQBase + 1 * 32 * sizeof(T) + qk * 4 * 32 * sizeof(T));
  }
  qqFp32_1 = qqT_chunk;

#pragma unroll
  for (int qk = 0; qk < 2; qk++) {
    qqT_chunk.template bit_cast_view<uint8_t>().template select<64, 1>(64 * qk) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      64,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQBase + 2 * 32 * sizeof(T) + qk * 4 * 32 * sizeof(T));
  }
  qqFp32_2 = qqT_chunk;

#pragma unroll
  for (int qk = 0; qk < 2; qk++) {
    qqT_chunk.template bit_cast_view<uint8_t>().template select<64, 1>(64 * qk) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      64,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQBase + 3 * 32 * sizeof(T) + qk * 4 * 32 * sizeof(T));
  }
  qqFp32_3 = qqT_chunk;

  simd<float, 64> ppOutFull;

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

      // Iterate over all 4 K-dim partitions (equivalent to what 4 threads do in GQA4)
#pragma unroll
      for (int hPart = 0; hPart < 4; hPart++) {
        simdOffsetsK = baseOffsetInc16AsSimd;
        simdOffsetsK =
          simdOffsetsK * headKv * headDim * sizeof(T) +
          offsetBaseK +
          hPart * 32 * sizeof(T) +
          pageIdx * kvCacheBatchStride1 * sizeof(T) +
          pageTableOffset * headKv * headDim * sizeof(T)
          ;

#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
#pragma unroll
          for (int kkk = 0; kkk < 4; kkk++) {
            kkCache.template bit_cast_view<uint32_t>().template select<64, 1>(256 * kk + 64 * kkk) =
              __ESIMD_ENS::lsc_gather<
              uint32_t,
              4,
              __ESIMD_ENS::lsc_data_size::u32,
              __ESIMD_ENS::cache_hint::cached,
              __ESIMD_ENS::cache_hint::cached,
              16,
              uint32_t
              >((uint32_t*)kState, simdOffsetsK, mask);
            simdOffsetsK += 8 * sizeof(T);
          }

          simdOffsetsK += 3 * 32 * sizeof(T);

#pragma unroll
          for (int kkk = 0; kkk < 16; kkk++) {
            kkFp32.select<16, 1>(32 * kkk + 16 * 0) = kkCache.template select<16, 2>(512 * kk + 32 * kkk + 0);
            kkFp32.select<16, 1>(32 * kkk + 16 * 1) = kkCache.template select<16, 2>(512 * kk + 32 * kkk + 1);
          }

#pragma unroll
          for (int kkk = 0; kkk < 32; kkk++) {
            ppFp32 = ppFp32 + kkFp32.select<16, 1>(16 * kkk) *
              ((hPart == 0) ? qqFp32_0[32 * kk + kkk] :
               (hPart == 1) ? qqFp32_1[32 * kk + kkk] :
               (hPart == 2) ? qqFp32_2[32 * kk + kkk] : qqFp32_3[32 * kk + kkk]);
          }
        }
      }

      ppFp32.merge(FP32_MIN, maskNeg);
    }
    else {
      ppFp32 = FP32_MIN;
    }

    ppOutFull.select<16, 1>(loopIdx * 16) = ppFp32;
    kvSeqOffset += 16;
  }

  // Apply scaling and compute softmax exp
  ppOutFull = ppOutFull * matMulQuantCoeff;
  simd<float, 16> maxTemp;
  maxTemp = ppOutFull.select<16, 1>(0);
#pragma unroll
  for (int pk = 1; pk < 4; pk++) {
    maxTemp = __ESIMD_NS::max<float, 16, float>(maxTemp, ppOutFull.select<16, 1>(16 * pk));
  }

  maxTemp.select<8, 1>(0) = __ESIMD_NS::max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
  maxTemp.select<4, 1>(0) = __ESIMD_NS::max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));
  maxTemp.select<2, 1>(0) = __ESIMD_NS::max<float, 2, float>(maxTemp.select<2, 1>(0), maxTemp.select<2, 1>(2));
  ppMax[0] = __ESIMD_NS::max<float>(maxTemp[0], maxTemp[1]);
  ppOutFull = ppOutFull - (float)ppMax[0];
  ppOutFull = __ESIMD_NS::pow<float, 64, float>(2.718281828459f, ppOutFull);

  {
    block_store<float, 64>((float*)pState + outputOffset, ppOutFull);
    simd<float, 1> zeros = 0.0f;
    pGroupMax[outputMaxOffset] = ppMax[0];
    uint32_t atomicOffset = (batchIdx * gqaRatio * headKv + qHeadIdx) * sizeof(float);
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
  }
}

template <typename T>
ESIMD_INLINE void sdpaDecodeGqa2Phase2(
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
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0); // [0, head dim / 16)
  int globalLinearId1 = ndi.get_group(1); // [0, kvHead * gqaGroups)
  int globalLinearId2 = ndi.get_group(2); // [0, bs)
  int hh = localLinearId & 0x1; // thread 0 or 1, each handles one Q head
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 2;
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

  // 64 P values (full output from one Phase1 work-group for this Q head)
  simd<float, 64> ppFp32;
  simd<float, 1> historicMax;
  // V cache: 4 x 16 rows x 16 cols (load all 64 seq positions)
  simd<T, 64 * 16> vvCache;
  simd<float, 16 * 16> vvFp32;
  simd<float, 16> softmaxSum;
  simd<float, 16> outputFp32;

  if (groupIdx >= kvSeqOutGroup) {
    return;
  }
  uint32_t reduceCount = (longestBatch + 1023) >> 10;
  uint32_t headQ = headKv * gqaRatio;
  uint32_t qHeadIdx = kvHeadIdx * gqaRatio + qGroupIdx * 2 + hh;
  uint32_t outputOffset = qHeadIdx * headDim + channelOffset * 16 + batchIdx * headQ * headDim;
  uint32_t outTempOffset = qHeadIdx * headDim + channelOffset * 16 + groupIdx * headQ * headDim + batchIdx * reduceCount * headQ * headDim;
  uint32_t outOffsetSoftmaxSum = qHeadIdx + groupIdx * headQ + batchIdx * reduceCount * headQ;
  uint32_t offsetP = qHeadIdx * pStride + groupIdx * 1024 + batchIdx * headQ * pStride;
  uint32_t offsetMax = qHeadIdx * maxStride + batchIdx * headQ * maxStride + groupIdx * 16;
  uint32_t offsetGlobalMax = batchIdx * headQ + qHeadIdx;
  uint32_t widthV = headKv * headDim * sizeof(T) - 1;
  uint32_t heightV = pageTableSize - 1;
  uint32_t vX = channelOffset * 16 + headDim * kvHeadIdx;
  uint32_t vY = 0;
  uint32_t totalPages = (kvSeqLen + pageTableSize - 1) >> pageTableSizeLog2;
  uint32_t lastPageHeight = (kvSeqLen - 1) & (pageTableSize - 1);
  if (kvSeqLen == 0) {
    lastPageHeight = 0;
  }
  T* vPtrBase = (T*)vState;

  historicMax[0] = FP32_MIN * matMulQuantCoeff;
  softmaxSum = 0;
  outputFp32 = 0;

  simd<float, 1> globalMax;
  simd<float, 1> currMax;
  simd<float, 1> compensationP;
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
  globalMax[0] = *(pGlobalMax + offsetGlobalMax);

#pragma unroll
  for (uint32_t loop = 0; loop < 16; loop++) {
    if (kvSeqOffset + loop * 64 < kvSeqLen) {
      T* currPtrV = vPtrBase + pageAlignedOffsets[loop];
      if (pageTableIndice[loop] + 1 < totalPages) {
        heightV = pageTableSize - 1;
      }
      else {
        heightV = lastPageHeight;
      }
      vY = pageOffsetsInner[loop];

      // Load all 64 P values for this Q head from Phase1 output
      ppFp32 = block_load<float, 64>((float*)pState + offsetP);
      currMax[0] = pGroupMax[offsetMax];

      // Load 64 V rows in 4 x 16-row blocks
      vvCache.template select<256, 1>(0 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        T,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
        >(currPtrV, widthV, heightV, widthV, vX, vY);

      vvCache.template select<256, 1>(1 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        T,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>
        (currPtrV, widthV, heightV, widthV, vX, vY + 16);

      vvCache.template select<256, 1>(2 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        T,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>
        (currPtrV, widthV, heightV, widthV, vX, vY + 32);

      vvCache.template select<256, 1>(3 * 256) =
        __ESIMD_ENS::lsc_load_2d<
        T,
        16,
        16,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>
        (currPtrV, widthV, heightV, widthV, vX, vY + 48);

      compensationP = currMax - globalMax;
      compensationP = exp(compensationP);

      ppFp32 = ppFp32 * (float)compensationP[0];

      // Accumulate softmax sum over all 64 P values
#pragma unroll
      for (int pk = 0; pk < 4; pk++) {
        softmaxSum = softmaxSum + ppFp32.select<16, 1>(16 * pk);
      }

      // Weighted sum: output += V^T * P (16 output channels += 64 tokens * P weights)
#pragma unroll
      for (int vk = 0; vk < 4; vk++) {
        vvFp32 = vvCache.template select<256, 1>(256 * vk);
#pragma unroll
        for (int pk = 0; pk < 16; pk++) {
          outputFp32 += vvFp32.select<16, 1>(16 * pk) * ppFp32[16 * vk + pk];
        }
      }

      offsetP += 64;
      offsetMax += 1;
    }
  }

  // Each thread has its own complete result -- no cross-thread reduction needed
  float softmaxMulVal = __ESIMD_DNS::sum<float, float, 16>(softmaxSum);

  if ((flag & 0x1) == 1) {
    softmaxMulVal = 1.0f / softmaxMulVal;
    outputFp32 = outputFp32 * softmaxMulVal;
    simd<T, 16> outputTemp = outputFp32;
    block_store<T, 16>((T*)out + outputOffset, outputTemp);
  }
  else {
    block_store<float, 16>((float*)pTempOut + outTempOffset, outputFp32);
    simd<float, 1> softmaxMulSimd = softmaxMulVal;
    block_store<float, 1>((float*)pSoftmaxSum + outOffsetSoftmaxSum, softmaxMulSimd);
  }
}

// Phase3 for GQA2 - reuses the same logic as GQA4 Phase3 since it operates
// per-qHead independently. Provided here as sdpaDecodeGqa2Phase3 for naming
// consistency, but it is identical in logic to sdpaDecodeGqa4Phase3.
template <typename T>
ESIMD_INLINE void sdpaDecodeGqa2Phase3(
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
  simd<T, 16> outputTemp = output.select<16, 1>(0);
  block_store<T, 16>((T*)out + outOffset, outputTemp);
}

#endif // PAGE_ATTN_GQA2_H
