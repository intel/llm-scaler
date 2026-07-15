// splitk_kernel.h — the HD-templated KV-split-K paged-decode attention kernel,
// shared by the perf bench (splitk.cpp) and the correctness driver
// (correctness.cpp) so they exercise the SAME code, not two divergent copies.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>
#include <algorithm>

namespace skattn {
using namespace sycl;
using namespace sycl::ext::intel::esimd;

#define SK_FP32_MIN (-1e30f)

template <typename T, uint32_t N, uint32_t CHUNK = 128>
ESIMD_INLINE simd<T, N> loadVec(const T* p) {
  simd<T, N> v;
#pragma unroll
  for (uint32_t i = 0; i < N; i += CHUNK) v.template select<CHUNK,1>(i) = block_load<T,CHUNK>(p+i);
  return v;
}
ESIMD_INLINE float skExp(float x){ simd<float,1> v=x; simd<float,1> r=exp(v); return r[0]; }
template <uint32_t N> ESIMD_INLINE float skReduceSum(simd<float,N> v){
  if constexpr (N==1) return v[0];
  else { simd<float,N/2> h=v.template select<N/2,1>(0)+v.template select<N/2,1>(N/2); return skReduceSum<N/2>(h); }
}

template <typename T, uint32_t HEAD_DIM> class splitKern;
template <typename T, uint32_t HEAD_DIM> class reduceKern;

// Phase A: each WI (b,h,g) does online-softmax over its KV chunk -> (m,l,acc).
template <typename T, uint32_t HEAD_DIM>
ESIMD_INLINE void phaseSplit(const T* query, const T* kCache, const T* vCache,
    const uint32_t* pageTable, const uint32_t* seqLens,
    float* pAcc, float* pM, float* pL,
    uint32_t numQHeads, uint32_t numKvHeads, uint32_t pageSize, uint32_t pageSizeLog2,
    uint32_t pageTableStride, uint32_t G, uint32_t hostChunk, float scale, nd_item<1>& ndi) {
  const uint32_t gid = ndi.get_global_id(0);
  const uint32_t g = gid % G;
  const uint32_t t = gid / G;
  const uint32_t h = t % numQHeads;
  const uint32_t b = t / numQHeads;
  const uint32_t gqaRatio = numQHeads / numKvHeads, kvHead = h / gqaRatio;
  const uint32_t kvSeqLen = seqLens[b], pageMask = pageSize - 1;
  // Derive chunk from the REAL per-batch seqlen (not the host-passed value).
  // Host `chunk` was (max_seq+G-1)/G, and max_seq is the caller's hint — correct
  // when it equals ctx (eager) but WRONG under XPU graph where it is pinned to the
  // preallocated page-table width (~70016), which left all-but-one split idle for
  // any real ctx. Using kvSeqLen makes one (high) G optimal across all contexts:
  // every split gets ceil(kvSeqLen/G) tokens; splits past the end early-return.
  (void)hostChunk;
  const uint32_t chunk = (kvSeqLen + G - 1) / G;
  const uint32_t start = g * chunk;
  uint32_t end = start + chunk; if (end > kvSeqLen) end = kvSeqLen;
  const uint64_t pbase = (uint64_t)(t * G + g);

  if (start >= end) { pM[pbase] = SK_FP32_MIN; pL[pbase] = 0.0f; return; }
  simd<float, HEAD_DIM> qF = loadVec<T,HEAD_DIM>(query + (uint64_t)(b*numQHeads+h)*HEAD_DIM);
  simd<float, HEAD_DIM> acc = 0.0f; float m = SK_FP32_MIN, l = 0.0f;
  for (uint32_t j = start; j < end; j++) {
    const uint32_t physPage = pageTable[b*pageTableStride + (j >> pageSizeLog2)];
    const uint64_t kvIdx = ((uint64_t)physPage*pageSize + (j & pageMask))*numKvHeads + kvHead;
    simd<float, HEAD_DIM> kF = loadVec<T,HEAD_DIM>(kCache + kvIdx*HEAD_DIM);
    const float score = skReduceSum<HEAD_DIM>(qF*kF) * scale;
    const float mNew = m > score ? m : score;
    const float corr = skExp(m - mNew), p = skExp(score - mNew);
    l = l*corr + p;
    acc = acc*corr + loadVec<T,HEAD_DIM>(vCache + kvIdx*HEAD_DIM)*p;
    m = mNew;
  }
  pM[pbase] = m; pL[pbase] = l;
#pragma unroll
  for (uint32_t i = 0; i < HEAD_DIM; i += 128)
    block_store<float,128>(pAcc + pbase*HEAD_DIM + i, acc.template select<128,1>(i));
}

// Phase B: combine G partials via flash rescale -> final out.
template <typename T, uint32_t HEAD_DIM>
ESIMD_INLINE void phaseReduce(const float* pAcc, const float* pM, const float* pL, T* out,
    uint32_t numQHeads, uint32_t G, nd_item<1>& ndi) {
  const uint32_t t = ndi.get_global_id(0);
  float gm = SK_FP32_MIN;
  for (uint32_t g = 0; g < G; g++) { float mg = pM[t*G+g]; gm = gm > mg ? gm : mg; }
  simd<float, HEAD_DIM> acc = 0.0f; float denom = 0.0f;
  for (uint32_t g = 0; g < G; g++) {
    float lg = pL[t*G+g];
    if (lg <= 0.0f) continue;
    float w = skExp(pM[t*G+g] - gm);
    denom += w * lg;
    simd<float, HEAD_DIM> a;
#pragma unroll
    for (uint32_t i = 0; i < HEAD_DIM; i += 128)
      a.template select<128,1>(i) = block_load<float,128>(pAcc + (uint64_t)(t*G+g)*HEAD_DIM + i);
    acc = acc + a * w;
  }
  simd<T, HEAD_DIM> outT;
  if (denom > 0.0f) { simd<float,HEAD_DIM> o = acc * (1.0f/denom); outT = o; }
  else outT = simd<T,HEAD_DIM>(T(0));
#pragma unroll
  for (uint32_t i = 0; i < HEAD_DIM; i += 128)
    block_store<T,128>(out + (uint64_t)t*HEAD_DIM + i, outT.template select<128,1>(i));
}

// Host launcher. in_order queue required (phase B reads phase A's pAcc).
template <typename T, uint32_t HEAD_DIM>
inline void launch(queue& q, const T* query, const T* kCache, const T* vCache,
    const uint32_t* pageTable, const uint32_t* seqLens, T* out,
    float* pAcc, float* pM, float* pL,
    uint32_t B, uint32_t HQ, uint32_t HKV, uint32_t S, uint32_t PS, uint32_t psLog2,
    uint32_t ptStride, uint32_t G) {
  const uint32_t chunk = (S + G - 1) / G;
  const float scale = 1.0f / sycl::sqrt((float)HEAD_DIM);
  // Phase B reads phase A's pAcc/pM/pL. In-order queue serializes them eager,
  // but under SYCL-Graph capture each submit is an independent node and the
  // implicit in-order edge is NOT recorded -> on replay phase B races phase A
  // -> garbage. Capture phase A's event; phase B depends_on it (real graph edge).
  auto evA = q.submit([&](handler& h){
    h.parallel_for<splitKern<T,HEAD_DIM>>(nd_range<1>(range<1>((size_t)B*HQ*G), range<1>(1)),
      [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        phaseSplit<T,HEAD_DIM>(query,kCache,vCache,pageTable,seqLens,pAcc,pM,pL,
          HQ,HKV,PS,psLog2,ptStride,G,chunk,scale,it);
      });
  });
  q.submit([&](handler& h){
    h.depends_on(evA);
    h.parallel_for<reduceKern<T,HEAD_DIM>>(nd_range<1>(range<1>((size_t)B*HQ), range<1>(1)),
      [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        phaseReduce<T,HEAD_DIM>(pAcc,pM,pL,out,HQ,G,it);
      });
  });
}

}  // namespace skattn
