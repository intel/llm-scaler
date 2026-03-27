// SDP ESIMD Flash Attention DLL — parameterized for hardware adaptation
// Compiled with: icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF"
// Exports: sdp_fp16 (FP16 optimized), sdp_bf16io (BF16 I/O hybrid)
// Tensor shape: [B, L, H, D] with D=HEAD_DIM, B=1, contiguous
//
// Hardware config is selected at compile time via sdp_config.h.
// To target a different GPU, define SDP_CONFIG_PVC / SDP_CONFIG_LNL before including.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "esimd_kernel_api.h"
#include "sdp_config.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef  ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))
#define FP32_MIN -3.402823466e+38f

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

// Import active configuration
using Cfg = sdp_config::ActiveConfig;

#include "single_kernels/flash.attn.b.mha128.fp16.opt.h"
#include "single_kernels/flash.attn.b.mha128.bf16io.h"

// ──────────────────────────────────────────────────────────────────────────────
// sdp_fp16: FP16 optimized Flash Attention
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sdp_fp16(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr)
{
    sycl::queue& q = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);

    int groupH = headQ;
    int groupV = (q_len + Cfg::Q_GROUP - 1) / Cfg::Q_GROUP;
    sycl::nd_range<2> ndr(
        {(size_t)(Cfg::WG_SIZE * groupH), (size_t)groupV},
        {(size_t)Cfg::WG_SIZE, 1});

    uint8_t* pQ = reinterpret_cast<uint8_t*>(Q);
    uint8_t* pK = reinterpret_cast<uint8_t*>(K);
    uint8_t* pV = reinterpret_cast<uint8_t*>(V);
    uint8_t* pA = reinterpret_cast<uint8_t*>(normAlpha);
    uint8_t* pO = reinterpret_cast<uint8_t*>(out);
    uint32_t aLen  = (uint32_t)q_len;
    uint32_t kvLen = (uint32_t)kv_len;
    uint32_t hQ    = (uint32_t)headQ;
    uint32_t hKv   = (uint32_t)headKv;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(
                pQ, pK, pV, pA, pO,
                aLen, kvLen, hQ, hKv, ndi);
        });
    }).wait();
}

// ──────────────────────────────────────────────────────────────────────────────
// sdp_bf16io: BF16 I/O hybrid Flash Attention
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sdp_bf16io(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr)
{
    sycl::queue& q = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);

    int groupH = headQ;
    int groupV = (q_len + Cfg::Q_GROUP - 1) / Cfg::Q_GROUP;
    sycl::nd_range<2> ndr(
        {(size_t)(Cfg::WG_SIZE * groupH), (size_t)groupV},
        {(size_t)Cfg::WG_SIZE, 1});

    uint8_t* pQ = reinterpret_cast<uint8_t*>(Q);
    uint8_t* pK = reinterpret_cast<uint8_t*>(K);
    uint8_t* pV = reinterpret_cast<uint8_t*>(V);
    uint8_t* pA = reinterpret_cast<uint8_t*>(normAlpha);
    uint8_t* pO = reinterpret_cast<uint8_t*>(out);
    uint32_t aLen  = (uint32_t)q_len;
    uint32_t kvLen = (uint32_t)kv_len;
    uint32_t hQ    = (uint32_t)headQ;
    uint32_t hKv   = (uint32_t)headKv;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(
                pQ, pK, pV, pA, pO,
                aLen, kvLen, hQ, hKv, ndi);
        });
    }).wait();
}
