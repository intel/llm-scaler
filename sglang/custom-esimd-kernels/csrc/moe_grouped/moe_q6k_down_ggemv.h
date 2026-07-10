#pragma once
// moe_q6k_down_ggemv.h — Q6_K-native grouped DOWN GGEMV for MoE prefill (doubleGRF).
// Mirrors moe_q5k_down_ggemv.h but Q6_K:
//   - 6-bit v6 = nibble | (qh_2bit<<4); w = scale*(v6 - 32)  (SYMMETRIC, no min)
//   - scale group = 16 (Q6_K per-16), vs Q5_K group=32
//   - qh in PLAIN element order: 2 bits/elem packed qh[N,K/4] (byte j = elems 4j..4j+3,
//     2 bits each), chosen so the DPAS K-loop reads the 2 high bits with simple indexing.
// down: K = intermediate_size, N(out) = hidden_size. group=16.
// Weight ql [E, H, IM/2] u8 interleaved; qh [E, H, IM/4] u8; scale [E, H, IM/16] fp16.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdlib>

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx_ns = sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;

#ifndef MOE_Q4K_CHUNK_DEFINED
#define MOE_Q4K_CHUNK_DEFINED
struct MoeQ4KChunkInfo { int eid; int t0; int nt; };
#endif

template <int MS_, int N_ = 32>
inline sycl::event moe_down_q6k_ggemv_t(
    sycl::queue& q,
    const fp16* intermediate, const uint8_t* w_ql, const uint8_t* w_qh,
    const fp16* w_scale,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size)
{
    constexpr int BS = 16;          // Q6_K scale group size
    constexpr int MS = MS_;
    constexpr int MAX_M = MS * 16;
    constexpr int N = N_;           // N-cols/WI (32 default; 8/16 = more WIs/occupancy)
    constexpr int NS = N / 8;
    constexpr int ACC_SZ = MS * NS * 128;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, hidden_size / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int nblocks = intermediate_size / BS;   // IM/16
            const int cid = (int)id[0];
            const int tid = (int)id[1];
            const int d_base = tid * N;

            const int eid = chunks[cid].eid;
            const int t0  = chunks[cid].t0;
            const int nt  = chunks[cid].nt;
            if (nt <= 0) return;

            simd<uint32_t, MAX_M> in_off =
                min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)t0,
                    simd<uint32_t, MAX_M>((uint32_t)(t0 + nt - 1)))
                * (uint32_t)(intermediate_size * sizeof(fp16));

            const size_t row_base = (size_t)eid * hidden_size + d_base;
            const uint8_t* w_ptr  = w_ql + row_base * (intermediate_size / 2);
            const uint8_t* qh_ptr = w_qh + row_base * (intermediate_size / 4);

            simd<float, ACC_SZ> acc(0.f);

            const simd<uint32_t, N> scl_byte_off =
                simd<uint32_t, N>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
            const fp16* s_base = w_scale + row_base * nblocks;

            // BS=16 group = 8 ql bytes (BS/2). Inner k_base loop 0..BS/2 step 8 -> 1 iter
            // covering the 16-elem group; one scale per block. 16 elems = 4 qh bytes (2bit each).
            for (int blk = 0; blk < nblocks; blk++) {
                simd<fp16, N> scl = gather<fp16, N>(s_base + blk, scl_byte_off);

                for (int k_base = 0; k_base < BS / 2; k_base += 8) {
                    const uint32_t k_off = (uint32_t)(blk * BS + k_base * 2) * (uint32_t)sizeof(fp16);
                    simd<fp16, MS * 256> b_tile;
                    #pragma unroll
                    for (int ms = 0; ms < MS; ms++) {
                        b_tile.template select<256, 1>(ms * 256).template bit_cast_view<uint32_t>() =
                            xesimd::lsc_gather<uint32_t, 8,
                                xesimd::lsc_data_size::u32,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                                16, uint32_t>(
                                reinterpret_cast<const uint32_t*>(intermediate),
                                in_off.template select<16, 1>(ms * 16).read() + k_off);
                    }

                    const int x_byte = blk * (BS / 2) + k_base;       // ql byte offset
                    const int elem0  = blk * BS + k_base * 2;          // first elem
                    const int qh_byte = elem0 / 4;                     // 16 elems = 4 qh bytes (2bit)
                    auto w_raw_all = xesimd::lsc_load_2d<uint8_t, 8, N, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        w_ptr, (unsigned)(intermediate_size/2-1), (unsigned)(N-1),
                        (unsigned)(intermediate_size/2-1), x_byte, 0);
                    auto qh_raw_all = xesimd::lsc_load_2d<uint8_t, 8, N, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        qh_ptr, (unsigned)(intermediate_size/4-1), (unsigned)(N-1),
                        (unsigned)(intermediate_size/4-1), qh_byte, 0);

                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++) {
                        simd<uint8_t, 64> w_raw = w_raw_all.template select<64, 1>(ns * 64);
                        simd<uint8_t, 64> lo = w_raw & (uint8_t)0x0F;
                        simd<uint8_t, 64> hi = w_raw >> 4;
                        simd<fp16, 128> w_tile;
                        w_tile.template select<64, 2>(0) = convert<fp16>(lo);  // elem 2j nibble
                        w_tile.template select<64, 2>(1) = convert<fp16>(hi);  // elem 2j+1 nibble
                        // add 2-bit qh per element. qh_raw_all row-major [32 rows x 8 bytes];
                        // row (ns*8+r)'s 4 needed qh bytes at flat (ns*8+r)*8 + {0..3}.
                        // element e in [0,16): byte e/4, bits 2*(e%4)..+1. In the combined
                        // q32 = qb0|qb1<<8|qb2<<16|qb3<<24, element e's 2-bit field is at bit
                        // 8*(e/4)+2*(e%4) == 2*e. So h2 = (q32 >> 2e) & 3. v6 = nibble+h2*16,
                        // w = scale*(v6-32). VECTORIZED per row (mirrors up; replaces the
                        // scalar `for e` lane R/W that capped down at ~28 GB/s, notes §10ax).
                        const simd<uint32_t, 16> sh6 = simd<uint32_t, 16>(0, 1) * (uint32_t)2;  // 0,2,..30
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            const int qrow = (ns * 8 + r) * 8;
                            uint32_t q32 = (uint32_t)qh_raw_all[qrow + 0]
                                         | ((uint32_t)qh_raw_all[qrow + 1] << 8)
                                         | ((uint32_t)qh_raw_all[qrow + 2] << 16)
                                         | ((uint32_t)qh_raw_all[qrow + 3] << 24);
                            fp16 ws = scl[ns*8+r];
                            simd<uint32_t, 16> h2 = (simd<uint32_t, 16>(q32) >> sh6) & (uint32_t)3;
                            simd<fp16, 16> add = convert<fp16>(h2) * (fp16)16;  // h2*16
                            auto wr = w_tile.template select<16, 1>(r * 16);
                            wr = (wr + add - (fp16)32) * ws;
                        }

                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            const int idx = ms * NS * 128 + ns * 128;
                            simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                            simd<float, 128> a = acc.template select<128, 1>(idx);
                            a = xmx_ns::dpas<8, 8, float, float, fp16, fp16>(a, bv, w_tile);
                            acc.template select<128, 1>(idx) = a;
                        }
                    }
                }
            }

            #pragma unroll
            for (int ms = 0; ms < MS; ms++) {
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    int tok = (ms * 16 + m < nt) ? (t0 + ms * 16 + m) : (t0 + nt - 1);
                    simd<float, N> row_f;
                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++)
                        row_f.template select<8, 1>(ns * 8) =
                            acc.template select<8, 16>(ms * NS * 128 + ns * 128 + m);
                    block_store<fp16, N>(expert_output + (size_t)tok * hidden_size + d_base,
                                         convert<fp16>(row_f));
                }
            }
        });
    });
}

inline sycl::event moe_down_q6k_ggemv(
    sycl::queue& q, const fp16* intermediate, const uint8_t* w_ql,
    const uint8_t* w_qh, const fp16* w_scale,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size) {
    return moe_down_q6k_ggemv_t<4>(q, intermediate, w_ql, w_qh, w_scale,
        expert_output, chunks, num_chunks, intermediate_size, hidden_size);
}
inline sycl::event moe_down_q6k_ggemv_smallm(
    sycl::queue& q, const fp16* intermediate, const uint8_t* w_ql,
    const uint8_t* w_qh, const fp16* w_scale,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size) {
    // small-M occupancy: finer N-tile (32->16) multiplies WIs (notes §10..).
    // MOE_N_TILE env A/B; default 16.
    static const int NT = [](){ const char* e=std::getenv("MOE_N_TILE"); return e?atoi(e):16; }();
    auto f = (NT==8)  ? moe_down_q6k_ggemv_t<1,8>
           : (NT==32) ? moe_down_q6k_ggemv_t<1,32>
                      : moe_down_q6k_ggemv_t<1,16>;
    return f(q, intermediate, w_ql, w_qh, w_scale,
        expert_output, chunks, num_chunks, intermediate_size, hidden_size);
}
