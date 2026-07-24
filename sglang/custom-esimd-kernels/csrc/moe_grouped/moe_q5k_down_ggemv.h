#pragma once
// moe_q5k_down_ggemv.h — Q5_K-native grouped DOWN GGEMV for MoE prefill (doubleGRF).
// Mirrors moe_q4k_ggemv.h (up) but:
//   - 5-bit weight: v5 = nibble | (qh_bit<<4); w = v5*scale - min   (asymmetric)
//   - qh in PLAIN ELEMENT ORDER: qh[N,K/8] u8, byte j bit b = elem 8j+b
//     (NOT the 512-tile pre-shuffle the decode kernel uses; chosen so the DPAS
//      K-loop can read the 5th bit with simple indexing).
// down: K = intermediate_size, N(out) = hidden_size. group=32 (Q5_K).
// Weight ql [E, H, IM/2] u8 interleaved; qh [E, H, IM/8] u8; scale,min [E,H,IM/32].

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
inline sycl::event moe_down_q5k_ggemv_t(
    sycl::queue& q,
    const fp16* intermediate, const uint8_t* w_ql, const uint8_t* w_qh,
    const fp16* w_scale, const fp16* w_min,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size)
{
    constexpr int BS = 32;          // Q5_K group size
    constexpr int MS = MS_;
    constexpr int MAX_M = MS * 16;
    constexpr int N = N_;           // N-cols/WI (32 default; 8/16 = more WIs/occupancy)
    constexpr int NS = N / 8;
    constexpr int ACC_SZ = MS * NS * 128;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, hidden_size / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int nblocks = intermediate_size / BS;   // IM/32
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
            const uint8_t* qh_ptr = w_qh + row_base * (intermediate_size / 8);

            simd<float, ACC_SZ> acc(0.f);

            const simd<uint32_t, N> scl_byte_off =
                simd<uint32_t, N>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
            const fp16* s_base = w_scale + row_base * nblocks;
            const fp16* m_base = w_min   + row_base * nblocks;

            for (int blk = 0; blk < nblocks; blk++) {
                simd<fp16, N> scl = gather<fp16, N>(s_base + blk, scl_byte_off);
                simd<fp16, N> mnv = gather<fp16, N>(m_base + blk, scl_byte_off);

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

                    // 16 K-elements this sub-tile: bytes [x_byte, x_byte+8) of ql,
                    // qh bits at element (blk*BS + k_base*2)..+16 = 2 qh bytes/row.
                    const int x_byte = blk * (BS / 2) + k_base;       // ql byte offset
                    const int elem0  = blk * BS + k_base * 2;          // first elem
                    const int qh_byte = elem0 / 8;                     // 16 elems = 2 bytes
                    auto w_raw_all = xesimd::lsc_load_2d<uint8_t, 8, N, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        w_ptr, (unsigned)(intermediate_size/2-1), (unsigned)(N-1),
                        (unsigned)(intermediate_size/2-1), x_byte, 0);
                    // qh: 2 bytes per N-row covering this 16-elem segment
                    auto qh_raw_all = xesimd::lsc_load_2d<uint8_t, 8, N, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        qh_ptr, (unsigned)(intermediate_size/8-1), (unsigned)(N-1),
                        (unsigned)(intermediate_size/8-1), qh_byte, 0);

                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++) {
                        simd<uint8_t, 64> w_raw = w_raw_all.template select<64, 1>(ns * 64);
                        simd<uint8_t, 64> lo = w_raw & (uint8_t)0x0F;
                        simd<uint8_t, 64> hi = w_raw >> 4;
                        // reconstruct v5 per element: low nibble -> even elem (2j), high -> 2j+1.
                        // Build w_tile [8 rows x 16 elems] = interleaved lo/hi, then add (qh_bit<<4).
                        simd<fp16, 128> w_tile;
                        // base nibble values
                        simd<fp16, 64> lo_val = convert<fp16>(lo);
                        simd<fp16, 64> hi_val = convert<fp16>(hi);
                        w_tile.template select<64, 2>(0) = lo_val;   // elem 2j
                        w_tile.template select<64, 2>(1) = hi_val;   // elem 2j+1
                        // add 5th bit. qh_raw_all is row-major [32 rows x 8 bytes]:
                        // row (ns*8+r)'s 2 needed qh bytes are at flat (ns*8+r)*8 + {0,1}.
                        // element e in [0,16): bit = (qh16 >> e) & 1, value += bit*16, where
                        // qh16 = b0 | (b1<<8). VECTORIZED per row (mirrors up's select<16,1>
                        // dequant; replaces the old scalar `for e` lane-indexed R/W that
                        // starved the loads (down slower than up), notes §10ax).
                        const simd<uint32_t, 16> sh5(0, 1);   // 0,1,...,15
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            const int qrow = (ns * 8 + r) * 8;
                            uint32_t qh16 = (uint32_t)qh_raw_all[qrow + 0]
                                          | ((uint32_t)qh_raw_all[qrow + 1] << 8);
                            fp16 ws = scl[ns*8+r];
                            fp16 wm = mnv[ns*8+r];
                            simd<uint32_t, 16> bit = (simd<uint32_t, 16>(qh16) >> sh5) & (uint32_t)1;
                            simd<fp16, 16> add = convert<fp16>(bit) * (fp16)16;  // 0 or 16
                            auto wr = w_tile.template select<16, 1>(r * 16);
                            wr = (wr + add) * ws - wm;
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

inline sycl::event moe_down_q5k_ggemv(
    sycl::queue& q, const fp16* intermediate, const uint8_t* w_ql,
    const uint8_t* w_qh, const fp16* w_scale, const fp16* w_min,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size) {
    return moe_down_q5k_ggemv_t<4>(q, intermediate, w_ql, w_qh, w_scale, w_min,
        expert_output, chunks, num_chunks, intermediate_size, hidden_size);
}
inline sycl::event moe_down_q5k_ggemv_smallm(
    sycl::queue& q, const fp16* intermediate, const uint8_t* w_ql,
    const uint8_t* w_qh, const fp16* w_scale, const fp16* w_min,
    fp16* expert_output, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size) {
    // small-M occupancy: finer N-tile (32->16) multiplies WIs (notes §10..),
    // improves small-M throughput. MOE_N_TILE env A/B; default 16.
    static const int NT = [](){ const char* e=std::getenv("MOE_N_TILE"); return e?atoi(e):16; }();
    auto f = (NT==8)  ? moe_down_q5k_ggemv_t<1,8>
           : (NT==32) ? moe_down_q5k_ggemv_t<1,32>
                      : moe_down_q5k_ggemv_t<1,16>;
    return f(q, intermediate, w_ql, w_qh, w_scale, w_min,
        expert_output, chunks, num_chunks, intermediate_size, hidden_size);
}
