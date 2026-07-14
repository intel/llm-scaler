#pragma once
// moe_q4k_ggemv.h — Q4_K-native grouped GGEMV for MoE prefill (doubleGRF DPAS).
// Ported from temp/ moe_prefill_lgrf.h moe_up_forward_v29, adapted for GGUF Q4_K:
//   - group_size BS: 128 -> 32 (Q4_K per-32 scale+min)
//   - dequant: (nibble-8)*scale  ->  nibble*scale - min  (asymmetric, drop -8)
//   - per-group: gather BOTH scale[N,K/32] and min[N,K/32]
// Weight layout = my _xpu_repack_q4_k: ql[N,K/2] interleaved (low->2j, high->2j+1),
// scale/minv [N,K/32] fp16. dequant matches _xpu_dequant_q4_k: w = scale*nibble - min.
//
// DPAS tiling (same as temp/): MS=4 (64 tok), NS=4 (32 N-cols), DPAS 8x8, doubleGRF.

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

// Fused Gate+Up Q4_K GGEMV. gate/up weights passed SEPARATELY (zero extra memory
// — reuses the decode reps; no [E,2*IM,...] concat copy):
//   gate_ql/up_ql [E, IM, H/2] u8 interleaved; gate_sc/up_sc, gate_mn/up_mn [E, IM, H/32].
// Output tile for column r_base in [0,IM) reads gate; [IM,2*IM) reads up (row r_base-IM).
// gate_buf [total_seqlen, 2*IM] (cols 0:IM gate, IM:2IM up).
// tok_ids: optional [total_seqlen] int32 per-route ORIGINAL token id (the
// expert-sort permutation tok_sorted). When non-null, the kernel reads the
// activation directly from `expert_states` at row tok_ids[t0+m] — folding the
// per-expert input gather INTO the load (no `es = xf.index_select(tok_sorted)`
// round-trip; saves the IndexKernel, notes §10be). When null,
// reads contiguous rows t0..t0+nt (legacy: caller pre-gathered).
// MS_ = row-tiles of 16 tokens each (MAX_M = MS_*16). MS_=4 -> 64-token tile
// (prefill, large M). MS_=1 -> 16-token tile: for SMALL M (e.g. MTP verify M=4)
// the 64-row DPAS wastes ~60/64 rows; a 16-row tile does much less matmul +
// smaller acc + the same b_tile gather. Templated so both variants share one body.
template <int MS_, int N_ = 32>
inline sycl::event moe_up_q4k_ggemv_t(
    sycl::queue& q,
    const fp16* expert_states,
    const uint8_t* gate_ql, const fp16* gate_sc, const fp16* gate_mn,
    const uint8_t* up_ql, const fp16* up_sc, const fp16* up_mn,
    fp16* gate_buf, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int hidden_size, int intermediate_size,
    const int* tok_ids = nullptr)
{
    constexpr int BS = 32;          // Q4_K group size
    constexpr int MS = MS_;         // row-tiles of 16
    constexpr int MAX_M = MS * 16;
    constexpr int N = N_;           // N-cols per WI (32 default; 8/16 = more WIs/occupancy)
    constexpr int NS = N / 8;       // dpas 8-col steps per WI
    constexpr int ACC_SZ = MS * NS * 128;

    const int fused_im = 2 * intermediate_size;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, fused_im / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int nblocks = hidden_size / BS;       // H/32
            const int cid = (int)id[0];
            const int tid = (int)id[1];
            const int r_base = tid * N;

            const int eid = chunks[cid].eid;
            const int t0  = chunks[cid].t0;
            const int nt  = chunks[cid].nt;
            if (nt <= 0) return;

            // token row byte-offsets into expert_states (clamp to last valid token).
            // Lane m -> sorted-route position (t0+m), clamped to t0+nt-1.
            simd<uint32_t, MAX_M> pos =
                min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)t0,
                    simd<uint32_t, MAX_M>((uint32_t)(t0 + nt - 1)));
            simd<uint32_t, MAX_M> row;
            if (tok_ids != nullptr) {
                // Fold the input gather in: row = tok_ids[pos] (original token id),
                // so we read xf directly instead of a pre-gathered es. one int32
                // gather of MAX_M ids per chunk-tile (tiny vs the H-length loads).
                row = gather<uint32_t, MAX_M>(
                    reinterpret_cast<const uint32_t*>(tok_ids),
                    pos * (uint32_t)sizeof(int));
            } else {
                row = pos;   // legacy: caller pre-gathered, rows are contiguous
            }
            simd<uint32_t, MAX_M> in_off = row * (uint32_t)(hidden_size * sizeof(fp16));

            // r_base in [0,IM) -> gate matrix row r_base; [IM,2IM) -> up matrix row r_base-IM.
            const int IM = intermediate_size;
            const bool is_up = (r_base >= IM);
            const int local_row = is_up ? (r_base - IM) : r_base;
            const uint8_t* W_QL = is_up ? up_ql : gate_ql;
            const fp16*    W_SC = is_up ? up_sc : gate_sc;
            const fp16*    W_MN = is_up ? up_mn : gate_mn;
            const size_t row_base = (size_t)eid * IM + local_row;
            const uint8_t* w_ptr = W_QL + row_base * (hidden_size / 2);

            simd<float, ACC_SZ> acc(0.f);

            // strided gather of N scale/min values (one per N-col) at a given block
            const simd<uint32_t, N> scl_byte_off =
                simd<uint32_t, N>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
            const fp16* s_base = W_SC + row_base * nblocks;
            const fp16* m_base = W_MN + row_base * nblocks;

            // Each block = 32 K-elements = 16 bytes = BS/2. Inner k_base loop runs
            // 0..BS/2 step 8 -> 2 iters covering the 32-elem group; one scale+min/block.
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
                                reinterpret_cast<const uint32_t*>(expert_states),
                                in_off.template select<16, 1>(ms * 16).read() + k_off);
                    }

                    const int x_byte = blk * (BS / 2) + k_base;
                    auto w_raw_all = xesimd::lsc_load_2d<uint8_t, 8, N, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        w_ptr, (unsigned)(hidden_size/2-1), (unsigned)(N-1),
                        (unsigned)(hidden_size/2-1), x_byte, 0);

                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++) {
                        simd<uint8_t, 64> w_raw = w_raw_all.template select<64, 1>(ns * 64);
                        simd<uint8_t, 64> lo = w_raw & (uint8_t)0x0F;
                        simd<uint8_t, 64> hi = w_raw >> 4;
                        // Q4_K: nibble is UNSIGNED [0,15] (no -8). w = nibble*scale - min.
                        simd<fp16, 64> lo_val = convert<fp16>(lo);
                        simd<fp16, 64> hi_val = convert<fp16>(hi);
                        simd<fp16, 128> w_tile;
                        w_tile.template select<64, 2>(0) = lo_val;   // interleave low->2j
                        w_tile.template select<64, 2>(1) = hi_val;   // high->2j+1
                        // apply per-N-col scale and min: each row r (8 of them) is one N-col,
                        // 16 K-elements (this k_base sub-tile) share the block's scale/min.
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            fp16 ws = scl[ns*8+r];
                            fp16 wm = mnv[ns*8+r];
                            w_tile.template select<16, 1>(r*16) =
                                w_tile.template select<16, 1>(r*16) * ws - wm;
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
                    block_store<fp16, N>(gate_buf + (size_t)tok * fused_im + r_base,
                                         convert<fp16>(row_f));
                }
            }
        });
    });
}

// Default (prefill / large M): 64-token row-tile (MS=4) — unchanged behavior.
inline sycl::event moe_up_q4k_ggemv(
    sycl::queue& q, const fp16* expert_states,
    const uint8_t* gate_ql, const fp16* gate_sc, const fp16* gate_mn,
    const uint8_t* up_ql, const fp16* up_sc, const fp16* up_mn,
    fp16* gate_buf, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int hidden_size, int intermediate_size, const int* tok_ids = nullptr) {
    return moe_up_q4k_ggemv_t<4>(q, expert_states, gate_ql, gate_sc, gate_mn,
        up_ql, up_sc, up_mn, gate_buf, chunks, num_chunks, hidden_size,
        intermediate_size, tok_ids);
}

// ── TRANSPOSED small-M DPAS (M=output-cols, N=tokens): fills the 16-row DPAS M
// dim with REAL output columns (IM=512, always full) instead of padding tokens
// to 16. The token (N) dim pads 4->8 (2x) instead of 16 (4x). Weight is operand
// A (16 cols x K, real); activations operand B (K x 8 toks). Validate cos=1.0
// (moe_smallm_check) — if dims wrong, cos!=1 flags it. ──
template <int CT_>   // output-col tiles of 16 per WI (CT_*16 cols)
inline sycl::event moe_up_q4k_ggemv_tN(
    sycl::queue& q, const fp16* expert_states,
    const uint8_t* gate_ql, const fp16* gate_sc, const fp16* gate_mn,
    const uint8_t* up_ql, const fp16* up_sc, const fp16* up_mn,
    fp16* gate_buf, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int hidden_size, int intermediate_size, const int* tok_ids) {
    constexpr int BS = 32;
    constexpr int HALF = BS / 2;        // 16 packed bytes/group
    constexpr int MC = 8;               // DPAS _M = RepeatCount = output cols/tile (full)
    constexpr int KK = 16;              // DPAS _K (fp16: SD8*OpsPerCh2)
    constexpr int NTOK = 8;             // DPAS _N = ExecutionSize = token slots (pad 4->8)
    const int fused_im = 2 * intermediate_size;
    const int IM = intermediate_size;
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, fused_im / MC),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int cid = (int)id[0];
            const int ctile = (int)id[1];          // which 16-col tile
            const int col0 = ctile * MC;
            const int nt  = chunks[cid].nt;
            if (nt <= 0) return;
            const int eid = chunks[cid].eid;
            const int t0  = chunks[cid].t0;
            const int nblocks = hidden_size / BS;
            const int M = nt < NTOK ? nt : NTOK;

            // token original rows
            int rows[NTOK];
            for (int t = 0; t < M; ++t) {
                int sp = t0 + t;
                rows[t] = (tok_ids != nullptr) ? tok_ids[sp] : sp;
            }
            const bool is_up = (col0 >= IM);
            const int lrow0 = is_up ? (col0 - IM) : col0;
            const uint8_t* W_QL = is_up ? up_ql : gate_ql;
            const fp16*    W_SC = is_up ? up_sc : gate_sc;
            const fp16*    W_MN = is_up ? up_mn : gate_mn;

            // C[_M=MC cols, _N=NTOK toks] = A[_M,_K] * B[_K,_N]. acc = MC*NTOK = 64.
            // A layout (row-major _M x _K): wsub[r*KK + k]. B layout (_K x _N,
            // VNNI): xsub[k*NTOK + t]. dpas call order: dpas(C, B, A).
            simd<float, MC * NTOK> acc(0.f);

            for (int blk = 0; blk < nblocks; ++blk) {
                // one Q4_K group = 32 K = 2 DPAS K-steps of KK=16.
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const int k0 = sub * KK;                 // 0 or 16 within the group
                    // A: MC weight rows (cols), each KK K-vals (this sub-step)
                    simd<fp16, MC * KK> wsub;
                    #pragma unroll
                    for (int r = 0; r < MC; ++r) {
                        const size_t rb = (size_t)eid * IM + (lrow0 + r);
                        const uint8_t* wq = W_QL + rb * (hidden_size / 2) + blk * HALF;
                        float sc = (float)W_SC[rb * nblocks + blk];
                        float mn = (float)W_MN[rb * nblocks + blk];
                        simd<uint8_t, HALF> raw = block_load<uint8_t, HALF>(wq);
                        simd<uint16_t, HALF> w16 = convert<uint16_t>(raw);
                        simd<fp16, BS> wcol;   // 32 dequant vals (low->2j, high->2j+1)
                        wcol.template select<HALF, 2>(0) =
                            convert<fp16>(convert<float>(w16 & 0x000F) * sc - mn);
                        wcol.template select<HALF, 2>(1) =
                            convert<fp16>(convert<float>((w16 >> 4) & 0x000F) * sc - mn);
                        wsub.template select<KK, 1>(r * KK) = wcol.template select<KK, 1>(k0);
                    }
                    // B: activations, K-major [_K x _N]: xsub[k*NTOK + t]
                    simd<fp16, KK * NTOK> xsub(0.f);
                    #pragma unroll
                    for (int t = 0; t < NTOK; ++t) {
                        if (t >= M) break;
                        const fp16* xr = expert_states + (size_t)rows[t] * hidden_size + blk * BS + k0;
                        simd<fp16, KK> xk = block_load<fp16, KK>(xr);  // [KK] for token t
                        #pragma unroll
                        for (int k = 0; k < KK; ++k) xsub[k * NTOK + t] = xk[k];
                    }
                    acc = xmx_ns::dpas<8, 8, float, float, fp16, fp16>(acc, xsub, wsub);
                }
            }
            // C[_M=col, _N=tok] row-major: acc[col*NTOK + tok]
            #pragma unroll
            for (int r = 0; r < MC; ++r)
                for (int t = 0; t < M; ++t)
                    gate_buf[(size_t)(t0 + t) * fused_im + (col0 + r)] =
                        (fp16)acc[r * NTOK + t];
        });
    });
}

// Small-M (MTP verify, M<=16): 16-token row-tile (MS=1). Much less wasted DPAS.
// NOTE: a register-GEMV smallm (no 16-row DPAS pad) was tried (per-col and
// 8-col-tile) — both regressed vs this DPAS MS=1 (GRF can't
// hold N-col x M accumulators; scalar weight loads don't coalesce like the DPAS
// lsc_load_2d). The DPAS MS=1 (near the weight-BW floor) is the best small-M path
// here; beating it needs a different scheme (e.g. 4-expert x 4-token packing).
inline sycl::event moe_up_q4k_ggemv_smallm(
    sycl::queue& q, const fp16* expert_states,
    const uint8_t* gate_ql, const fp16* gate_sc, const fp16* gate_mn,
    const uint8_t* up_ql, const fp16* up_sc, const fp16* up_mn,
    fp16* gate_buf, const MoeQ4KChunkInfo* chunks, int num_chunks,
    int hidden_size, int intermediate_size, const int* tok_ids = nullptr) {
    // occupancy probe (notes §10..): at M=4 the N=32 tile launches only
    // num_chunks*(fused_im/32) WIs, causing occupancy starvation vs M=16,
    // not DPAS compute (MS1==MS4 wall).
    // Finer N-tiling multiplies WI count at fixed M (same bytes, no reduction).
    // MOE_N_TILE env selects 8/16/32 for A/B; default 16 (best probe headroom).
    static const int NT = [](){ const char* e=std::getenv("MOE_N_TILE"); return e?atoi(e):16; }();
    auto f = (NT==8)  ? moe_up_q4k_ggemv_t<1,8>
           : (NT==32) ? moe_up_q4k_ggemv_t<1,32>
                      : moe_up_q4k_ggemv_t<1,16>;
    return f(q, expert_states, gate_ql, gate_sc, gate_mn,
        up_ql, up_sc, up_mn, gate_buf, chunks, num_chunks, hidden_size,
        intermediate_size, tok_ids);
}
