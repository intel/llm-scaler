#pragma once
// ============================================================================
// INT4 N-major MoE Grouped GEMM — DPAS-based, adapted from moe_prefill_int4.sycl
//
// Weight: [E, N, K/8] int32 (N-major, implement_zp two's complement nibbles)
//         OR equivalently [E, N, K/2] uint8
// Scale:  [E, N, K/GS] fp16 (N-major, per-group)
// Input:  [total_tokens, K] fp16
// Output: [total_tokens, N] fp16
//
// Compared to prefill kernel (K-major weight [E, K/8, N]):
//   - a_tile construction reads along K (contiguous) for each N row
//   - b_tile (input) construction is identical
//   - DPAS, scale grouping, SiLU, scatter all identical
//
// Grid: range<2>(num_experts, N / N_TILE)
// Each WI handles one expert × one N-tile (N_TILE=16 for up, 32 for down)
// M dimension: loop over M_TILE=32 chunks
// ============================================================================

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
namespace xmx_ns = sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// ============================================================================
// Gate+Up+SiLU kernel: N-major INT4 weight
//
// Template params match prefill kernel:
//   IT = fp16, BS = group_size (128), MAX_M = M-tile (32), N = N-tile (16)
//
// Weight [E, 2*I, K/8] int32 N-major (K contiguous within each N row)
// Scale  [E, 2*I, K/GS] fp16 N-major
// ============================================================================
template<typename IT, int BS, int MAX_M = 32, int N = 16>
sycl::event moe_up_int4_nmajor_kernel(
    sycl::queue& q,
    const IT* input, const uint32_t* gate_up_qweight, const IT* gate_up_scale,
    IT* intermediate, const int* expert_offsets, const int* expert_tokens,
    int num_experts, int total_seqlen, int hidden_size, int intermediate_size, int top_k)
{
    static_assert(MAX_M % 16 == 0);
    static_assert(N == 16);
    static_assert(BS % 16 == 0);
    constexpr int MS      = MAX_M / 16;
    constexpr int NS      = N / 8;
    constexpr int ACC_SZ  = MS * NS * 128;
    constexpr int KP_PER_GROUP = BS / 8;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_experts, intermediate_size / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int K_packed = hidden_size / 8;
            const int K_groups = hidden_size / BS;
            const int two_inter = 2 * intermediate_size;
            const int eid  = (int)id[0];
            const int tid  = (int)id[1];
            const int n_start = tid * N;
            const int t0 = expert_offsets[eid];
            const int t1 = (eid + 1 < num_experts) ? expert_offsets[eid + 1] : total_seqlen;
            if (t0 == t1) return;

            // N-major: weight [E, 2*I, K_packed], stride per N-row = K_packed
            const uint32_t* w_base = gate_up_qweight + (size_t)eid * two_inter * K_packed;
            const IT* s_base       = gate_up_scale   + (size_t)eid * two_inter * K_groups;

            for (int m_base = t0; m_base < t1; m_base += MAX_M) {
                simd<uint32_t, MAX_M> sorted_idxs =
                    min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)m_base,
                        simd<uint32_t, MAX_M>((uint32_t)(t1 - 1)));
                simd<uint32_t, MAX_M> pair_idxs =
                    convert<uint32_t>(gather<int, MAX_M>(expert_tokens, sorted_idxs * 4u));
                simd<uint32_t, MAX_M> in_off =
                    (pair_idxs / (uint32_t)top_k) * (uint32_t)(hidden_size * sizeof(IT));

                simd<fp16, ACC_SZ> g_acc(fp16(0));
                simd<fp16, ACC_SZ> u_acc(fp16(0));

                for (int kg = 0; kg < K_groups; kg++) {
                    // N-major scale: s_base[n_row * K_groups + kg]
                    simd<fp16, N> gate_scl, up_scl;
                    #pragma unroll
                    for (int ni = 0; ni < N; ni++) {
                        gate_scl[ni] = s_base[(n_start + ni) * K_groups + kg];
                        up_scl[ni]   = s_base[(intermediate_size + n_start + ni) * K_groups + kg];
                    }

                    for (int kp_off = 0; kp_off < KP_PER_GROUP; kp_off += 2) {
                        const int kp_start = kg * KP_PER_GROUP + kp_off;

                        // b_tile: gather input [MS*16, 16K] — same as prefill
                        const uint32_t k_off = (uint32_t)(kp_start * 8) * (uint32_t)sizeof(IT);
                        simd<fp16, MS * 256> b_tile;
                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            simd<IT, 256> btmp;
                            btmp.template bit_cast_view<uint32_t>() =
                                xesimd::lsc_gather<uint32_t, 8,
                                    xesimd::lsc_data_size::u32,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                                    16, uint32_t>(
                                    reinterpret_cast<const uint32_t*>(input),
                                    in_off.template select<16, 1>(ms * 16).read() + k_off);
                            b_tile.template select<256, 1>(ms * 256) = convert<fp16>(btmp);
                        }

                        // a_tile construction: N-major weight
                        // For each NS (N sub-tile of 8), load 2 kp rows × 8 N cols
                        // N-major: w_base[(n_col) * K_packed + kp]
                        auto dequant_dpas = [&](int col_base, const simd<fp16, N>& scl,
                                                simd<fp16, ACC_SZ>& acc) SYCL_ESIMD_FUNCTION {
                            #pragma unroll
                            for (int ns = 0; ns < NS; ns++) {
                                simd<fp16, 128> a_tile;
                                #pragma unroll
                                for (int r = 0; r < 8; r++) {
                                    const int n_col = col_base + ns * 8 + r;
                                    // N-major: read 2 consecutive kp values from row n_col
                                    const uint32_t w0r = w_base[(size_t)n_col * K_packed + kp_start];
                                    const uint32_t w1r = w_base[(size_t)n_col * K_packed + kp_start + 1];
                                    const fp16 s = scl[ns * 8 + r];
                                    simd<fp16, 16> row;
                                    // No unshuffle needed: natural nibble order
                                    // implement_zp two's complement dequant
                                    #pragma unroll
                                    for (int k = 0; k < 8; k++) {
                                        uint32_t n0 = (w0r >> (k * 4)) & 0xFu;
                                        uint32_t n1 = (w1r >> (k * 4)) & 0xFu;
                                        row[k]     = fp16((int)(n0 >= 8u ? n0 - 16u : n0));
                                        row[8 + k] = fp16((int)(n1 >= 8u ? n1 - 16u : n1));
                                    }
                                    row *= s;
                                    a_tile.template select<16, 1>(r * 16) = row;
                                }

                                #pragma unroll
                                for (int ms = 0; ms < MS; ms++) {
                                    const int idx = ms * NS * 128 + ns * 128;
                                    simd<fp16, 128> a  = acc.template select<128, 1>(idx);
                                    simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                                    a = xmx_ns::dpas<8, 8, fp16, fp16, fp16, fp16>(a, bv, a_tile);
                                    acc.template select<128, 1>(idx) = a;
                                }
                            }
                        };

                        dequant_dpas(n_start,                    gate_scl, g_acc);
                        dequant_dpas(n_start + intermediate_size, up_scl,   u_acc);
                    }
                }

                simd<fp16, ACC_SZ> sv =
                    (g_acc / (fp16(1) + sycl::ext::intel::esimd::exp(-g_acc))) * u_acc;

                simd<uint32_t, MAX_M> out_off =
                    pair_idxs * (uint32_t)(intermediate_size * sizeof(IT));

                #pragma unroll
                for (int ms = 0; ms < MS; ms++) {
                    simd<uint32_t, 16> ms_off = out_off.template select<16, 1>(ms * 16).read();
                    #pragma unroll
                    for (int r = 0; r < N; r++) {
                        uint32_t ch = (uint32_t)(n_start + r) * (uint32_t)sizeof(IT);
                        simd<IT, 16> val = sv.template select<16, 1>(ms * N * 16 + r * 16);
                        scatter<IT, 16>(intermediate, ms_off + ch, val);
                    }
                }
            }
        });
    });
}

// ============================================================================
// Down projection kernel: N-major INT4 weight
// Weight [E, H, I/8] int32 N-major
// Scale  [E, H, I/GS] fp16 N-major
// ============================================================================
template<typename IT, int BS, int MAX_M = 32, int N = 32>
sycl::event moe_down_int4_nmajor_kernel(
    sycl::queue& q,
    const IT* intermediate, const uint32_t* down_qweight, const IT* down_scale,
    IT* output, const int* expert_offsets, const int* expert_tokens,
    const fp16* routing_weights,
    int num_experts, int total_seqlen, int hidden_size, int intermediate_size, int top_k)
{
    static_assert(MAX_M % 16 == 0);
    static_assert(N == 32);
    static_assert(BS % 16 == 0);
    constexpr int MS      = MAX_M / 16;
    constexpr int NS      = N / 8;
    constexpr int ACC_SZ  = MS * NS * 128;
    constexpr int KP_PER_GROUP = BS / 8;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_experts, hidden_size / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int I_packed = intermediate_size / 8;
            const int I_groups = intermediate_size / BS;
            const int eid  = (int)id[0];
            const int tid  = (int)id[1];
            const int n_start = tid * N;
            const int t0 = expert_offsets[eid];
            const int t1 = (eid + 1 < num_experts) ? expert_offsets[eid + 1] : total_seqlen;
            if (t0 == t1) return;

            // N-major: [E, H, I_packed]
            const uint32_t* w_base = down_qweight + (size_t)eid * hidden_size * I_packed;
            const IT* s_base       = down_scale   + (size_t)eid * hidden_size * I_groups;

            for (int m_base = t0; m_base < t1; m_base += MAX_M) {
                simd<uint32_t, MAX_M> sorted_idxs =
                    min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)m_base,
                        simd<uint32_t, MAX_M>((uint32_t)(t1 - 1)));
                simd<uint32_t, MAX_M> pair_idxs =
                    convert<uint32_t>(gather<int, MAX_M>(expert_tokens, sorted_idxs * 4u));
                simd<uint32_t, MAX_M> in_off =
                    pair_idxs * (uint32_t)(intermediate_size * sizeof(IT));

                // Routing weights for each pair
                simd<fp16, MAX_M> rw;
                #pragma unroll
                for (int mi = 0; mi < MAX_M; mi++) {
                    uint32_t si = min(sorted_idxs[mi], (uint32_t)(t1 - 1));
                    rw[mi] = si < (uint32_t)total_seqlen ? routing_weights[si] : fp16(0);
                }

                simd<fp16, ACC_SZ> acc(fp16(0));

                for (int kg = 0; kg < I_groups; kg++) {
                    simd<fp16, N> scl;
                    #pragma unroll
                    for (int ni = 0; ni < N; ni++) {
                        scl[ni] = s_base[(n_start + ni) * I_groups + kg];
                    }

                    for (int kp_off = 0; kp_off < KP_PER_GROUP; kp_off += 2) {
                        const int kp_start = kg * KP_PER_GROUP + kp_off;

                        const uint32_t k_off = (uint32_t)(kp_start * 8) * (uint32_t)sizeof(IT);
                        simd<fp16, MS * 256> b_tile;
                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            simd<IT, 256> btmp;
                            btmp.template bit_cast_view<uint32_t>() =
                                xesimd::lsc_gather<uint32_t, 8,
                                    xesimd::lsc_data_size::u32,
                                    xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                                    16, uint32_t>(
                                    reinterpret_cast<const uint32_t*>(intermediate),
                                    in_off.template select<16, 1>(ms * 16).read() + k_off);
                            b_tile.template select<256, 1>(ms * 256) = convert<fp16>(btmp);
                        }

                        #pragma unroll
                        for (int ns = 0; ns < NS; ns++) {
                            simd<fp16, 128> a_tile;
                            #pragma unroll
                            for (int r = 0; r < 8; r++) {
                                const int n_col = n_start + ns * 8 + r;
                                const uint32_t w0r = w_base[(size_t)n_col * I_packed + kp_start];
                                const uint32_t w1r = w_base[(size_t)n_col * I_packed + kp_start + 1];
                                const fp16 s = scl[ns * 8 + r];
                                simd<fp16, 16> row;
                                #pragma unroll
                                for (int k = 0; k < 8; k++) {
                                    uint32_t n0 = (w0r >> (k * 4)) & 0xFu;
                                    uint32_t n1 = (w1r >> (k * 4)) & 0xFu;
                                    row[k]     = fp16((int)(n0 >= 8u ? n0 - 16u : n0));
                                    row[8 + k] = fp16((int)(n1 >= 8u ? n1 - 16u : n1));
                                }
                                row *= s;
                                a_tile.template select<16, 1>(r * 16) = row;
                            }

                            #pragma unroll
                            for (int ms = 0; ms < MS; ms++) {
                                const int idx = ms * NS * 128 + ns * 128;
                                simd<fp16, 128> a = acc.template select<128, 1>(idx);
                                simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                                a = xmx_ns::dpas<8, 8, fp16, fp16, fp16, fp16>(a, bv, a_tile);
                                acc.template select<128, 1>(idx) = a;
                            }
                        }
                    }
                }

                // Apply routing weight and scatter output
                simd<uint32_t, MAX_M> out_off_bytes =
                    (pair_idxs / (uint32_t)top_k) * (uint32_t)(hidden_size * sizeof(IT));

                #pragma unroll
                for (int ms = 0; ms < MS; ms++) {
                    simd<uint32_t, 16> ms_off = out_off_bytes.template select<16, 1>(ms * 16).read();
                    simd<fp16, 16> ms_rw = rw.template select<16, 1>(ms * 16);
                    #pragma unroll
                    for (int r = 0; r < N; r++) {
                        uint32_t ch = (uint32_t)(n_start + r) * (uint32_t)sizeof(IT);
                        simd<IT, 16> val = acc.template select<16, 1>(ms * N * 16 + r * 16);
                        val *= ms_rw;
                        // Atomic add for accumulate across experts
                        simd<IT, 16> old;
                        old.template bit_cast_view<uint32_t>() =
                            xesimd::lsc_gather<uint32_t, 1,
                                xesimd::lsc_data_size::u16,
                                xesimd::cache_hint::uncached, xesimd::cache_hint::uncached,
                                16, uint32_t>(
                                reinterpret_cast<const uint32_t*>(output),
                                ms_off + ch);
                        scatter<IT, 16>(output, ms_off + ch, old + val);
                    }
                }
            }
        });
    });
}
