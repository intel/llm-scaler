// ============================================================================
// MoE decode-only expert GEMV (M==1 fast path)
//
// Replaces the DPAS GEMM kernels for decode. Root cause fixed: DPAS path uses
// lsc_load_2d<uint8_t,16,16,1> (16B-wide 2D tiles, 1/4 of BMG's 64B cacheline),
// capping MoE decode at ~316 GB/s vs dense GEMV's ~574. Expert weight is plain
// row-major inside each expert (gate_up [E,2*inter,hidden], down [E,hidden,inter]),
// so 1D block_load<uint8_t,VL> along K reads contiguously like fp8_GEMV_bmg.
//
// One work-item computes one output element via VL-strided 1D loads + tail.
// KS=1 (no K-split) — first version focuses on the load-width fix.
// ============================================================================
#pragma once
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace esimd_math = sycl::ext::intel::esimd;

// Forward decl: defined in moe.sycl above the include point's use site.
template<int N>
SYCL_ESIMD_FUNCTION simd<sycl::half, N> fp8e4m3_to_half(simd<uint8_t, N> raw);

SYCL_ESIMD_FUNCTION simd<sycl::half, 256>
fp8e4m3_block_to_vnni(simd<uint8_t, 256> raw);

SYCL_ESIMD_FUNCTION inline simd<sycl::half, 256>
fp8g_load_wtile_native(const uint8_t* base, uint32_t N_total,
                       uint32_t K_total, uint32_t k0, uint32_t n0);

// Fast E4M3→half: uint16 bit-twiddle for normals + correct subnormal handling.
// e4m3 bias=7, fp16 bias=15 → exp_fp16 = exp_e4m3 + 8. mant 3b → fp16 mant top.
// Subnormal (e==0): value = mant * 2^-9 (representable as normal fp16).
template<int N>
SYCL_ESIMD_FUNCTION simd<float, N> fp8e4m3_dequant_fast(simd<uint8_t, N> raw) {
    using namespace sycl::ext::intel::esimd;
    simd<uint16_t, N> u = convert<uint16_t>(raw);
    simd<uint16_t, N> sign = (u >> 7) & 1;
    simd<uint16_t, N> e = (u >> 3) & 0xF;
    simd<uint16_t, N> m = u & 0x7;
    // Normal path
    simd<uint16_t, N> norm_bits = (sign << 15) | ((e + 8) << 10) | (m << 7);
    simd<fp16, N> hn = norm_bits.template bit_cast_view<fp16>();
    // Subnormal path: m * 2^-9, with sign
    simd<fp16, N> hs = convert<fp16>(m) * fp16(1.0f / 512.0f);
    simd<fp16, N> hs_signed = hs;
    hs_signed.merge(-hs, sign == 1);
    // Select subnormal where e==0
    simd<fp16, N> out = hn;
    out.merge(hs_signed, e == 0);
    return simd<float, N>(out);
}


// VL-wide full chunks plus 32-element tail chunks. KS=1.
// Up + gelu_tanh. gate_up_weight [E, 2*inter, hidden], K = hidden.
template<int VL>
struct MoeUpDecodeGeluTanh {
    const fp16*    x;
    const uint8_t* gate_up_weight;
    const float*   gate_up_scale;
    const int*     selected_experts;
    fp16*          intermediates;   // [top_k, inter]
    int hidden, inter, top_k, fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int route = (int)item.get_global_id(0);
        const int n     = (int)item.get_global_id(1);
        if (n >= inter) return;

        const int two_inter = 2 * inter;
        const int eid = selected_experts[route];
        const uint8_t* wbase = gate_up_weight + (size_t)eid * two_inter * hidden;
        const uint8_t* w_gate = wbase + (size_t)n * hidden;
        const uint8_t* w_up   = wbase + (size_t)(inter + n) * hidden;

        const int kp_full = (hidden / VL) * VL;
        simd<float, VL> g_acc(0.f), u_acc(0.f);
        for (int k = 0; k < kp_full; k += VL) {
            simd<fp16, VL> xv = block_load<fp16, VL>(x + k);
            simd<float, VL> xf = xv;
            g_acc += xf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(w_gate + k)));
            u_acc += xf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(w_up + k)));
        }
        float g_sum = reduce<float>(g_acc, std::plus<>());
        float u_sum = reduce<float>(u_acc, std::plus<>());
        simd<float, 32> g_tail(0.f), u_tail(0.f);
        for (int k = kp_full; k < hidden; k += 32) {
            simd<float, 32> xf = block_load<fp16, 32>(x + k);
            g_tail += xf * fp8e4m3_dequant_fast<32>(
                block_load<uint8_t, 32>(w_gate + k));
            u_tail += xf * fp8e4m3_dequant_fast<32>(
                block_load<uint8_t, 32>(w_up + k));
        }
        g_sum += reduce<float>(g_tail, std::plus<>());
        u_sum += reduce<float>(u_tail, std::plus<>());

        float scale = gate_up_scale[eid];
        float gs = g_sum * scale, us = u_sum * scale;
        constexpr float sqrt_2_over_pi = 0.7978845608f, coeff = 0.044715f;
        float gs3 = gs*gs*gs;
        float inner = sqrt_2_over_pi * (gs + coeff*gs3);
        float two_z = 2.0f * inner;
        if (two_z > 30.0f) two_z = 30.0f;
        if (two_z < -30.0f) two_z = -30.0f;
        float e2 = sycl::exp(two_z);
        float tanh_v = (e2 - 1.0f)/(e2 + 1.0f);
        float gelu = 0.5f*gs*(1.0f + tanh_v);
        intermediates[(size_t)route*inter + n] = fp16(gelu * us);
    }
};

// Down. down_weight [E, hidden, inter], K = inter.
template<int VL>
struct MoeDownDecode {
    const fp16*    intermediates;   // [top_k, inter]
    const uint8_t* down_weight;
    const float*   down_scale;
    const fp16*    routing_weights;
    const int*     selected_experts;
    fp16*          output;          // [top_k, hidden]
    int hidden, inter, top_k, fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int route = (int)item.get_global_id(0);
        const int h     = (int)item.get_global_id(1);
        if (h >= hidden) return;

        const int eid = selected_experts[route];
        const uint8_t* wrow = down_weight + (size_t)eid * hidden * inter + (size_t)h * inter;
        const fp16* hi = intermediates + (size_t)route * inter;

        const int kp_full = (inter / VL) * VL;
        simd<float, VL> acc(0.f);
        for (int k = 0; k < kp_full; k += VL) {
            simd<fp16, VL> hv = block_load<fp16, VL>(hi + k);
            simd<float, VL> hf = hv;
            acc += hf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(wrow + k)));
        }
        float s = reduce<float>(acc, std::plus<>());
        simd<float, 32> tail(0.f);
        for (int k = kp_full; k < inter; k += 32) {
            simd<float, 32> hf = block_load<fp16, 32>(hi + k);
            tail += hf * fp8e4m3_dequant_fast<32>(
                block_load<uint8_t, 32>(wrow + k));
        }
        s += reduce<float>(tail, std::plus<>());

        float w = (float)routing_weights[route];
        float ds = down_scale[eid];
        output[(size_t)route*hidden + h] = fp16(s * w * ds);
    }
};

// Native XPU weights are K-major: gate_up [E, hidden, 2*inter] and
// down [E, inter, hidden]. One work-item computes 16 output channels for one
// route. DPAS operates on an 8-row tile; decode fills row 0 and leaves the
// remaining rows zero.
struct MoeUpDecodeGeluTanhNative {
    const fp16*    x;
    const uint8_t* gate_up_weight;
    const float*   gate_up_scale;
    const int*     selected_experts;
    fp16*          intermediates;
    int hidden, inter, top_k;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        using namespace sycl::ext::intel::esimd::xmx;
        const int route = (int)item.get_global_id(0);
        const int n0 = (int)item.get_global_id(1) * 16;
        if (n0 >= inter) return;

        const int eid = selected_experts[route];
        const int two_inter = 2 * inter;
        const uint8_t* wbase = gate_up_weight
            + (size_t)eid * hidden * two_inter;
        const fp16 scale = fp16(gate_up_scale[eid]);
        simd<fp16, 128> gate_acc(fp16(0));
        simd<fp16, 128> up_acc(fp16(0));

        for (int k = 0; k < hidden; k += 16) {
            simd<fp16, 128> input_tile(fp16(0));
            input_tile.template select<16, 1>(0) =
                block_load<fp16, 16>(x + k);
            simd<fp16, 256> gate_tile = fp8g_load_wtile_native(
                wbase, two_inter, hidden, k, n0);
            simd<fp16, 256> up_tile = fp8g_load_wtile_native(
                wbase, two_inter, hidden, k, inter + n0);
            gate_acc = dpas<8, 8, fp16, fp16, fp16, fp16>(
                gate_acc, gate_tile * scale, input_tile);
            up_acc = dpas<8, 8, fp16, fp16, fp16, fp16>(
                up_acc, up_tile * scale, input_tile);
        }

        simd<float, 16> gate =
            simd<float, 16>(gate_acc.template select<16, 1>(0));
        simd<float, 16> up =
            simd<float, 16>(up_acc.template select<16, 1>(0));
        constexpr float c0 = 0.7978845608f;
        constexpr float c1 = 0.044715f;
        simd<float, 16> inner = c0 * (gate + c1 * gate * gate * gate);
        inner = min<float, 16>(
            max<float, 16>(inner, simd<float, 16>(-30.0f)),
            simd<float, 16>(30.0f));
        simd<float, 16> e2 =
            sycl::ext::intel::esimd::exp<float, 16>(2.0f * inner);
        simd<float, 16> gelu = 0.5f * gate
            * (1.0f + (e2 - 1.0f) / (e2 + 1.0f));
        block_store<fp16, 16>(
            intermediates + (size_t)route * inter + n0,
            convert<fp16>(gelu * up));
    }
};

struct MoeDownDecodeNative {
    const fp16*    intermediates;
    const uint8_t* down_weight;
    const float*   down_scale;
    const fp16*    routing_weights;
    const int*     selected_experts;
    fp16*          output;
    int hidden, inter, top_k;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        using namespace sycl::ext::intel::esimd::xmx;
        const int route = (int)item.get_global_id(0);
        const int n0 = (int)item.get_global_id(1) * 16;
        if (n0 >= hidden) return;

        const int eid = selected_experts[route];
        const uint8_t* wbase = down_weight
            + (size_t)eid * inter * hidden;
        const fp16* input = intermediates + (size_t)route * inter;
        const fp16 scale = fp16(down_scale[eid]);
        simd<fp16, 128> acc(fp16(0));

        for (int k = 0; k < inter; k += 16) {
            simd<fp16, 128> input_tile(fp16(0));
            input_tile.template select<16, 1>(0) =
                block_load<fp16, 16>(input + k);
            simd<fp16, 256> weight_tile = fp8g_load_wtile_native(
                wbase, hidden, inter, k, n0);
            acc = dpas<8, 8, fp16, fp16, fp16, fp16>(
                acc, weight_tile * scale, input_tile);
        }

        simd<float, 16> row =
            simd<float, 16>(acc.template select<16, 1>(0));
        row *= (float)routing_weights[route];
        block_store<fp16, 16>(
            output + (size_t)route * hidden + n0,
            convert<fp16>(row));
    }
};

// Native-layout decode variant using a 16-thread K reduction. The serial
// native kernel above performs the same number of DPAS operations per route,
// but one work-item walks the entire hidden/intermediate dimension. Matching
// the routed native kernel's local geometry exposes K parallelism and avoids
// the transposed dword load/repack in fp8g_load_wtile_native.
struct MoeUpDecodeGeluTanhNativeGrouped {
    const fp16*    x;
    const uint8_t* gate_up_weight;
    const float*   gate_up_scale;
    const int*     selected_experts;
    fp16*          intermediates;
    int hidden, inter, top_k;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        using namespace sycl::ext::intel::esimd::xmx;
        constexpr int GS = 16;
        slm_init<GS * 32 * sizeof(float)>();

        const int route = (int)item.get_group(0);
        const int n_tile = (int)item.get_group(1);
        const int tid = (int)item.get_local_id(1);
        const int n0 = n_tile * 16;
        if (n0 >= inter) return;

        const int eid = selected_experts[route];
        const int two_inter = 2 * inter;
        const uint8_t* base = gate_up_weight
            + (size_t)eid * hidden * two_inter;
        const int up_n0 = inter + n0;
        simd<float, 16> gate_acc(0.f), up_acc(0.f);

        for (int k = 16 * tid; k < hidden; k += 16 * GS) {
            simd<fp16, 16> a_tile = block_load<fp16, 16>(x + k);

            xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> gate_pay(
                base, (uint32_t)two_inter - 1u, (uint32_t)hidden - 1u,
                (uint32_t)two_inter - 1u, (uint32_t)n0, (uint32_t)k);
            auto gate_raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false, xesimd::cache_hint::cached,
                xesimd::cache_hint::cached>(gate_pay);
            gate_acc = dpas<8, 1, float, float, fp16, fp16>(
                gate_acc, fp8e4m3_block_to_vnni(gate_raw), a_tile);

            xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> up_pay(
                base, (uint32_t)two_inter - 1u, (uint32_t)hidden - 1u,
                (uint32_t)two_inter - 1u, (uint32_t)up_n0, (uint32_t)k);
            auto up_raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false, xesimd::cache_hint::cached,
                xesimd::cache_hint::cached>(up_pay);
            up_acc = dpas<8, 1, float, float, fp16, fp16>(
                up_acc, fp8e4m3_block_to_vnni(up_raw), a_tile);
        }

        const uint32_t slm_off = (uint32_t)(tid * 32) * 4u;
        slm_block_store<float, 16>(slm_off, gate_acc);
        slm_block_store<float, 16>(slm_off + 64u, up_acc);
        barrier();

        if (tid == 0) {
            simd<float, 16> gate(0.f), up(0.f);
            #pragma unroll
            for (int i = 0; i < GS; i++) {
                gate += slm_block_load<float, 16>((uint32_t)(i * 128));
                up += slm_block_load<float, 16>((uint32_t)(i * 128 + 64));
            }

            const float scale = gate_up_scale[eid];
            gate *= scale;
            up *= scale;
            constexpr float c0 = 0.7978845608f;
            constexpr float c1 = 0.044715f;
            simd<float, 16> inner = c0 * (gate + c1 * gate * gate * gate);
            simd<float, 16> two_z = 2.0f * inner;
            two_z.merge(simd<float, 16>(30.0f), two_z > 30.0f);
            two_z.merge(simd<float, 16>(-30.0f), two_z < -30.0f);
            simd<float, 16> exp2z = esimd_math::exp<float, 16>(two_z);
            simd<float, 16> tanh_v = (exp2z - 1.0f) / (exp2z + 1.0f);
            simd<float, 16> result = 0.5f * gate * (1.0f + tanh_v) * up;
            block_store<fp16, 16>(
                intermediates + (size_t)route * inter + n0,
                convert<fp16>(result));
        }
    }
};

struct MoeDownDecodeNativeGrouped {
    const fp16*    intermediates;
    const uint8_t* down_weight;
    const float*   down_scale;
    const fp16*    routing_weights;
    const int*     selected_experts;
    fp16*          output;
    int hidden, inter, top_k;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        using namespace sycl::ext::intel::esimd::xmx;
        constexpr int GS = 16;
        slm_init<GS * 32 * sizeof(float)>();

        const int route = (int)item.get_group(0);
        const int n_tile = (int)item.get_group(1);
        const int tid = (int)item.get_local_id(1);
        const int n0 = n_tile * 16;
        if (n0 >= hidden) return;

        const int eid = selected_experts[route];
        const uint8_t* base = down_weight
            + (size_t)eid * inter * hidden;
        const fp16* input = intermediates + (size_t)route * inter;
        simd<float, 16> acc(0.f);

        for (int k = 16 * tid; k < inter; k += 16 * GS) {
            simd<fp16, 16> a_tile = block_load<fp16, 16>(input + k);
            xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> pay(
                base, (uint32_t)hidden - 1u, (uint32_t)inter - 1u,
                (uint32_t)hidden - 1u, (uint32_t)n0, (uint32_t)k);
            auto raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1,
                false, false, xesimd::cache_hint::cached,
                xesimd::cache_hint::cached>(pay);
            acc = dpas<8, 1, float, float, fp16, fp16>(
                acc, fp8e4m3_block_to_vnni(raw), a_tile);
        }

        const uint32_t slm_off = (uint32_t)(tid * 16) * 4u;
        slm_block_store<float, 16>(slm_off, acc);
        barrier();

        if (tid == 0) {
            simd<float, 16> row(0.f);
            #pragma unroll
            for (int i = 0; i < GS; i++) {
                row += slm_block_load<float, 16>((uint32_t)(i * 64));
            }
            row *= (float)routing_weights[route] * down_scale[eid];
            block_store<fp16, 16>(
                output + (size_t)route * hidden + n0,
                convert<fp16>(row));
        }
    }
};


// ── per_expert_scale fold: topk_weight[r] *= scale[idx[r]] (top_k items) ─────
// gemma folds a learnable per-expert scale into the routing weights. One
// work-item per route. Tiny (top_k=8) — pure launch, but stays on-device.
struct MoeFoldExpertScale {
    fp16*        topk_weight;   // [top_k] in/out
    const int*   topk_idx;      // [top_k]
    const float* per_expert_scale;  // [n_experts]
    int top_k;
    void operator()(sycl::id<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int r = (int)it[0];
        if (r >= top_k) return;
        float s = per_expert_scale[topk_idx[r]];
        topk_weight[r] = fp16((float)topk_weight[r] * s);
    }
};
