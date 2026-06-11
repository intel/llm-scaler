/* moe_fp8_grouped.h — FP8 e4m3 expert-grouped MoE GEMM (DPAS, M-tiled).
 *
 * Replaces the per-route fp8 MoE (moe_forward_full_gelu_tanh_routed), which
 * does an M==1 DPAS per (token,expert) route and reloads weights for every
 * route — ~31.5ms/layer at M=256 vs Triton ~6.9ms. This version groups tokens
 * by expert (expert_offsets/expert_tokens, same gather as the int4 prefill
 * path) and runs an 8-row-M-tile DPAS GEMM so each loaded weight tile is
 * reused across 8 token rows.
 *
 * DPAS convention mirrors GEMM_fp8_pert_dpas_kernel (fp8_GEMM_pert.h):
 *   dpas<8,8,float,float,fp16,fp16>(acc[128], b_tile[256], a_tile[128])
 *     a_tile[128] = 8 rows × 16 K   (input)
 *     b_tile[256] = 16 N × 16 K in VNNI (weight)
 *     acc[128]    = 8 rows × 16 N   (acc.select<16,1>(m*16) = row m)
 *
 * Weight layout = unmodified vllm FusedMoE, K-major rows:
 *   gate_up [E, 2*inter, hidden] uint8 e4m3   (row = N (2*inter), contiguous K=hidden)
 *   down    [E, hidden, inter]   uint8 e4m3   (row = N (hidden), contiguous K=inter)
 * Per-tensor scales gate_up_scale[E], down_scale[E] (float).
 *
 * Reuses moe_int4_prefill_ops gather (moe_prefill_gather_forward_v2) and
 * accumulate (moe_prefill_accumulate_forward_v2) on the Python side; this
 * header provides only the two GEMM kernels + host wrappers.
 *
 * Requires (from moe.sycl, before include):
 *   using namespace sycl::ext::intel::esimd; ::xmx;
 *   fp8e4m3_to_half<N>(simd<uint8_t,N>) -> simd<fp16,N>
 *   submit_kernel(cgf, device, name)
 */
#pragma once

// Load one [16 N-rows × 16 K] e4m3 weight tile and return it VNNI-packed for
// DPAS. base points at weight[eid]; row stride = K_total (K-major). (n0, k0)
// is the top-left tile coord. Mirrors the per-route kernel's proven
// lsc_load_2d + fp8e4m3_block_to_vnni_nk path (no big register buffer, no
// indirect addressing — the previous K_CHUNK-preload version spilled a
// 4096-byte array that the BMG RA could not allocate).
SYCL_ESIMD_FUNCTION inline simd<fp16, 256>
fp8g_load_wtile(const uint8_t* base, uint32_t K_total, uint32_t n_rows_total,
                uint32_t k0, uint32_t n0) {
    xesimd::config_2d_mem_access<uint8_t, 16, 16, 1> pay(
        base, K_total - 1u, n_rows_total - 1u, K_total - 1u, k0, n0);
    auto raw = xesimd::lsc_load_2d<uint8_t, 16, 16, 1, false, false,
        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(pay);
    return fp8e4m3_block_to_vnni_nk(raw);
}

// ---- Up projection: gate_up GEMM + gelu_tanh(gate)*up -----------------------
// Grid: (num_experts, intermediate_size / 16). TILE_M=8 rows per m-tile.
// Output written to intermediate[routed_row, n] for each gathered route row.
template <int TILE_M = 8>
void moe_up_fp8_grouped_gelu_tanh_kernel(
    const fp16* x,                       // [n_tokens, hidden]
    const uint8_t* gate_up_weight,       // [E, 2*inter, hidden] e4m3
    const float* gate_up_scale,          // [E] per-tensor
    const int* expert_offsets,           // [E] exclusive prefix sum
    const int* expert_tokens,            // [total_routes] pair_idx sorted by expert
    fp16* intermediate,                  // [n_tokens*top_k, inter]
    int num_experts, int total_routes,
    int hidden_size, int intermediate_size, int top_k,
    const torch::Device& device) {
    const int n_ntiles = intermediate_size / 16;

    auto cgf = [&](sycl::handler& cgh) {
        cgh.parallel_for<class MoeUpFp8Grouped>(
            sycl::range<2>(num_experts, n_ntiles),
            [=](sycl::item<2> item) SYCL_ESIMD_KERNEL {
                const int eid    = (int)item.get_id(0);
                const int n_tile = (int)item.get_id(1);
                const int ng     = n_tile * 16;                 // gate N base
                const int nu     = intermediate_size + ng;      // up N base (row)

                const int t0 = expert_offsets[eid];
                const int t1 = (eid + 1 < num_experts) ? expert_offsets[eid + 1]
                                                       : total_routes;
                if (t0 >= t1) return;

                // K-major weight: row stride = hidden_size. base[(N row)*hidden + k]
                const uint8_t* wbase = gate_up_weight
                    + (size_t)eid * (size_t)(2 * intermediate_size) * hidden_size;
                const float scale = gate_up_scale[eid];

                for (int m_base = t0; m_base < t1; m_base += TILE_M) {
                    const int m_cnt = (t1 - m_base) < TILE_M ? (t1 - m_base) : TILE_M;

                    // Source token row (in x) and dest row (in intermediate) per m.
                    size_t x_off[TILE_M];
                    int dst_row[TILE_M];
                    #pragma unroll
                    for (int m = 0; m < TILE_M; m++) {
                        int s = m_base + m;
                        int pair = (s < t1) ? expert_tokens[s] : 0;
                        dst_row[m] = pair;                       // = token*top_k + k
                        x_off[m] = (size_t)(pair / top_k) * hidden_size;
                    }

                    // fp16 DPAS accumulators (XeLPG only supports fp16-acc
                    // dpas<8,8>; float-acc dpas2 is rejected at AOT for mtl-h).
                    // Matches the working int4 grouped GEMM convention.
                    simd<fp16, 128> g_acc(fp16(0));  // 8 rows × 16 N (gate)
                    simd<fp16, 128> u_acc(fp16(0));  // 8 rows × 16 N (up)

                    // Loop K in 16-wide sub-tiles; load weight + input per step.
                    for (int k = 0; k < hidden_size; k += 16) {
                        simd<fp16, 128> a_tile(fp16(0));
                        #pragma unroll
                        for (int m = 0; m < TILE_M; m++) {
                            if (m < m_cnt)
                                a_tile.template select<16, 1>(m * 16) =
                                    block_load<fp16, 16>(x + x_off[m] + k);
                        }
                        simd<fp16, 256> bg = fp8g_load_wtile(
                            wbase, (uint32_t)hidden_size,
                            (uint32_t)(2 * intermediate_size), (uint32_t)k, (uint32_t)ng);
                        simd<fp16, 256> bu = fp8g_load_wtile(
                            wbase, (uint32_t)hidden_size,
                            (uint32_t)(2 * intermediate_size), (uint32_t)k, (uint32_t)nu);
                        g_acc = dpas<8, 8, fp16, fp16, fp16, fp16>(g_acc, bg, a_tile);
                        u_acc = dpas<8, 8, fp16, fp16, fp16, fp16>(u_acc, bu, a_tile);
                    }

                    // gelu_tanh(gate*scale) * (up*scale) in float for stability.
                    simd<float, 128> gf = simd<float, 128>(g_acc) * scale;
                    simd<float, 128> uf = simd<float, 128>(u_acc) * scale;
                    constexpr float c0 = 0.7978845608f;  // sqrt(2/pi)
                    constexpr float c1 = 0.044715f;
                    simd<float, 128> g3 = gf * gf * gf;
                    simd<float, 128> inner = c0 * (gf + c1 * g3);
                    simd<float, 128> e2 = esimd_math::exp<float, 128>(2.0f * inner);
                    simd<float, 128> tanh_v = (e2 - 1.0f) / (e2 + 1.0f);
                    simd<float, 128> res = (0.5f * gf * (1.0f + tanh_v)) * uf;

                    #pragma unroll
                    for (int m = 0; m < TILE_M; m++) {
                        if (m < m_cnt) {
                            simd<float, 16> rf = res.template select<16, 1>(m * 16);
                            simd<fp16, 16> row = convert<fp16>(rf);
                            block_store<fp16, 16>(
                                intermediate + (size_t)dst_row[m] * intermediate_size + ng,
                                row);
                        }
                    }
                }
            });
    };
    submit_kernel(cgf, device, "moe up fp8 grouped gelu_tanh");
}

// ---- Down projection: down GEMM, apply routing weight, accumulate -----------
// Grid: (num_experts, hidden_size / 16). TILE_M=8.
// Reads intermediate[routed_row, :inter], writes output[routed_row, :hidden]
// (per-route output rows; Python accumulate folds top_k rows back per token).
template <int TILE_M = 8>
void moe_down_fp8_grouped_kernel(
    const fp16* intermediate,            // [n_tokens*top_k, inter]
    const uint8_t* down_weight,          // [E, hidden, inter] e4m3
    const float* down_scale,             // [E] per-tensor
    const fp16* routing_weights,         // [n_tokens*top_k] flat (pair-indexed)
    const int* expert_offsets,
    const int* expert_tokens,
    fp16* output,                        // [n_tokens*top_k, hidden]
    int num_experts, int total_routes,
    int hidden_size, int intermediate_size, int top_k,
    const torch::Device& device) {
    const int n_ntiles = hidden_size / 16;

    auto cgf = [&](sycl::handler& cgh) {
        cgh.parallel_for<class MoeDownFp8Grouped>(
            sycl::range<2>(num_experts, n_ntiles),
            [=](sycl::item<2> item) SYCL_ESIMD_KERNEL {
                const int eid    = (int)item.get_id(0);
                const int n_tile = (int)item.get_id(1);
                const int ng     = n_tile * 16;                 // hidden N base

                const int t0 = expert_offsets[eid];
                const int t1 = (eid + 1 < num_experts) ? expert_offsets[eid + 1]
                                                       : total_routes;
                if (t0 >= t1) return;

                // down weight K-major: [E, hidden, inter], row stride = inter.
                const uint8_t* wbase = down_weight
                    + (size_t)eid * (size_t)hidden_size * intermediate_size;
                const float scale = down_scale[eid];

                for (int m_base = t0; m_base < t1; m_base += TILE_M) {
                    const int m_cnt = (t1 - m_base) < TILE_M ? (t1 - m_base) : TILE_M;
                    size_t in_off[TILE_M];
                    int dst_row[TILE_M];
                    fp16 rw[TILE_M];
                    #pragma unroll
                    for (int m = 0; m < TILE_M; m++) {
                        int s = m_base + m;
                        int pair = (s < t1) ? expert_tokens[s] : 0;
                        dst_row[m] = pair;
                        in_off[m] = (size_t)pair * intermediate_size;
                        rw[m] = (s < t1) ? routing_weights[pair] : fp16(0);
                    }

                    simd<fp16, 128> acc(fp16(0));  // 8 rows × 16 N (hidden)

                    for (int k = 0; k < intermediate_size; k += 16) {
                        simd<fp16, 128> a_tile(fp16(0));
                        #pragma unroll
                        for (int m = 0; m < TILE_M; m++) {
                            if (m < m_cnt)
                                a_tile.template select<16, 1>(m * 16) =
                                    block_load<fp16, 16>(intermediate + in_off[m] + k);
                        }
                        simd<fp16, 256> bd = fp8g_load_wtile(
                            wbase, (uint32_t)intermediate_size,
                            (uint32_t)hidden_size, (uint32_t)k, (uint32_t)ng);
                        acc = dpas<8, 8, fp16, fp16, fp16, fp16>(acc, bd, a_tile);
                    }

                    simd<float, 128> accf = simd<float, 128>(acc) * scale;
                    #pragma unroll
                    for (int m = 0; m < TILE_M; m++) {
                        if (m < m_cnt) {
                            simd<float, 16> r = accf.template select<16, 1>(m * 16);
                            r = r * (float)rw[m];
                            simd<fp16, 16> orow = convert<fp16>(r);
                            block_store<fp16, 16>(
                                output + (size_t)dst_row[m] * hidden_size + ng, orow);
                        }
                    }
                }
            });
    };
    submit_kernel(cgf, device, "moe down fp8 grouped");
}

// ---- Host wrappers ----------------------------------------------------------
// Gather (expert_offsets/expert_tokens) is computed in Python via the existing
// moe_int4_prefill_ops.moe_prefill_gather_forward_v2 (weight-dtype agnostic).
// These two ops are the fp8 grouped GEMMs.

// Up: returns intermediate [n_tokens*top_k, inter] (gelu_tanh(gate)*up).
inline torch::Tensor moe_up_fp8_grouped(
    torch::Tensor x,                 // [n_tokens, hidden] fp16
    torch::Tensor gate_up_weight,    // [E, 2*inter, hidden] e4m3 (uint8 view)
    torch::Tensor gate_up_scale,     // [E] float
    torch::Tensor expert_offsets,    // [E] int32
    torch::Tensor expert_tokens,     // [total_routes] int32
    int64_t top_k, int64_t n_routed_experts) {
    TORCH_CHECK(x.scalar_type() == torch::kHalf);
    int n_tokens = x.size(0);
    int hidden_size = x.size(1);
    int intermediate_size = gate_up_weight.size(1) / 2;
    int total_routes = n_tokens * (int)top_k;
    auto intermediate = torch::empty({total_routes, intermediate_size},
        torch::device(x.device()).dtype(torch::kHalf));
    moe_up_fp8_grouped_gelu_tanh_kernel<>(
        (const fp16*)x.data_ptr(),
        (const uint8_t*)gate_up_weight.data_ptr(),
        gate_up_scale.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        expert_tokens.data_ptr<int>(),
        (fp16*)intermediate.data_ptr(),
        (int)n_routed_experts, total_routes,
        hidden_size, intermediate_size, (int)top_k, x.device());
    return intermediate;
}

// Down: returns per-route output [n_tokens*top_k, hidden] (routing-weighted).
// Python then sums the top_k rows per token (moe_prefill_accumulate or a view-sum).
inline torch::Tensor moe_down_fp8_grouped(
    torch::Tensor intermediate,      // [n_tokens*top_k, inter] fp16
    torch::Tensor down_weight,       // [E, hidden, inter] e4m3 (uint8 view)
    torch::Tensor down_scale,        // [E] float
    torch::Tensor routing_weights,   // [n_tokens*top_k] fp16 (pair-indexed flat)
    torch::Tensor expert_offsets,
    torch::Tensor expert_tokens,
    int64_t top_k, int64_t n_routed_experts) {
    TORCH_CHECK(intermediate.scalar_type() == torch::kHalf);
    int total_routes = intermediate.size(0);
    int intermediate_size = intermediate.size(1);
    int hidden_size = down_weight.size(1);
    auto output = torch::empty({total_routes, hidden_size},
        torch::device(intermediate.device()).dtype(torch::kHalf));
    moe_down_fp8_grouped_kernel<>(
        (const fp16*)intermediate.data_ptr(),
        (const uint8_t*)down_weight.data_ptr(),
        down_scale.data_ptr<float>(),
        (const fp16*)routing_weights.data_ptr(),
        expert_offsets.data_ptr<int>(),
        expert_tokens.data_ptr<int>(),
        (fp16*)output.data_ptr(),
        (int)n_routed_experts, total_routes,
        hidden_size, intermediate_size, (int)top_k, intermediate.device());
    return output;
}
