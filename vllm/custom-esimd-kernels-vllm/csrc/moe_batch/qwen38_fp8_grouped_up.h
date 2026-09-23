/* Qwen3.8 FP8 W8A16 grouped-UP candidate.
 *
 * This header is intentionally standalone: include it from
 * qwen38_fp8_moe.sycl inside its existing anonymous namespace, after
 * decode/vnni/load_tile.  It uses the same physical Qwen3.8 layout as the
 * native per-expert UP implementation:
 *
 *   routed w: [E, H, 2*I], FP8 e4m3, with row-major H x (2*I) storage
 *   shared w: [2*I, H], FP8 e4m3, native row-major storage
 *   scale:   one FP32 scalar per routed expert/shared UP matrix
 *   x/inter: FP16 [M, H]/[M, (K+1), I]
 *
 * The public function below deliberately has the exact ABI of up<I>().
 * It is a candidate only; the caller must opt into it explicitly.
 */
#pragma once

template <int I>
class Q38Fp8GroupedUp;

// Keep the old UP epilogue's rounding points visible.  In particular, the
// gate and up accumulators are rounded to FP16 after applying the FP32 scalar
// scale, then SiLU(gate) * up is rounded once more when stored.
SYCL_ESIMD_FUNCTION inline void qwen38_fp8_grouped_up_store(
    simd<float, 16> gate_acc,
    simd<float, 16> up_acc,
    float scale,
    fp16* dst) {
    gate_acc = convert<float>(convert<fp16>(gate_acc * scale));
    up_acc = convert<float>(convert<fp16>(up_acc * scale));
    simd<float, 16> value =
        (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
    block_store<fp16, 16>(dst, convert<fp16>(value));
}

/*
 * Pair adjacent input tokens without changing the [M, R, I/16] work-group
 * shape.  Each group still has the current implementation's 16 lanes, with
 * one lane responsible for each 16-wide K tile.
 *
 * For an even token t, the work-group is a leader.  A routed leader searches the
 * next token's K routes for the same expert.  On a hit, one FP8 gate tile and
 * one FP8 up tile are decoded/read and used by two independent DPAS
 * accumulators.  The odd token's matching item returns, so its output is
 * produced exactly once.  If there is no hit, both rows retain independent
 * work-items.  The same pairing is unconditional for the single shared
 * expert row.  A final odd token remains a normal one-row computation.
 *
 * The top-k producer must emit unique expert IDs per token, as the current
 * top-k contract does.  Duplicate IDs are outside this candidate's contract.
 */
template <int I>
void grouped_up(sycl::queue& q, const fp16* x, const uint8_t* w,
                const float* scale, const uint8_t* shared, const float* ss,
                const int* ids, fp16* inter, int m) {
    constexpr int GS = 16;
    constexpr int N_TILES = I / 16;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class Q38Fp8GroupedUp<I>>(
            sycl::nd_range<2>({size_t(m * R), size_t(N_TILES * GS)},
                               {1, GS}),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
            // Four FP32 partials per lane (gate/up for each possible row)
            // need 16 * 256 bytes.  This mirrors the old two-accumulator
            // SLM reduction while retaining the original K split.
            slm_init<GS * 64 * sizeof(float)>();
            const int row = (int)item.get_group(0);
            const int token = row / R;
            const int route = row % R;
            const int n = (int)item.get_group(1) * 16;
            const int tid = (int)item.get_local_id(1);

            int row0 = -1;
            int row1 = -1;
            int route0 = -1;
            int route1 = -1;
            int expert = 0;
            const bool is_shared = route == K;

            if (is_shared) {
                // The shared expert has the same weights for every token, so
                // every even/odd adjacent pair can reuse both FP8 tiles.
                if (token & 1) return;
                row0 = token;
                route0 = K;
                if (token + 1 < m) {
                    row1 = token + 1;
                    route1 = K;
                }
            } else if ((token & 1) == 0) {
                row0 = token;
                route0 = route;
                expert = ids[token * K + route];

                if (token + 1 < m) {
                    for (int next_route = 0; next_route < K; ++next_route) {
                        if (ids[(token + 1) * K + next_route] == expert) {
                            row1 = token + 1;
                            route1 = next_route;
                            break;
                        }
                    }
                }
            } else {
                // If the previous even token owns this expert, its leader
                // already wrote this row and route.  Otherwise this route is
                // an independent one-row computation.
                const int previous = token - 1;
                expert = ids[token * K + route];
                bool paired = false;
                for (int previous_route = 0; previous_route < K;
                     ++previous_route) {
                    if (ids[previous * K + previous_route] == expert) {
                        paired = true;
                        break;
                    }
                }
                if (paired) return;
                row0 = token;
                route0 = route;
            }

            const uint8_t* base = is_shared
                ? shared
                : w + size_t(expert) * H * 2 * I;
            simd<float, 16> gate0(0.0f), up0(0.0f);
            simd<float, 16> gate1(0.0f), up1(0.0f);

            // The original UP assigns ten 16-wide K tiles to each of the 16
            // lanes for H=2560.  Keep that split; grouping only changes how
            // the same loaded weight tile feeds the second row.
            for (int k = tid * 16; k < H; k += GS * 16) {
                // The tile is loaded and FP8-decoded once, then consumed by
                // both rows when the adjacent pair shares an expert.
                auto gate = is_shared
                    ? load_tile<true>(base, H, 2 * I, n, k)
                    : load_tile<false>(base, 2 * I, H, n, k);
                auto up = is_shared
                    ? load_tile<true>(base, H, 2 * I, n + I, k)
                    : load_tile<false>(base, 2 * I, H, n + I, k);

                auto a = block_load<fp16, 16>(x + size_t(row0) * H + k);
                gate0 = dpas<8, 1, float, float, fp16, fp16>(gate0, gate, a);
                up0 = dpas<8, 1, float, float, fp16, fp16>(up0, up, a);
                if (row1 >= 0) {
                    auto a = block_load<fp16, 16>(
                        x + size_t(row1) * H + k);
                    gate1 = dpas<8, 1, float, float, fp16, fp16>(
                        gate1, gate, a);
                    up1 = dpas<8, 1, float, float, fp16, fp16>(
                        up1, up, a);
                }
            }

            const int slm_base = tid * 128;
            slm_block_store<float, 16>(slm_base, gate0);
            slm_block_store<float, 16>(slm_base + 64, up0);
            if (row1 >= 0) {
                slm_block_store<float, 16>(GS*128 + slm_base, gate1);
                slm_block_store<float, 16>(GS*128 + slm_base + 64, up1);
            }
            barrier();

            if (tid == 0) {
                gate0 = 0.0f;
                up0 = 0.0f;
                gate1 = 0.0f;
                up1 = 0.0f;
                #pragma unroll
                for (int lane = 0; lane < GS; ++lane) {
                    const int base = lane * 128;
                    gate0 += slm_block_load<float, 16>(base);
                    up0 += slm_block_load<float, 16>(base + 64);
                    if (row1 >= 0) {
                        gate1 += slm_block_load<float, 16>(GS*128 + base);
                        up1 += slm_block_load<float, 16>(GS*128 + base + 64);
                    }
                }

                const float up_scale = is_shared ? ss[0] : scale[expert];
                qwen38_fp8_grouped_up_store(
                    gate0, up0, up_scale,
                    inter + (size_t)(row0 * R + route0) * I + n);
                if (row1 >= 0) {
                    qwen38_fp8_grouped_up_store(
                        gate1, up1, up_scale,
                        inter + (size_t)(row1 * R + route1) * I + n);
                }
            }
            });
    });
}
