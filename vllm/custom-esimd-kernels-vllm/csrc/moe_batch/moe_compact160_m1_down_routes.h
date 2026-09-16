// SPDX-License-Identifier: Apache-2.0
// TP4 M1: route-parallel DOWN, one workgroup per output tile. All route
// dots stay FP32; the owner adds them in the original route order, including
// the separate shared-expert tail. Scratch lives only in this workgroup.
#pragma once

template <int Tile, bool VectorReduce> class MoeCompact160M1DownRoutes;

template <int Tile, bool VectorReduce = false>
static void compact160_m1_down_routes(
    const fp16* routed, const uint8_t* weights, const fp16* scales,
    const fp16* route_weight, const int32_t* ids,
    const fp16* shared, const float* gate, const fp16* shared_weight,
    fp16* output, const torch::Device& device) {
    static_assert(2560 % Tile == 0);
    auto cgf = [&](sycl::handler& cgh) {
        cgh.parallel_for<MoeCompact160M1DownRoutes<Tile, VectorReduce>>(
            sycl::nd_range<1>((2560 / Tile) * 16, 16),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                slm_init<12 * Tile * sizeof(float)>();
                const int route = item.get_local_id(0);
                const int hbase = item.get_group(0) * Tile;
                if (route < 10) {
                    const int expert = ids[route];
                    const fp16* in = routed + route * 160;
                    simd<fp16,128> x0 = block_load<fp16,128>(in);
                    simd<fp16,128> x1(fp16(0));
                    x1.template select<32,1>(0) = block_load<fp16,32>(in + 128);
                    simd<float,64> e0 = x0.template select<64,2>(0);
                    simd<float,64> o0 = x0.template select<64,2>(1);
                    simd<float,64> e1 = x1.template select<64,2>(0);
                    simd<float,64> o1 = x1.template select<64,2>(1);
                    #pragma unroll
                    for (int h = 0; h < Tile; ++h) {
                        const auto offset = size_t(expert) * 2560 + hbase + h;
                        simd<float,64> acc(0.0f);
                        simd<uint8_t,64> w0 = block_load<uint8_t,64>(weights + offset * 80);
                        simd<uint8_t,64> w1(uint8_t(0));
                        w1.template select<16,1>(0) = block_load<uint8_t,16>(weights + offset * 80 + 64);
                        moe_compact_down_accumulate<64>(acc, e0, o0, w0, float(scales[offset*2]));
                        moe_compact_down_accumulate<16>(acc, e1, o1, w1, float(scales[offset*2+1]));
                        const float dot = sycl::ext::intel::esimd::detail::sum<float,float,64>(acc);
                        slm_block_store<float,1>((route*Tile+h)*sizeof(float),simd<float,1>(dot));
                    }
                } else if (route == 10) {
                    simd<float,64> x0 = block_load<fp16,64>(shared);
                    simd<float,64> x1 = block_load<fp16,64>(shared+64);
                    simd<float,32> tail = block_load<fp16,32>(shared+128);
                    #pragma unroll
                    for (int h = 0; h < Tile; ++h) {
                        const fp16* w = shared_weight + (hbase+h)*160;
                        simd<float,64> acc(0.0f);
                        acc += x0 * simd<float,64>(block_load<fp16,64>(w));
                        acc += x1 * simd<float,64>(block_load<fp16,64>(w+64));
                        const float dot = sycl::ext::intel::esimd::detail::sum<float,float,64>(acc);
                        const float end = sycl::ext::intel::esimd::detail::sum<float,float,32>(
                            tail * simd<float,32>(block_load<fp16,32>(w+128)));
                        slm_block_store<float,1>((10*Tile+h)*sizeof(float),simd<float,1>(dot));
                        slm_block_store<float,1>((11*Tile+h)*sizeof(float),simd<float,1>(end));
                    }
                }
                barrier();
                if (route == 0) {
                    if constexpr (VectorReduce) {
                        // Output lanes are independent: vectorize only the
                        // owner epilogue, keeping each lane's route order.
                        simd<float, Tile> sum(0.0f);
                        for (int r = 0; r < 10; ++r)
                            sum += float(route_weight[r]) * slm_block_load<float, Tile>(r*Tile*sizeof(float));
                        sum += gate[0] * slm_block_load<float, Tile>(10*Tile*sizeof(float));
                        sum += gate[0] * slm_block_load<float, Tile>(11*Tile*sizeof(float));
                        block_store<fp16, Tile>(output+hbase, simd<fp16, Tile>(sum));
                    } else {
                    #pragma unroll
                    for (int h = 0; h < Tile; ++h) {
                        float sum = 0.0f;
                        for (int r = 0; r < 10; ++r)
                            sum += float(route_weight[r]) * float(slm_block_load<float,1>((r*Tile+h)*sizeof(float))[0]);
                        sum += gate[0] * float(slm_block_load<float,1>((10*Tile+h)*sizeof(float))[0]);
                        sum += gate[0] * float(slm_block_load<float,1>((11*Tile+h)*sizeof(float))[0]);
                        output[hbase+h] = fp16(sum);
                    }
                    }
                }
            });
    };
    submit_kernel(cgf, device, "compact160 M1 DOWN route parallel");
}
