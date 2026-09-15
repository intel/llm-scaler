#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace qwen38_small_n {
using namespace sycl::ext::intel::esimd;
using half = sycl::half;

// GDN's TP4/TP8 b/a projection: N=24/12 cannot fill a 16-column DPAS tile.
// Parallelize over (row, column, K-split), without padding the weights/output.
// Keep FP16 dequantization, FP32 accumulation and the original group128 scale.
struct Int4SmallNKernel {
    const half* input;
    const uint8_t* weight;
    const half* scale;
    half* output;
    int N;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int K = 2560, KS = 4, VL = 64;
        slm_init<KS * sizeof(float)>();
        const int row = item.get_group(0) / N;
        const int n = item.get_group(0) % N;
        const int split = item.get_local_id(0);
        simd<float, VL> acc = 0.0f;
        for (int k = split * (K / KS); k < (split + 1) * (K / KS); k += 128) {
            const simd<uint8_t, VL> packed = block_load<uint8_t, VL>(weight + n * (K / 2) + k / 2);
            const simd<uint16_t, VL> raw = convert<uint16_t>(packed);
            simd<half, VL> lo = convert<half>(raw & 15);
            simd<half, VL> hi = convert<half>((raw >> 4) & 15);
            const simd<half, VL> s = scale[n * (K / 128) + k / 128];
            const simd<half, VL> zero = s * half(-8.0f);
            lo = lo * s + zero;
            hi = hi * s + zero;
            simd<half, 128> x = block_load<half, 128>(input + row * K + k);
            const simd<float, VL> even = x.template select<VL, 2>(0);
            const simd<float, VL> odd = x.template select<VL, 2>(1);
            acc += even * convert<float>(lo) + odd * convert<float>(hi);
        }
        slm_block_store<float, 1>(split * sizeof(float),
            simd<float, 1>(reduce<float>(acc, std::plus<>())));
        barrier();
        if (split == 0) {
            const simd<float, KS> parts = slm_block_load<float, KS>(0);
            output[row * N + n] = half(reduce<float>(parts, std::plus<>()));
        }
    }
};

inline void launch(const half* input, const uint8_t* weight, const half* scale,
                   half* output, int M, int N, sycl::queue& queue) {
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(M * N * 4, 4),
                       Int4SmallNKernel{input, weight, scale, output, N});
    });
}
}  // namespace qwen38_small_n
