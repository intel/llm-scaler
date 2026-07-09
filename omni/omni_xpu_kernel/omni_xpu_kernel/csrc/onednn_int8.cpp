// ============================================================================
// oneDNN INT8 GEMM (s8 × s8 → s32 with per-channel weight scaling)
// ============================================================================
// High-performance INT8 matmul for ComfyUI int8_tensorwise/int8_convrot models.
//
// Architecture:
//   - Activation: s8 [M, K] with per-row scales (dynamic quantization)
//   - Weight: s8 [N, K] with per-channel or scalar scale
//   - Output: bf16/f16 [M, N] after rescaling
//
// The kernel fuses:
//   1. s8 × s8 → s32 matmul via oneDNN DPAS primitive
//   2. int32 → float32 → rescale by (x_scale * w_scale) → out_dtype
//
// Primitive caching: keyed by {device, M, K, N} to amortize creation cost.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "utils.h"

namespace omni_xpu {
namespace int8_ops {

// Forward declaration — defined in int8_quantize_esimd.cpp
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(torch::Tensor x);
// Forward declaration — defined in int8_scaleback_esimd.cpp
torch::Tensor fused_scaleback(torch::Tensor gemm_result, torch::Tensor x_scale,
                              torch::Tensor w_scale, std::optional<torch::Tensor> bias,
                              int64_t out_dtype_code);

namespace {

using DT = dnnl::memory::data_type;

// ============================================================================
// Primitive Cache
// ============================================================================

struct Int8CacheKey {
    int device_index;
    int64_t m;
    int64_t k;
    int64_t n;

    bool operator==(const Int8CacheKey& other) const {
        return device_index == other.device_index
            && m == other.m
            && k == other.k
            && n == other.n;
    }
};

struct Int8CacheKeyHash {
    size_t operator()(const Int8CacheKey& key) const {
        size_t seed = 0;
        auto combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(std::hash<int>{}(key.device_index));
        combine(std::hash<int64_t>{}(key.m));
        combine(std::hash<int64_t>{}(key.k));
        combine(std::hash<int64_t>{}(key.n));
        return seed;
    }
};

struct Int8PrimitiveState {
    dnnl::engine engine;
    dnnl::memory::desc src_md;   // [M, K] s8
    dnnl::memory::desc wei_md;   // [K, N] s8 (weight transposed)
    dnnl::memory::desc dst_md;   // [M, N] s32
    dnnl::matmul primitive;

    Int8PrimitiveState(
        dnnl::engine engine,
        dnnl::memory::desc src_md,
        dnnl::memory::desc wei_md,
        dnnl::memory::desc dst_md,
        dnnl::matmul::primitive_desc pd
    ) : engine(std::move(engine)),
        src_md(std::move(src_md)),
        wei_md(std::move(wei_md)),
        dst_md(std::move(dst_md)),
        primitive(std::move(pd)) {}
};

struct Int8CacheCounters {
    int64_t hits = 0;
    int64_t misses = 0;
};

std::mutex& int8_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

Int8CacheCounters& int8_cache_counters() {
    static Int8CacheCounters counters;
    return counters;
}

std::unordered_map<Int8CacheKey, std::shared_ptr<Int8PrimitiveState>, Int8CacheKeyHash>& int8_primitive_cache() {
    static std::unordered_map<Int8CacheKey, std::shared_ptr<Int8PrimitiveState>, Int8CacheKeyHash> cache;
    return cache;
}

// ============================================================================
// Scaled INT8 GEMM Primitive (fuses rescale into matmul output)
// dst[M,N] bf16 = (src_s8[M,K] × wei_s8[K,N]) * src_scale[M] * wei_scale[N or 1]
// Eliminates the separate scaleback kernel entirely.
// ============================================================================

struct Int8ScaledState {
    dnnl::engine engine;
    dnnl::memory::desc src_md;
    dnnl::memory::desc wei_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc src_scale_md;
    dnnl::memory::desc wei_scale_md;
    dnnl::memory::desc bias_md;
    dnnl::matmul primitive;
    bool has_bias;
    bool w_scale_is_scalar;
};

struct Int8ScaledCacheKey {
    int device_index;
    int out_dtype;  // 0=f32, 1=f16, 2=bf16
    int64_t m;
    int64_t k;
    int64_t n;
    bool has_bias;
    bool w_scale_is_scalar;

    bool operator==(const Int8ScaledCacheKey& other) const {
        return device_index == other.device_index
            && out_dtype == other.out_dtype
            && m == other.m && k == other.k && n == other.n
            && has_bias == other.has_bias
            && w_scale_is_scalar == other.w_scale_is_scalar;
    }
};

struct Int8ScaledCacheKeyHash {
    size_t operator()(const Int8ScaledCacheKey& key) const {
        size_t seed = 0;
        auto combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(std::hash<int>{}(key.device_index));
        combine(std::hash<int>{}(key.out_dtype));
        combine(std::hash<int64_t>{}(key.m));
        combine(std::hash<int64_t>{}(key.k));
        combine(std::hash<int64_t>{}(key.n));
        combine(std::hash<bool>{}(key.has_bias));
        combine(std::hash<bool>{}(key.w_scale_is_scalar));
        return seed;
    }
};

std::unordered_map<Int8ScaledCacheKey, std::shared_ptr<Int8ScaledState>, Int8ScaledCacheKeyHash>& int8_scaled_cache() {
    static std::unordered_map<Int8ScaledCacheKey, std::shared_ptr<Int8ScaledState>, Int8ScaledCacheKeyHash> cache;
    return cache;
}

std::shared_ptr<Int8ScaledState> get_or_create_int8_scaled_primitive(
    int64_t m, int64_t k, int64_t n,
    int out_dtype_code,  // 0=f32, 1=f16, 2=bf16
    bool has_bias,
    bool w_scale_is_scalar,
    const torch::Device& device,
    const sycl::queue& queue
) {
    Int8ScaledCacheKey key{device.index(), out_dtype_code, m, k, n, has_bias, w_scale_is_scalar};

    auto& cache = int8_scaled_cache();
    auto& counters = int8_cache_counters();
    std::lock_guard<std::mutex> lock(int8_cache_mutex());

    auto it = cache.find(key);
    if (it != cache.end()) {
        ++counters.hits;
        return it->second;
    }

    dnnl::engine engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());

    DT dst_dt;
    switch (out_dtype_code) {
        case 0: dst_dt = DT::f32; break;
        case 1: dst_dt = DT::f16; break;
        case 2: dst_dt = DT::bf16; break;
        default: dst_dt = DT::bf16; break;
    }

    auto state = std::make_shared<Int8ScaledState>();
    state->engine = engine;
    state->has_bias = has_bias;
    state->w_scale_is_scalar = w_scale_is_scalar;

    // src: [M, K] s8 row-major
    state->src_md = dnnl::memory::desc({m, k}, DT::s8, dnnl::memory::format_tag::ab);
    // wei: logical [K, N] s8, physical [N, K] row-major → ba format
    state->wei_md = dnnl::memory::desc({k, n}, DT::s8, dnnl::memory::format_tag::ba);
    // dst: [M, N] in output dtype
    state->dst_md = dnnl::memory::desc({m, n}, dst_dt, dnnl::memory::format_tag::ab);
    // src_scale: [M] f32 — per-row activation scale
    state->src_scale_md = dnnl::memory::desc({m, 1}, DT::f32, dnnl::memory::format_tag::ab);
    // wei_scale: [N] or [1] f32 — per-channel or scalar weight scale
    if (w_scale_is_scalar) {
        state->wei_scale_md = dnnl::memory::desc({1, 1}, DT::f32, dnnl::memory::format_tag::ab);
    } else {
        state->wei_scale_md = dnnl::memory::desc({1, n}, DT::f32, dnnl::memory::format_tag::ab);
    }
    // bias: [1, N] in output dtype
    if (has_bias) {
        state->bias_md = dnnl::memory::desc({1, n}, dst_dt, dnnl::memory::format_tag::ab);
    }

    dnnl::primitive_attr attr;
    // Per-row src scales: mask = (1 << 0) means scale varies along dim 0 (rows)
    attr.set_scales(DNNL_ARG_SRC, (1 << 0), {1, 1}, DT::f32);
    // Weight scales: mask = (1 << 1) for per-col, mask = 0 for scalar
    if (w_scale_is_scalar) {
        attr.set_scales(DNNL_ARG_WEIGHTS, 0, {}, DT::f32);
    } else {
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1), {1, 1}, DT::f32);
    }
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd = has_bias
        ? dnnl::matmul::primitive_desc(engine, state->src_md, state->wei_md, state->bias_md, state->dst_md, attr)
        : dnnl::matmul::primitive_desc(engine, state->src_md, state->wei_md, state->dst_md, attr);

    const std::string impl = pd.impl_info_str();
    OMNI_DEBUG("int8", "scaled cache MISS: impl=%s (M=%ld K=%ld N=%ld dst=%d)",
               impl.c_str(), m, k, n, out_dtype_code);
    if (impl.find("ref") != std::string::npos) {
        std::fprintf(stderr, "[omni_xpu::int8] WARNING: scaled oneDNN ref impl for M=%ld K=%ld N=%ld: %s\n",
                     m, k, n, impl.c_str());
    }

    state->primitive = dnnl::matmul(pd);
    cache.emplace(key, state);
    ++counters.misses;
    return state;
}

std::shared_ptr<Int8PrimitiveState> get_or_create_int8_primitive(
    int64_t m,
    int64_t k,
    int64_t n,
    bool weight_transposed,  // true: weight is [N,K] row-major (needs ba format)
    const torch::Device& device,
    const sycl::queue& queue
) {
    // Encode transpose flag into cache key via sign convention
    Int8CacheKey key{device.index(), m, k, weight_transposed ? -n : n};

    auto& cache = int8_primitive_cache();
    auto& counters = int8_cache_counters();
    std::lock_guard<std::mutex> lock(int8_cache_mutex());

    auto it = cache.find(key);
    if (it != cache.end()) {
        ++counters.hits;
        OMNI_DEBUG("int8", "cache HIT (M=%ld K=%ld N=%ld wt=%d hits=%ld)", m, k, n, weight_transposed, counters.hits);
        return it->second;
    }

    dnnl::engine engine = dnnl::sycl_interop::make_engine(
        queue.get_device(),
        queue.get_context()
    );

    // src: [M, K] s8 row-major
    dnnl::memory::desc src_md({m, k}, DT::s8, dnnl::memory::format_tag::ab);
    // wei: logical [K, N] s8
    //   If weight_transposed=true: physical storage is [N, K] row-major → use format ba
    //   If weight_transposed=false: physical storage is [K, N] row-major → use format ab
    dnnl::memory::desc wei_md({k, n}, DT::s8,
        weight_transposed ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
    // dst: [M, N] s32
    dnnl::memory::desc dst_md({m, n}, DT::s32, dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(engine, src_md, wei_md, dst_md, attr);

    const std::string impl = pd.impl_info_str();
    OMNI_DEBUG("int8", "cache MISS: impl=%s (M=%ld K=%ld N=%ld)", impl.c_str(), m, k, n);
    if (impl.find("ref") != std::string::npos) {
        std::fprintf(stderr, "[omni_xpu::int8] WARNING: oneDNN reference impl for M=%ld K=%ld N=%ld: %s\n",
                     m, k, n, impl.c_str());
    }

    auto state = std::make_shared<Int8PrimitiveState>(
        std::move(engine),
        std::move(src_md),
        std::move(wei_md),
        std::move(dst_md),
        std::move(pd)
    );
    cache.emplace(key, state);
    ++counters.misses;
    return state;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

void int8_cache_clear() {
    std::lock_guard<std::mutex> lock(int8_cache_mutex());
    int8_primitive_cache().clear();
    int8_scaled_cache().clear();
    int8_cache_counters() = {};
}

std::tuple<int64_t, int64_t, int64_t> int8_cache_stats() {
    std::lock_guard<std::mutex> lock(int8_cache_mutex());
    const auto& counters = int8_cache_counters();
    return {
        counters.hits,
        counters.misses,
        static_cast<int64_t>(int8_primitive_cache().size() + int8_scaled_cache().size()),
    };
}

torch::Tensor mm_int8(
    torch::Tensor a,
    torch::Tensor b
) {
    TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K], got ", a.dim(), "D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D [K, N], got ", b.dim(), "D");
    TORCH_CHECK(a.scalar_type() == torch::kInt8, "a must be int8");
    TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be int8");
    TORCH_CHECK(a.device().is_xpu(), "a must be on XPU device");
    TORCH_CHECK(b.device().is_xpu(), "b must be on XPU device");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on same device");
    TORCH_CHECK(a.size(1) == b.size(0), "K dimension mismatch: a.size(1)=",
                a.size(1), " vs b.size(0)=", b.size(0));

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);

    // Ensure contiguous
    a = a.contiguous();
    b = b.contiguous();

    // Allocate output
    torch::Tensor output = torch::empty({m, n},
        torch::TensorOptions().dtype(torch::kInt32).device(a.device()));

    if (m == 0 || k == 0 || n == 0) {
        return output.zero_();
    }

    sycl::queue& queue = omni_xpu::utils::get_queue(a.device());
    auto state = get_or_create_int8_primitive(m, k, n, /*weight_transposed=*/false, a.device(), queue);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(state->engine, queue);

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(state->src_md, state->engine, a.data_ptr())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(state->wei_md, state->engine, b.data_ptr())},
        {DNNL_ARG_DST, dnnl::memory(state->dst_md, state->engine, output.data_ptr())},
    };

    state->primitive.execute(stream, args);

    return output;
}

torch::Tensor int8_linear(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_scale,
    std::optional<torch::Tensor> bias,
    int64_t out_dtype_code,
    bool convrot,
    int64_t convrot_groupsize
) {
    // Validate inputs
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(weight.device().is_xpu(), "weight must be on XPU device");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(
        x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16,
        "x must be float16 or bfloat16, got ", x.scalar_type()
    );

    const int64_t k = x.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(weight.size(1) == k,
        "weight.size(1)=", weight.size(1), " must match x.size(-1)=", k);

    // Reshape to 2D
    auto orig_sizes = x.sizes().vec();
    x = x.reshape({-1, k}).contiguous();
    const int64_t m = x.size(0);

    weight = weight.contiguous();
    weight_scale = weight_scale.to(torch::kFloat32).reshape(-1).contiguous();
    TORCH_CHECK(weight_scale.numel() == 1 || weight_scale.numel() == n,
        "weight_scale must be scalar or [N], got numel=", weight_scale.numel());

    // Step 1: ConvRot (online activation rotation)
    // For now, this falls back to PyTorch — ESIMD rotation kernel will replace later
    if (convrot) {
        TORCH_CHECK(k % convrot_groupsize == 0,
            "ConvRot group size ", convrot_groupsize, " does not divide K=", k);
        // ConvRot rotation is handled by the Python layer calling us;
        // native fused version comes in int8-convrot-esimd task.
        // For now, the Python dispatch layer handles rotation before calling native.
    }

    // Step 2: Dynamic per-row quantization of activation
    // Use ESIMD fused kernel: single pass absmax + scale + quantize
    auto [x_int8, x_scale] = quantize_int8_rowwise_fused(x);
    x_int8 = x_int8.contiguous();

    // Step 3: INT8 GEMM with per-channel weight scale fused via oneDNN
    // oneDNN computes: dst = (src_s8 × wei_s8) * wei_scale[n] → f32/bf16
    // Then we apply x_scale[m] separately (much cheaper: just elementwise mul)
    bool w_scale_is_scalar = (weight_scale.numel() == 1);

    sycl::queue& queue = omni_xpu::utils::get_queue(x.device());

    // Try scaled primitive with per-channel weight scale only (no src scale)
    Int8ScaledCacheKey skey{x.device().index(), static_cast<int>(out_dtype_code), m, k, n, false, w_scale_is_scalar};
    std::shared_ptr<Int8ScaledState> sstate;
    {
        auto& cache = int8_scaled_cache();
        auto& counters = int8_cache_counters();
        std::lock_guard<std::mutex> lock(int8_cache_mutex());

        auto it = cache.find(skey);
        if (it != cache.end()) {
            ++counters.hits;
            sstate = it->second;
        } else {
            dnnl::engine engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());

            DT dst_dt;
            switch (out_dtype_code) {
                case 0: dst_dt = DT::f32; break;
                case 1: dst_dt = DT::f16; break;
                case 2: dst_dt = DT::bf16; break;
                default: dst_dt = DT::bf16; break;
            }

            sstate = std::make_shared<Int8ScaledState>();
            sstate->engine = engine;
            sstate->has_bias = false;
            sstate->w_scale_is_scalar = w_scale_is_scalar;

            sstate->src_md = dnnl::memory::desc({m, k}, DT::s8, dnnl::memory::format_tag::ab);
            sstate->wei_md = dnnl::memory::desc({k, n}, DT::s8, dnnl::memory::format_tag::ba);
            sstate->dst_md = dnnl::memory::desc({m, n}, dst_dt, dnnl::memory::format_tag::ab);

            if (w_scale_is_scalar) {
                sstate->wei_scale_md = dnnl::memory::desc({1}, DT::f32, dnnl::memory::format_tag::a);
            } else {
                sstate->wei_scale_md = dnnl::memory::desc({n}, DT::f32, dnnl::memory::format_tag::a);
            }

            dnnl::primitive_attr attr;
            // Per-channel weight scale: mask = 2 (varies along N dimension of dst)
            attr.set_scales_mask(DNNL_ARG_WEIGHTS, w_scale_is_scalar ? 0 : 2);
            attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

            dnnl::matmul::primitive_desc pd(engine, sstate->src_md, sstate->wei_md, sstate->dst_md, attr);
            sstate->primitive = dnnl::matmul(pd);

            const std::string impl = pd.impl_info_str();
            OMNI_DEBUG("int8", "scaled(w-only) cache MISS: impl=%s (M=%ld K=%ld N=%ld)", impl.c_str(), m, k, n);

            cache.emplace(skey, sstate);
            ++counters.misses;
        }
    }

    torch::ScalarType out_dtype;
    switch (out_dtype_code) {
        case 0: out_dtype = torch::kFloat; break;
        case 1: out_dtype = torch::kHalf; break;
        case 2: out_dtype = torch::kBFloat16; break;
        default: out_dtype = torch::kBFloat16; break;
    }

    torch::Tensor output = torch::empty({m, n},
        torch::TensorOptions().dtype(out_dtype).device(x.device()));

    dnnl::stream stream = dnnl::sycl_interop::make_stream(sstate->engine, queue);
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, dnnl::memory(sstate->src_md, sstate->engine, x_int8.data_ptr())},
        {DNNL_ARG_WEIGHTS, dnnl::memory(sstate->wei_md, sstate->engine, weight.data_ptr())},
        {DNNL_ARG_DST, dnnl::memory(sstate->dst_md, sstate->engine, output.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(sstate->wei_scale_md, sstate->engine, weight_scale.data_ptr())},
    };
    sstate->primitive.execute(stream, args);

    // Step 4: Apply per-row x_scale (cheap elementwise multiply)
    // output[m,n] *= x_scale[m]  (broadcasts over N dimension)
    output.mul_(x_scale.to(out_dtype));

    // Step 5: Bias
    if (bias.has_value()) {
        TORCH_CHECK(bias->size(0) == n, "bias size must match N=", n);
        output.add_(bias->to(out_dtype).reshape({1, -1}));
    }

    // Reshape back to original batch dimensions
    std::vector<int64_t> out_sizes(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes.push_back(n);
    return output.reshape(out_sizes);
}

// ============================================================================
// Quantization kernels (placeholder — ESIMD versions in int8_quantize.cpp)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_tensorwise(
    torch::Tensor x,
    std::optional<torch::Tensor> scale_opt,
    int64_t stochastic_rounding
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");

    // Compute scale from absmax if not provided
    torch::Tensor scale;
    if (scale_opt.has_value()) {
        scale = scale_opt->to(torch::kFloat32).to(x.device());
    } else {
        auto abs_max = x.abs().max();
        scale = (abs_max.to(torch::kFloat32) / 127.0f).clamp_min(1e-30f);
    }

    // Quantize
    auto scale_cast = scale.to(x.dtype());
    auto x_scaled = x / scale_cast;
    torch::Tensor q;
    if (stochastic_rounding > 0) {
        auto gen = at::detail::createCPUGenerator(stochastic_rounding);
        auto rng = torch::rand_like(x_scaled);
        q = (x_scaled + rng).floor().clamp(-128.0f, 127.0f).to(torch::kInt8);
    } else {
        q = x_scaled.round().clamp(-128.0f, 127.0f).to(torch::kInt8);
    }

    return {q, scale};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise(
    torch::Tensor x,
    int64_t stochastic_rounding
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");

    auto abs_max = x.abs().amax(-1, true); // [..., 1]
    auto scale = (abs_max.to(torch::kFloat32) / 127.0f).clamp_min(1e-30f);

    auto scale_cast = scale.to(x.dtype());
    auto x_scaled = x / scale_cast;
    torch::Tensor q;
    if (stochastic_rounding > 0) {
        auto rng = torch::rand_like(x_scaled);
        q = (x_scaled + rng).floor().clamp(-128.0f, 127.0f).to(torch::kInt8);
    } else {
        q = x_scaled.round().clamp(-128.0f, 127.0f).to(torch::kInt8);
    }

    return {q, scale};
}

torch::Tensor dequantize_int8_simple(
    torch::Tensor q,
    torch::Tensor scale
) {
    TORCH_CHECK(q.scalar_type() == torch::kInt8, "q must be int8");
    return q.to(torch::kFloat32) * scale.to(torch::kFloat32);
}

torch::Tensor dequantize_int8_simple_dtype(
    torch::Tensor q,
    torch::Tensor scale,
    int64_t output_dtype_code
) {
    torch::ScalarType out_dtype;
    switch (output_dtype_code) {
        case 0: out_dtype = torch::kFloat; break;
        case 1: out_dtype = torch::kHalf; break;
        case 2: out_dtype = torch::kBFloat16; break;
        default: out_dtype = torch::kFloat; break;
    }
    return dequantize_int8_simple(q, scale).to(out_dtype);
}

}  // namespace int8_ops
}  // namespace omni_xpu
