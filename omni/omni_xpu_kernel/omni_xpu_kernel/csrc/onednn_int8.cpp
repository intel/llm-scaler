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

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <torch/extension.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

#include "utils.h"

namespace omni_xpu {
namespace int8_ops {

torch::Tensor dequantize_int8_fused(
    torch::Tensor input,
    torch::Tensor scale,
    torch::ScalarType out_dtype);
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_tensorwise_fused(
    torch::Tensor input,
    std::optional<torch::Tensor> scale_opt);

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
    state->src_scale_md = dnnl::memory::desc({m}, DT::f32, dnnl::memory::format_tag::a);
    // wei_scale: [N] or [1] f32 — per-channel or scalar weight scale
    if (w_scale_is_scalar) {
        state->wei_scale_md = dnnl::memory::desc({1}, DT::f32, dnnl::memory::format_tag::a);
    } else {
        state->wei_scale_md = dnnl::memory::desc({n}, DT::f32, dnnl::memory::format_tag::a);
    }
    // bias: [1, N] f32
    if (has_bias) {
        state->bias_md = dnnl::memory::desc({1, n}, DT::f32, dnnl::memory::format_tag::ab);
    }

    dnnl::primitive_attr attr;
    // Per-token src scale via the grouped-scale API: mask over {M,K}, group {1,K}
    // == one scale per row. The (1<<0),{1,1} form is rejected on XPU s8 matmul.
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, k}, DT::f32);
    // Weight scales: mask = (1 << 1) for per-col, mask = 0 for scalar
    if (w_scale_is_scalar) {
        attr.set_scales(DNNL_ARG_WEIGHTS, 0, {}, DT::f32);
    } else {
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1), {}, DT::f32);
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

torch::Tensor int8_linear_prequantized(
    torch::Tensor x_int8,
    torch::Tensor x_scale,
    torch::Tensor weight,
    torch::Tensor weight_scale,
    std::optional<torch::Tensor> bias,
    int64_t out_dtype_code
) {
    TORCH_CHECK(x_int8.dim() >= 1,
        "x_int8 must have at least one dimension");
    TORCH_CHECK(weight.dim() == 2,
        "weight must be 2D [N, K], got ", weight.dim(), "D");
    TORCH_CHECK(x_int8.device().is_xpu(),
        "x_int8 must be on XPU device");
    TORCH_CHECK(weight.device().is_xpu(),
        "weight must be on XPU device");
    TORCH_CHECK(x_int8.device() == weight.device(),
        "x_int8 and weight must be on the same XPU device");
    TORCH_CHECK(x_int8.scalar_type() == torch::kInt8,
        "x_int8 must be int8, got ", x_int8.scalar_type());
    TORCH_CHECK(weight.scalar_type() == torch::kInt8,
        "weight must be int8, got ", weight.scalar_type());
    TORCH_CHECK(out_dtype_code >= 0 && out_dtype_code <= 2,
        "out_dtype_code must be 0 (float32), 1 (float16), or 2 (bfloat16), got ",
        out_dtype_code);

    const int64_t k = x_int8.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(k > 0, "x_int8.size(-1) must be greater than zero");
    TORCH_CHECK(n > 0, "weight.size(0) must be greater than zero");
    TORCH_CHECK(weight.size(1) == k,
        "weight.size(1)=", weight.size(1),
        " must match x_int8.size(-1)=", k);

    const auto orig_sizes = x_int8.sizes().vec();
    x_int8 = x_int8.reshape({-1, k}).contiguous();
    const int64_t m = x_int8.size(0);
    weight = weight.contiguous();

    // oneDNN consumes one fp32 activation scale per flattened input row and a
    // scalar or per-output-channel fp32 weight scale.
    x_scale = x_scale.to(x_int8.device()).to(torch::kFloat32).reshape(-1).contiguous();
    weight_scale = weight_scale.to(x_int8.device()).to(torch::kFloat32).reshape(-1).contiguous();
    TORCH_CHECK(x_scale.numel() == m,
        "x_scale must contain one value per flattened activation row (M=", m,
        "), got numel=", x_scale.numel());
    TORCH_CHECK(weight_scale.numel() == 1 || weight_scale.numel() == n,
        "weight_scale must be scalar or contain one value per output channel (N=",
        n, "), got numel=", weight_scale.numel());

    torch::ScalarType out_dtype = torch::kBFloat16;
    switch (out_dtype_code) {
        case 0: out_dtype = torch::kFloat; break;
        case 1: out_dtype = torch::kHalf; break;
        case 2: out_dtype = torch::kBFloat16; break;
    }

    std::vector<int64_t> out_sizes(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes.push_back(n);
    if (m == 0) {
        return torch::empty(
            out_sizes,
            torch::TensorOptions().dtype(out_dtype).device(x_int8.device()));
    }

    const bool w_scale_is_scalar = (weight_scale.numel() == 1);
    const bool has_bias = bias.has_value();
    torch::Tensor bias_f32;
    if (has_bias) {
        TORCH_CHECK(bias->numel() == n,
            "bias must contain one value per output channel (N=", n,
            "), got numel=", bias->numel());
        bias_f32 = bias->to(x_int8.device()).to(torch::kFloat32).reshape({1, n}).contiguous();
    }

    sycl::queue& queue = omni_xpu::utils::get_queue(x_int8.device());
    auto state = get_or_create_int8_scaled_primitive(
        m, k, n, static_cast<int>(out_dtype_code), has_bias,
        w_scale_is_scalar, x_int8.device(), queue);

    auto output = torch::empty(
        {m, n}, torch::TensorOptions().dtype(out_dtype).device(x_int8.device()));

    dnnl::stream stream = dnnl::sycl_interop::make_stream(state->engine, queue);
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,
         dnnl::memory(state->src_md, state->engine, x_int8.data_ptr())},
        {DNNL_ARG_WEIGHTS,
         dnnl::memory(state->wei_md, state->engine, weight.data_ptr())},
        {DNNL_ARG_DST,
         dnnl::memory(state->dst_md, state->engine, output.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
         dnnl::memory(state->src_scale_md, state->engine, x_scale.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         dnnl::memory(state->wei_scale_md, state->engine, weight_scale.data_ptr())},
    };
    if (has_bias) {
        args.emplace(
            DNNL_ARG_BIAS,
            dnnl::memory(state->bias_md, state->engine, bias_f32.data_ptr()));
    }
    state->primitive.execute(stream, args);

    return output.reshape(out_sizes);
}

std::tuple<torch::Tensor, torch::Tensor> int8_linear_shared_input(
    torch::Tensor x,
    torch::Tensor weight1,
    torch::Tensor weight_scale1,
    torch::Tensor weight2,
    torch::Tensor weight_scale2,
    std::optional<torch::Tensor> bias1,
    std::optional<torch::Tensor> bias2,
    int64_t out_dtype_code
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(
        x.scalar_type() == torch::kHalf || x.scalar_type() == torch::kBFloat16,
        "x must be float16 or bfloat16, got ", x.scalar_type());
    TORCH_CHECK(weight1.dim() == 2 && weight2.dim() == 2,
        "weight1 and weight2 must be 2D [N, K]");

    const int64_t k = x.size(-1);
    TORCH_CHECK(weight1.size(1) == k,
        "weight1.size(1)=", weight1.size(1),
        " must match x.size(-1)=", k);
    TORCH_CHECK(weight2.size(1) == k,
        "weight2.size(1)=", weight2.size(1),
        " must match x.size(-1)=", k);

    const auto orig_sizes = x.sizes().vec();
    x = x.reshape({-1, k}).contiguous();
    auto [x_int8, x_scale] = quantize_int8_rowwise_fused(x);

    auto output1 = int8_linear_prequantized(
        x_int8, x_scale, weight1, weight_scale1, bias1, out_dtype_code);
    auto output2 = int8_linear_prequantized(
        x_int8, x_scale, weight2, weight_scale2, bias2, out_dtype_code);

    std::vector<int64_t> out_sizes1(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes1.push_back(weight1.size(0));
    std::vector<int64_t> out_sizes2(orig_sizes.begin(), orig_sizes.end() - 1);
    out_sizes2.push_back(weight2.size(0));
    return {
        output1.reshape(out_sizes1),
        output2.reshape(out_sizes2),
    };
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

    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K], got ", weight.dim(), "D");

    const int64_t k = x.size(-1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(weight.size(1) == k,
        "weight.size(1)=", weight.size(1), " must match x.size(-1)=", k);

    // Reshape to 2D
    auto orig_sizes = x.sizes().vec();
    x = x.reshape({-1, k}).contiguous();

    // Step 1: ConvRot (online activation rotation)
    // For now, this falls back to PyTorch — ESIMD rotation kernel will replace later
    if (convrot) {
        TORCH_CHECK(k % convrot_groupsize == 0,
            "ConvRot group size ", convrot_groupsize, " does not divide K=", k);
        // ConvRot rotation is handled by the Python layer calling us;
        // native fused version comes in int8-convrot-esimd task.
        // For now, the Python dispatch layer handles rotation before calling native.
    }

    // Step 2: Dynamic per-row quantization of activation.
    // One plain-SYCL launch performs two coalesced passes for absmax + quantize.
    auto [x_int8, x_scale] = quantize_int8_rowwise_fused(x);
    auto output = int8_linear_prequantized(
        x_int8, x_scale, weight, weight_scale, bias, out_dtype_code);

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

    if (stochastic_rounding <= 0 &&
        (x.scalar_type() == torch::kBFloat16 ||
         x.scalar_type() == torch::kHalf ||
         x.scalar_type() == torch::kFloat)) {
        return quantize_int8_tensorwise_fused(x, scale_opt);
    }

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
        auto gen = at::xpu::detail::createXPUGenerator(x.device().index());
        gen.set_current_seed(stochastic_rounding);
        auto rng = at::rand(x_scaled.sizes(), gen, x_scaled.scalar_type(), c10::nullopt, x.device(), c10::nullopt);
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

    if (stochastic_rounding <= 0 && x.dim() >= 1 &&
        (x.scalar_type() == torch::kBFloat16 ||
         x.scalar_type() == torch::kHalf ||
         x.scalar_type() == torch::kFloat)) {
        return quantize_int8_rowwise_fused(x);
    }

    auto abs_max = x.abs().amax(-1, true); // [..., 1]
    auto scale = (abs_max.to(torch::kFloat32) / 127.0f).clamp_min(1e-30f);

    auto scale_cast = scale.to(x.dtype());
    auto x_scaled = x / scale_cast;
    torch::Tensor q;
    if (stochastic_rounding > 0) {
        auto gen = at::xpu::detail::createXPUGenerator(x.device().index());
        gen.set_current_seed(stochastic_rounding);
        auto rng = at::rand(x_scaled.sizes(), gen, x_scaled.scalar_type(), c10::nullopt, x.device(), c10::nullopt);
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
    return dequantize_int8_fused(q, scale, torch::kFloat);
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
    return dequantize_int8_fused(q, scale, out_dtype);
}

}  // namespace int8_ops
}  // namespace omni_xpu
