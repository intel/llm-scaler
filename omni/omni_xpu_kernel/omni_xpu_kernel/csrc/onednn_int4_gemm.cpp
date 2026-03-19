// ============================================================================
// oneDNN INT4 Fused Dequant + GEMM
// ============================================================================
// Uses oneDNN's native u4 matmul primitive with per-group block quantization
// scales to fuse the dequantize_w4 + bf16 matmul into a single oneDNN call.
//
// Reference: ModelTC/LightX2V lightx2v_kernel_xpu/csrc/onednn.cpp
//
// SVDQuant stores signed INT4 [-8,7] packed as uint8. We map to unsigned
// u4 [0,15] by setting zero_point = 8 (signed_val = unsigned_val - 8).
//
// Performance: Primitive creation is expensive (~5ms). We cache primitives
// keyed by {ActDT, M, K, N, group_size} so repeated calls (1224 per image)
// only pay the creation cost once.
// ============================================================================

#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <map>
#include <tuple>
#include <mutex>

#include "utils.h"

namespace omni_xpu {
namespace svdq {

// Cache key: (act_dtype_int, M, K, N, group_size)
using CacheKey = std::tuple<int, int64_t, int64_t, int64_t, int64_t>;

struct CachedPrimitive {
    dnnl::engine eng;
    dnnl::stream strm;
    dnnl::matmul prim;
    dnnl::memory::desc src_md;
    dnnl::memory::desc wei_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc scale_md;
    dnnl::memory::desc zp_md;
};

static std::map<CacheKey, CachedPrimitive> g_cache;       // plain GEMM
static std::map<CacheKey, CachedPrimitive> g_cache_sum;   // GEMM + append_sum
static std::mutex g_cache_mutex;

// Persistent engine/stream (created once per device)
static bool g_engine_initialized = false;
static dnnl::engine g_engine;
static dnnl::stream g_stream;

static void ensure_engine_initialized(const torch::Device& device) {
    if (!g_engine_initialized) {
        sycl::queue& q = omni_xpu::utils::get_queue(device);
        g_engine = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        g_stream = dnnl::sycl_interop::make_stream(g_engine, q);
        g_engine_initialized = true;
    }
}

template <dnnl::memory::data_type ActDT>
static CachedPrimitive& get_or_create_primitive(
    std::map<CacheKey, CachedPrimitive>& cache,
    const CacheKey& key,
    int64_t M, int64_t K, int64_t N, int64_t group_size,
    bool use_sum_postop,
    dnnl::memory::data_type dst_dt
) {
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    CachedPrimitive cp;
    cp.eng = g_engine;
    cp.strm = g_stream;

    cp.src_md = dnnl::memory::desc({M, K}, ActDT, dnnl::memory::format_tag::ab);
    cp.wei_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::u4,
                                   dnnl::memory::format_tag::ba);
    cp.dst_md = dnnl::memory::desc({M, N}, dst_dt, dnnl::memory::format_tag::ab);

    int64_t num_groups = K / group_size;
    cp.scale_md = dnnl::memory::desc({num_groups, N}, dnnl::memory::data_type::f16,
                                     dnnl::memory::format_tag::ab);
    cp.zp_md = dnnl::memory::desc({1}, dnnl::memory::data_type::u8,
                                  dnnl::memory::format_tag::a);

    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                    {group_size, 1}, dnnl::memory::data_type::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::u8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    if (use_sum_postop) {
        dnnl::post_ops po;
        po.append_sum(1.0f);
        attr.set_post_ops(po);
    }

    dnnl::matmul::primitive_desc pd(cp.eng, cp.src_md, cp.wei_md, cp.dst_md, attr);

    std::string impl_info = pd.impl_info_str();
    const char* tag = use_sum_postop ? "onednn_int4_gemm_sum" : "onednn_int4_gemm";
    fprintf(stderr, "[%s] CACHE MISS: impl=%s (M=%ld K=%ld N=%ld gs=%ld)\n",
            tag, impl_info.c_str(), M, K, N, group_size);
    if (impl_info.find("ref") != std::string::npos) {
        fprintf(stderr, "[%s] WARNING: reference fallback (slow)\n", tag);
    }

    cp.prim = dnnl::matmul(pd);

    auto [ins, _] = cache.emplace(key, std::move(cp));
    return ins->second;
}

template <dnnl::memory::data_type ActDT>
static void onednn_int4_gemm_kernel(
    void* act_ptr,
    void* weight_ptr,
    void* scales_ptr,
    void* zero_ptr,
    void* output_ptr,
    int64_t M, int64_t K, int64_t N,
    int64_t group_size,
    const torch::Device& device
) {
    CacheKey key(static_cast<int>(ActDT), M, K, N, group_size);

    CachedPrimitive* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        ensure_engine_initialized(device);
        cached = &get_or_create_primitive<ActDT>(
            g_cache, key, M, K, N, group_size, false, ActDT);
    }

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,                                  dnnl::memory(cached->src_md,   cached->eng, act_ptr)},
        {DNNL_ARG_WEIGHTS,                              dnnl::memory(cached->wei_md,   cached->eng, weight_ptr)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,       dnnl::memory(cached->scale_md, cached->eng, scales_ptr)},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,  dnnl::memory(cached->zp_md,    cached->eng, zero_ptr)},
        {DNNL_ARG_DST,                                  dnnl::memory(cached->dst_md,   cached->eng, output_ptr)},
    };

    cached->prim.execute(cached->strm, args);
}

// GEMM with append_sum: dst = GEMM(act, wgt) + dst
// Caller must pre-fill dst with the residual before calling this.
// Activation: f16, Weights: u4, Dst: bf16.
template <dnnl::memory::data_type ActDT, dnnl::memory::data_type DstDT>
static void onednn_int4_gemm_sum_kernel(
    void* act_ptr,
    void* weight_ptr,
    void* scales_ptr,
    void* zero_ptr,
    void* output_ptr,
    int64_t M, int64_t K, int64_t N,
    int64_t group_size,
    const torch::Device& device
) {
    CacheKey key(static_cast<int>(ActDT), M, K, N, group_size);

    CachedPrimitive* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        ensure_engine_initialized(device);
        cached = &get_or_create_primitive<ActDT>(
            g_cache_sum, key, M, K, N, group_size, true, DstDT);
    }

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,                                  dnnl::memory(cached->src_md,   cached->eng, act_ptr)},
        {DNNL_ARG_WEIGHTS,                              dnnl::memory(cached->wei_md,   cached->eng, weight_ptr)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,       dnnl::memory(cached->scale_md, cached->eng, scales_ptr)},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,  dnnl::memory(cached->zp_md,    cached->eng, zero_ptr)},
        {DNNL_ARG_DST,                                  dnnl::memory(cached->dst_md,   cached->eng, output_ptr)},
    };

    cached->prim.execute(cached->strm, args);
}


torch::Tensor onednn_int4_gemm_preconverted(
    const torch::Tensor& act,
    const torch::Tensor& packed_u4,
    const torch::Tensor& scales_f16
);

// Convenience wrapper: converts signed INT4 packed → u4 and bf16 scales → f16 per call
torch::Tensor onednn_int4_gemm(
    const torch::Tensor& act,
    const torch::Tensor& packed,
    const torch::Tensor& wscales
) {
    // Convert signed packed to unsigned: XOR with 0x88 flips sign bit of each nibble
    torch::Tensor packed_u4 = (packed ^ 0x88).contiguous();
    // Convert scales bf16 → f16 for oneDNN
    torch::Tensor scales_f16 = wscales.to(torch::kFloat16).contiguous();

    return onednn_int4_gemm_preconverted(act, packed_u4, scales_f16);
}


// Fast path: accepts pre-converted u4 weights and f16 scales (no per-call conversion)
torch::Tensor onednn_int4_gemm_preconverted(
    const torch::Tensor& act,
    const torch::Tensor& packed_u4,
    const torch::Tensor& scales_f16
) {
    TORCH_CHECK(act.dim() == 2, "act must be 2D [M, K], got ", act.dim(), "D");
    TORCH_CHECK(packed_u4.dim() == 2, "packed_u4 must be 2D [N, K/2], got ", packed_u4.dim(), "D");
    TORCH_CHECK(scales_f16.dim() == 2, "scales_f16 must be 2D [G, N], got ", scales_f16.dim(), "D");
    TORCH_CHECK(act.device().is_xpu(), "act must be on XPU device");
    TORCH_CHECK(packed_u4.device().is_xpu(), "packed_u4 must be on XPU device");
    TORCH_CHECK(scales_f16.device().is_xpu(), "scales_f16 must be on XPU device");

    int64_t M = act.size(0);
    int64_t K = act.size(1);
    int64_t N = packed_u4.size(0);

    TORCH_CHECK(packed_u4.size(1) == K / 2,
                "packed_u4.size(1)=", packed_u4.size(1), " must equal K/2=", K / 2);
    TORCH_CHECK(packed_u4.scalar_type() == torch::kUInt8,
                "packed_u4 must be uint8");
    TORCH_CHECK(scales_f16.scalar_type() == torch::kFloat16,
                "scales_f16 must be float16 (pre-converted)");

    int64_t num_groups = scales_f16.size(0);
    TORCH_CHECK(scales_f16.size(1) == N,
                "scales_f16.size(1)=", scales_f16.size(1), " must equal N=", N);

    int64_t group_size = K / num_groups;
    TORCH_CHECK(group_size * num_groups == K,
                "K=", K, " must be divisible by num_groups=", num_groups);

    torch::Tensor output = torch::empty({M, N},
        torch::TensorOptions().dtype(act.scalar_type()).device(act.device()));

    // Persistent scalar zero-point = 8 on XPU
    static torch::Tensor zp;
    if (!zp.defined() || zp.device() != act.device()) {
        zp = torch::tensor({8}, torch::TensorOptions().dtype(torch::kUInt8).device(act.device()));
    }

    torch::Tensor act_c = act.contiguous();

    switch (act_c.scalar_type()) {
        case torch::kBFloat16:
            onednn_int4_gemm_kernel<dnnl::memory::data_type::bf16>(
                act_c.data_ptr(), packed_u4.data_ptr(), scales_f16.data_ptr(),
                zp.data_ptr(), output.data_ptr(), M, K, N, group_size, act_c.device());
            break;
        case torch::kFloat16:
            onednn_int4_gemm_kernel<dnnl::memory::data_type::f16>(
                act_c.data_ptr(), packed_u4.data_ptr(), scales_f16.data_ptr(),
                zp.data_ptr(), output.data_ptr(), M, K, N, group_size, act_c.device());
            break;
        case torch::kFloat:
            onednn_int4_gemm_kernel<dnnl::memory::data_type::f32>(
                act_c.data_ptr(), packed_u4.data_ptr(), scales_f16.data_ptr(),
                zp.data_ptr(), output.data_ptr(), M, K, N, group_size, act_c.device());
            break;
        default:
            TORCH_CHECK(false, "Unsupported activation dtype: ", act_c.scalar_type(),
                        ". Only bf16, f16, f32 are supported.");
    }

    return output;
}


// dst += GEMM(f16_act, u4_wgt) — fused GEMM + accumulate into bf16 output.
// Caller pre-fills dst with the value to accumulate onto (e.g. LoRA residual).
void onednn_int4_gemm_add_to_output(
    const torch::Tensor& act,
    const torch::Tensor& packed_u4,
    const torch::Tensor& scales_f16,
    torch::Tensor& dst
) {
    TORCH_CHECK(act.dim() == 2, "act must be 2D [M, K]");
    TORCH_CHECK(packed_u4.dim() == 2, "packed_u4 must be 2D [N, K/2]");
    TORCH_CHECK(scales_f16.dim() == 2, "scales_f16 must be 2D [G, N]");
    TORCH_CHECK(dst.dim() == 2, "dst must be 2D [M, N]");
    TORCH_CHECK(act.device().is_xpu(), "act must be on XPU device");

    int64_t M = act.size(0);
    int64_t K = act.size(1);
    int64_t N = packed_u4.size(0);

    TORCH_CHECK(dst.size(0) == M && dst.size(1) == N,
                "dst shape [", dst.size(0), ",", dst.size(1), "] must match [", M, ",", N, "]");
    TORCH_CHECK(packed_u4.size(1) == K / 2);
    TORCH_CHECK(packed_u4.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales_f16.scalar_type() == torch::kFloat16);
    TORCH_CHECK(dst.scalar_type() == torch::kBFloat16, "dst must be bf16");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");

    int64_t num_groups = scales_f16.size(0);
    int64_t group_size = K / num_groups;

    static torch::Tensor zp;
    if (!zp.defined() || zp.device() != act.device()) {
        zp = torch::tensor({8}, torch::TensorOptions().dtype(torch::kUInt8).device(act.device()));
    }

    torch::Tensor act_c = act.contiguous();

    TORCH_CHECK(act_c.scalar_type() == torch::kFloat16,
                "act must be f16 for gemm_add_to_output (got ", act_c.scalar_type(), ")");

    onednn_int4_gemm_sum_kernel<dnnl::memory::data_type::f16, dnnl::memory::data_type::bf16>(
        act_c.data_ptr(), packed_u4.data_ptr(), scales_f16.data_ptr(),
        zp.data_ptr(), dst.data_ptr(), M, K, N, group_size, act_c.device());
}

}  // namespace svdq
}  // namespace omni_xpu
