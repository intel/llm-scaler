// ============================================================================
// oneDNN W4A8 GEMM: s8 activations x u4 weights, per-group scales, optional
// per-block zero points (TINT4/torchao asymmetric INT4).
//
// src scales f32 (QW-class bf16 activations can have group absmax > 8.3e6,
// which overflows an f16 scale to inf -> dequant NaN -> black image);
// wei scales f16; wei zero point = scalar 8 (wa4) or per-block [G_wei, N]
// (tint4/torchao, w = (q - zp) * scale inside oneDNN, no Python correction).
// src/wei group sizes are decoupled (src 32/64, wei up to 128).
// ============================================================================

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <torch/extension.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "utils.h"

namespace omni_xpu {
namespace svdq {

// Cache key: (dst_dtype_int, M, K, N, shape_id(+zp flag))
using CacheKey = std::tuple<int, int64_t, int64_t, int64_t, int64_t>;

struct CachedPrimitive {
    dnnl::engine eng;
    dnnl::stream strm;
    dnnl::matmul prim;
    dnnl::memory::desc src_md;
    dnnl::memory::desc wei_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc xscale_md;
    dnnl::memory::desc wscale_md;
    dnnl::memory::desc zp_md;
};

static std::map<CacheKey, CachedPrimitive> g_cache;
static std::mutex g_cache_mutex;

static std::map<int, std::pair<dnnl::engine, dnnl::stream>> g_engine_map;

static std::pair<dnnl::engine, dnnl::stream>& ensure_engine_initialized(
    const torch::Device& device) {
    int dev_idx = device.index();
    auto it = g_engine_map.find(dev_idx);
    if (it != g_engine_map.end()) return it->second;
    sycl::queue& q = omni_xpu::utils::get_queue(device);
    auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
    auto [ins, _] = g_engine_map.emplace(
        dev_idx, std::make_pair(std::move(eng), std::move(strm)));
    return ins->second;
}

template <dnnl::memory::data_type DstDT>
static CachedPrimitive& get_or_create_s8u4(
    const CacheKey& key,
    int64_t M, int64_t K, int64_t N,
    int64_t act_gs, int64_t wei_gs, bool per_block_zp,
    const dnnl::engine& eng,
    const dnnl::stream& strm) {
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    CachedPrimitive cp;
    cp.eng = eng;
    cp.strm = strm;
    cp.src_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s8,
                                   dnnl::memory::format_tag::ab);
    cp.wei_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::u4,
                                   dnnl::memory::format_tag::ba);
    cp.dst_md = dnnl::memory::desc({M, N}, DstDT, dnnl::memory::format_tag::ab);
    const int64_t G_src = K / act_gs;
    const int64_t G_wei = K / wei_gs;
    cp.xscale_md = dnnl::memory::desc({M, G_src}, dnnl::memory::data_type::f32,
                                      dnnl::memory::format_tag::ab);
    // wei scale f16（与 a16 tint4 gs=128 路径一致；f32 wei scale 在 gs=128
    // 下输出全零）。src scale f32 防激活溢出。
    cp.wscale_md = dnnl::memory::desc({G_wei, N}, dnnl::memory::data_type::f16,
                                      dnnl::memory::format_tag::ab);
    cp.zp_md = per_block_zp
        ? dnnl::memory::desc({G_wei, N}, dnnl::memory::data_type::u8,
                             dnnl::memory::format_tag::ab)
        : dnnl::memory::desc({1}, dnnl::memory::data_type::u8,
                             dnnl::memory::format_tag::a);

    dnnl::primitive_attr attr;
    // src scales: per-row per-K-group — mask3 {1, gs}
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1),
                    {1, act_gs}, dnnl::memory::data_type::f32);
    // weight scales: per-group per-N — mask3 {gs, 1}
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                    {wei_gs, 1}, dnnl::memory::data_type::f16);
    if (per_block_zp) {
        // TINT4/torchao 非对称：per-block zero points（mask3 {gs, 1}）
        attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                             {wei_gs, 1}, dnnl::memory::data_type::u8);
    } else {
        // signed int4 -> u4 需要标量 zp=8
        attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {},
                             dnnl::memory::data_type::u8);
    }
    // A770 u4 matmul: fp16-class accumulation overflows to NaN at Qwen-scale
    // K; fpmath any allows bf16 accumulation (wide exponent).
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(cp.eng, cp.src_md, cp.wei_md, cp.dst_md, attr);
    std::string impl_info = pd.impl_info_str();
    fprintf(stderr,
            "[onednn_s8u4_gemm] CACHE MISS: impl=%s (M=%ld K=%ld N=%ld "
            "act_gs=%ld wei_gs=%ld zp=%d)\n",
            impl_info.c_str(), (long)M, (long)K, (long)N,
            (long)act_gs, (long)wei_gs, (int)per_block_zp);
    if (impl_info.find("ref") != std::string::npos) {
        fprintf(stderr, "[onednn_s8u4_gemm] WARNING: reference fallback (slow)\n");
    }
    cp.prim = dnnl::matmul(pd);
    auto [ins, _] = g_cache.emplace(key, std::move(cp));
    return ins->second;
}

template <dnnl::memory::data_type DstDT>
static void onednn_s8u4_kernel(
    void* act_ptr, void* xscales_ptr, void* packed_ptr, void* wscales_ptr,
    void* zp_ptr, void* output_ptr,
    int64_t M, int64_t K, int64_t N,
    int64_t act_gs, int64_t wei_gs, bool per_block_zp,
    const torch::Device& device) {
    CacheKey key(static_cast<int>(DstDT), M, K, N,
                 (act_gs * 1000 + wei_gs) + (per_block_zp ? (1 << 21) : 0));
    CachedPrimitive* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto& [eng, strm] = ensure_engine_initialized(device);
        cached = &get_or_create_s8u4<DstDT>(
            key, M, K, N, act_gs, wei_gs, per_block_zp, eng, strm);
    }
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,                            dnnl::memory(cached->src_md,   cached->eng, act_ptr)},
        {DNNL_ARG_WEIGHTS,                        dnnl::memory(cached->wei_md,   cached->eng, packed_ptr)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,     dnnl::memory(cached->xscale_md, cached->eng, xscales_ptr)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(cached->wscale_md, cached->eng, wscales_ptr)},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(cached->zp_md, cached->eng, zp_ptr)},
        {DNNL_ARG_DST,                            dnnl::memory(cached->dst_md,   cached->eng, output_ptr)},
    };
    cached->prim.execute(cached->strm, args);
}

torch::Tensor onednn_s8u4_gemm(
    const torch::Tensor& act,
    const torch::Tensor& xscales,
    const torch::Tensor& packed_u4,
    const torch::Tensor& scales_f16,
    torch::ScalarType out_dtype,
    std::optional<torch::Tensor> zp_u8) {
    TORCH_CHECK(act.dim() == 2, "act must be [M, K]");
    TORCH_CHECK(act.scalar_type() == torch::kInt8, "act must be int8");
    TORCH_CHECK(act.device().is_xpu(), "act must be on XPU");
    TORCH_CHECK(packed_u4.scalar_type() == torch::kUInt8, "packed_u4 must be uint8");
    TORCH_CHECK(xscales.scalar_type() == torch::kFloat16 ||
                    xscales.scalar_type() == torch::kFloat,
                "xscales must be f16 or f32");
    TORCH_CHECK(scales_f16.scalar_type() == torch::kFloat16 ||
                    scales_f16.scalar_type() == torch::kFloat,
                "scales_f16 must be f16 or f32");

    const int64_t M = act.size(0);
    const int64_t K = act.size(1);
    const int64_t N = packed_u4.size(0);
    TORCH_CHECK(packed_u4.size(1) == K / 2, "packed_u4.size(1) must equal K/2");
    const int64_t G_src = xscales.size(1);
    const int64_t G_wei = scales_f16.size(0);
    TORCH_CHECK(G_src > 0, "xscales must have at least one group column");
    TORCH_CHECK(xscales.size(0) == M, "xscales must be [M, G_src]");
    TORCH_CHECK(scales_f16.size(1) == N, "scales_f16 must be [G_wei, N]");
    const int64_t act_gs = K / G_src;
    const int64_t wei_gs = K / G_wei;
    TORCH_CHECK(act_gs * G_src == K, "K must be divisible by G_src");
    TORCH_CHECK(wei_gs * G_wei == K, "K must be divisible by G_wei");

    const bool per_block_zp = zp_u8.has_value()
        && zp_u8->defined() && zp_u8->numel() > 0;
    torch::Tensor zp_c;   // 持久的连续副本（per-block）
    static torch::Tensor zp_scalar;
    const void* zp_ptr;
    if (per_block_zp) {
        const torch::Tensor& zpt = zp_u8.value();
        TORCH_CHECK(zpt.scalar_type() == torch::kUInt8, "zp_u8 must be uint8");
        TORCH_CHECK(zpt.dim() == 2, "zp_u8 must be [G_wei, N]");
        TORCH_CHECK(zpt.size(0) == G_wei && zpt.size(1) == N,
                    "zp_u8 must be [G_wei, N]");
        zp_c = zpt.contiguous();
        zp_ptr = zp_c.data_ptr();
    } else {
        if (!zp_scalar.defined() || zp_scalar.device() != act.device()) {
            zp_scalar = torch::tensor({8},
                torch::TensorOptions().dtype(torch::kUInt8).device(act.device()));
        }
        zp_ptr = zp_scalar.data_ptr();
    }

    torch::Tensor output = torch::empty({M, N},
        torch::TensorOptions().dtype(out_dtype).device(act.device()));
    const auto act_c = act.contiguous();
    const auto xs_c = xscales.contiguous();
    const auto wu_c = packed_u4.contiguous();
    const auto ws_c = scales_f16.contiguous();
    const auto xs_f32 = xs_c.scalar_type() == torch::kFloat
                            ? xs_c
                            : xs_c.to(torch::kFloat);

    switch (out_dtype) {
        case torch::kFloat:
            onednn_s8u4_kernel<dnnl::memory::data_type::f32>(
                act_c.data_ptr(), xs_f32.data_ptr(), wu_c.data_ptr(),
                ws_c.data_ptr(), const_cast<void*>(zp_ptr), output.data_ptr(),
                M, K, N, act_gs, wei_gs, per_block_zp, act_c.device());
            break;
        case torch::kHalf:
            onednn_s8u4_kernel<dnnl::memory::data_type::f16>(
                act_c.data_ptr(), xs_f32.data_ptr(), wu_c.data_ptr(),
                ws_c.data_ptr(), const_cast<void*>(zp_ptr), output.data_ptr(),
                M, K, N, act_gs, wei_gs, per_block_zp, act_c.device());
            break;
        default:
            onednn_s8u4_kernel<dnnl::memory::data_type::bf16>(
                act_c.data_ptr(), xs_f32.data_ptr(), wu_c.data_ptr(),
                ws_c.data_ptr(), const_cast<void*>(zp_ptr), output.data_ptr(),
                M, K, N, act_gs, wei_gs, per_block_zp, act_c.device());
            break;
    }
    return output;
}

}  // namespace svdq
}  // namespace omni_xpu
