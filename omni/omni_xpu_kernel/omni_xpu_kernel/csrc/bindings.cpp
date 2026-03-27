// ============================================================================
// omni_xpu_kernel - Python Bindings
// ============================================================================
// High-performance Intel XPU ESIMD kernels for ComfyUI
// 
// GGUF Dequantization: Q4_0, Q8_0, Q4_K, Q6_K
// Normalization: RMSNorm, LayerNorm
// SVDQuant: W4A4 dequantization, quantization, and oneDNN GEMM for nunchaku
// Rotary: Fused rotary position embedding
// ============================================================================

#include <torch/extension.h>

namespace omni_xpu {
namespace gguf {
    torch::Tensor dequantize_q4_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q8_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q4_k(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q6_k(const torch::Tensor& input, torch::ScalarType dtype);
    std::vector<torch::Tensor> dequantize_batch(
        const std::vector<torch::Tensor>& inputs,
        const std::vector<std::string>& formats,
        torch::ScalarType dtype);
}
namespace norm {
    torch::Tensor rms_norm(torch::Tensor weight, torch::Tensor input, double eps);
    torch::Tensor layer_norm(torch::Tensor input, std::optional<torch::Tensor> weight, std::optional<torch::Tensor> bias, double eps);
    void fused_add_rms_norm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps);
    torch::Tensor fused_rms_norm_linear(torch::Tensor input, torch::Tensor norm_weight, torch::Tensor proj_weight, double eps);
}
namespace svdq {
    torch::Tensor dequantize_svdq_w4(const torch::Tensor& packed, const torch::Tensor& scales, torch::ScalarType out_dtype);
    torch::Tensor unpack_svdq_int4(const torch::Tensor& packed, bool is_signed);
    std::tuple<torch::Tensor, torch::Tensor> quantize_svdq_act_int4(const torch::Tensor& input, int64_t group_size);
    torch::Tensor onednn_int4_gemm(const torch::Tensor& act, const torch::Tensor& packed, const torch::Tensor& wscales);
    torch::Tensor onednn_int4_gemm_preconverted(const torch::Tensor& act, const torch::Tensor& packed_u4, const torch::Tensor& scales_f16);
    void onednn_int4_gemm_add_to_output(const torch::Tensor& act, const torch::Tensor& packed_u4, const torch::Tensor& scales_f16, torch::Tensor& dst);
    void fused_convert_add(torch::Tensor& out, const torch::Tensor& result, const torch::Tensor& residual);
    torch::Tensor fused_smooth_convert(const torch::Tensor& x, const torch::Tensor& smooth_factor);
    torch::Tensor fused_smooth_mul_convert(const torch::Tensor& x, const torch::Tensor& rcp_smooth);
}
namespace rotary {
    torch::Tensor rotary_emb(const torch::Tensor& x, const torch::Tensor& cos_cache, const torch::Tensor& sin_cache, int64_t seq_len, int64_t heads);
}
namespace sdp {
    torch::Tensor sdp(torch::Tensor q, torch::Tensor k, torch::Tensor v);
}
namespace linear {
    torch::Tensor onednn_w8a16_fp8(torch::Tensor input, torch::Tensor weight, torch::Tensor scale_w, std::optional<torch::Tensor> bias);
    void fp8_cache_clear();
    std::tuple<int64_t, int64_t, int64_t> fp8_cache_stats();
}
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "omni_xpu_kernel - High-performance Intel XPU ESIMD kernels for ComfyUI";
    
    // GGUF Dequantization
    auto gguf = m.def_submodule("gguf", "GGUF dequantization kernels");
    
    gguf.def("dequantize_q4_0", &omni_xpu::gguf::dequantize_q4_0,
        "Dequantize Q4_0 tensor (18 bytes/block -> 32 elements)",
        py::arg("input"), py::arg("dtype") = torch::kFloat16);
    
    gguf.def("dequantize_q8_0", &omni_xpu::gguf::dequantize_q8_0,
        "Dequantize Q8_0 tensor (34 bytes/block -> 32 elements)",
        py::arg("input"), py::arg("dtype") = torch::kFloat16);
    
    gguf.def("dequantize_q4_k", &omni_xpu::gguf::dequantize_q4_k,
        "Dequantize Q4_K tensor (144 bytes/block -> 256 elements)",
        py::arg("input"), py::arg("dtype") = torch::kFloat16);
    
    gguf.def("dequantize_q6_k", &omni_xpu::gguf::dequantize_q6_k,
        "Dequantize Q6_K tensor (210 bytes/block -> 256 elements)",
        py::arg("input"), py::arg("dtype") = torch::kFloat16);

    gguf.def("dequantize_batch", &omni_xpu::gguf::dequantize_batch,
        "Batch dequantize multiple tensors in fewer kernel launches.\n"
        "Groups tensors by format, concatenates, launches one kernel per format group,\n"
        "then splits outputs. Reduces N submissions to num_format_types submissions.\n"
        "Input: inputs=[tensor1, tensor2, ...], formats=['q4_0', 'q8_0', ...], dtype\n"
        "Output: list of dequantized tensors in same order as inputs",
        py::arg("inputs"), py::arg("formats"), py::arg("dtype") = torch::kFloat16);

    // Normalization
    auto norm = m.def_submodule("norm", "Normalization kernels");
    
    norm.def("rms_norm", &omni_xpu::norm::rms_norm,
        "RMSNorm using ESIMD optimization",
        py::arg("weight"), py::arg("input"), py::arg("eps") = 1e-6);
    
    norm.def("layer_norm", &omni_xpu::norm::layer_norm,
        "LayerNorm using ESIMD optimization",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("bias") = py::none(), py::arg("eps") = 1e-5);
    
    norm.def("fused_add_rms_norm", &omni_xpu::norm::fused_add_rms_norm,
        "Fused Add + RMSNorm using ESIMD optimization (in-place: residual += input, input = rmsnorm(residual) * weight)",
        py::arg("input"), py::arg("residual"), py::arg("weight"), py::arg("eps") = 1e-6);

    norm.def("fused_rms_norm_linear", &omni_xpu::norm::fused_rms_norm_linear,
        "Fused RMSNorm + Linear projection in single C++ call.\n"
        "Chains norm and matmul without Python roundtrip, keeping normalized data in L3 cache.\n"
        "output = RMSNorm(input, norm_weight, eps) @ proj_weight.T\n"
        "Input: input [M, K], norm_weight [K], proj_weight [N, K]\n"
        "Output: [M, N]",
        py::arg("input"), py::arg("norm_weight"), py::arg("proj_weight"), py::arg("eps") = 1e-6);

    // SVDQuant W4A4 Dequantization/Quantization (nunchaku)
    auto svdq = m.def_submodule("svdq", "SVDQuant W4A4 dequantization and quantization kernels for nunchaku");

    svdq.def("dequantize_svdq_w4", &omni_xpu::svdq::dequantize_svdq_w4,
        "Dequantize SVDQuant W4 packed weights: unpack INT4 + apply per-group scales -> output dtype\n"
        "Input: packed [N, K/2] uint8, scales [num_groups, N]\n"
        "Output: [N, K] dequantized values",
        py::arg("packed"), py::arg("scales"), py::arg("out_dtype") = torch::kBFloat16);

    svdq.def("unpack_svdq_int4", &omni_xpu::svdq::unpack_svdq_int4,
        "Unpack SVDQuant INT4 packed tensor to int8 (no scaling)\n"
        "Input: packed [M, K/2] uint8\n"
        "Output: [M, K] int8 signed values",
        py::arg("packed"), py::arg("is_signed") = true);

    svdq.def("quantize_svdq_act_int4", &omni_xpu::svdq::quantize_svdq_act_int4,
        "Quantize activation to SVDQuant INT4 with per-group absmax scaling\n"
        "Input: [M, K] bf16/f32\n"
        "Output: (packed [M, K/2] uint8, scales [num_groups, M])",
        py::arg("input"), py::arg("group_size") = 64);

    svdq.def("onednn_int4_gemm", &omni_xpu::svdq::onednn_int4_gemm,
        "Fused INT4 dequant + GEMM using oneDNN u4 matmul primitive\n"
        "Converts signed INT4 to u4 and bf16 scales to f16 per call\n"
        "Input: act [M, K] bf16/f16/f32, packed [N, K/2] uint8, wscales [G, N] bf16\n"
        "Output: [M, N] same dtype as act",
        py::arg("act"), py::arg("packed"), py::arg("wscales"));

    svdq.def("onednn_int4_gemm_preconverted", &omni_xpu::svdq::onednn_int4_gemm_preconverted,
        "Fused INT4 dequant + GEMM using oneDNN u4 matmul (pre-converted weights)\n"
        "Accepts already-converted u4 weights (packed^0x88) and f16 scales\n"
        "Input: act [M, K] bf16/f16/f32, packed_u4 [N, K/2] uint8, scales_f16 [G, N] f16\n"
        "Output: [M, N] same dtype as act",
        py::arg("act"), py::arg("packed_u4"), py::arg("scales_f16"));

    svdq.def("onednn_int4_gemm_add_to_output", &omni_xpu::svdq::onednn_int4_gemm_add_to_output,
        "Fused INT4 GEMM + accumulate into bf16 output using oneDNN append_sum post-op\n"
        "dst += GEMM(f16_act, u4_wgt) — caller pre-fills dst with residual\n"
        "Input: act [M, K] f16, packed_u4 [N, K/2] uint8, scales_f16 [G, N] f16, dst [M, N] bf16\n"
        "Output: dst modified in-place (dst += GEMM result)",
        py::arg("act"), py::arg("packed_u4"), py::arg("scales_f16"), py::arg("dst"));

    svdq.def("fused_convert_add", &omni_xpu::svdq::fused_convert_add,
        "Fused f16->bf16 conversion + bf16 addition in single ESIMD kernel\n"
        "Writes: out = bf16(result[:Mo,:No]) + residual[:Mo,:No]\n"
        "Input: out [Mo, No] bf16, result [Mr, Nr] f16, residual [Mo, No] bf16",
        py::arg("out"), py::arg("result"), py::arg("residual"));

    svdq.def("fused_smooth_convert", &omni_xpu::svdq::fused_smooth_convert,
        "Fused smooth division + bf16->f16 conversion (legacy, uses division)\n"
        "Input: x [M, K] bf16, smooth_factor [K] bf16\n"
        "Output: [M, K] f16 = (x / smooth_factor).to(f16)",
        py::arg("x"), py::arg("smooth_factor"));

    svdq.def("fused_smooth_mul_convert", &omni_xpu::svdq::fused_smooth_mul_convert,
        "Fused smooth multiply + bf16->f16 conversion (optimized, uses multiply-by-reciprocal)\n"
        "Input: x [M, K] bf16, rcp_smooth [K] f16 (pre-computed 1/smooth_factor)\n"
        "Output: [M, K] f16 = (x * rcp_smooth).to(f16)",
        py::arg("x"), py::arg("rcp_smooth"));

    // Rotary Embedding
    auto rotary = m.def_submodule("rotary", "Rotary position embedding kernels");

    rotary.def("rotary_emb", &omni_xpu::rotary::rotary_emb,
        "Fused rotary position embedding using ESIMD optimization\n"
        "Fuses bf16→f32 promotion + rotary rotation + f32→bf16 demotion\n"
        "Input: x [total_rows, head_dim] bf16/f16/f32\n"
        "       cos_cache [S, head_dim/2] f32\n"
        "       sin_cache [S, head_dim/2] f32\n"
        "Output: [total_rows, head_dim] same dtype as x",
        py::arg("x"), py::arg("cos_cache"), py::arg("sin_cache"),
        py::arg("seq_len"), py::arg("heads"));

    // FP8 Linear (oneDNN W8A16)
    auto linear = m.def_submodule("linear", "FP8 linear kernels");
    linear.def("onednn_w8a16_fp8", &omni_xpu::linear::onednn_w8a16_fp8,
        "FP8 GEMM: W8A16 matmul with E4M3/E5M2 weights via oneDNN.\n"
        "Input: x [M, K] fp16/bf16, weight [N, K] float8, scales [N] f32\n"
        "Output: [M, N] same dtype as x",
        py::arg("input"), py::arg("weight"), py::arg("scale_w"), py::arg("bias") = py::none());
    linear.def("fp8_cache_clear", &omni_xpu::linear::fp8_cache_clear,
        "Clear FP8 primitive cache");
    linear.def("fp8_cache_stats", &omni_xpu::linear::fp8_cache_stats,
        "Return FP8 cache stats as (hits, misses, size)");

    // Scaled Dot-Product Attention (ESIMD Flash Attention)
    auto sdp = m.def_submodule("sdp", "Scaled dot-product attention kernels");
    sdp.def("sdp", &omni_xpu::sdp::sdp,
        "ESIMD Flash Attention for Intel XPU\n"
        "Input: q/k/v [B, L, H, 128] fp16/bf16 contiguous on XPU\n"
        "Constraints: B == 1, head_dim == 128\n"
        "V is automatically per-head scaled to prevent fp16 accumulator overflow.\n"
        "Output: [B, Lq, H, 128] same dtype as q",
        py::arg("q"), py::arg("k"), py::arg("v"));
}
