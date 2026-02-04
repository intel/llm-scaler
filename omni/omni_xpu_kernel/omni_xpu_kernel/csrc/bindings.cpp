// ============================================================================
// omni_xpu_kernel - Python Bindings
// ============================================================================
// High-performance Intel XPU ESIMD kernels for ComfyUI
// 
// GGUF Dequantization: Q4_0, Q8_0, Q4_K, Q6_K
// Normalization: RMSNorm, LayerNorm
// ============================================================================

#include <torch/extension.h>

namespace omni_xpu {
namespace gguf {
    torch::Tensor dequantize_q4_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q8_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q4_k(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q6_k(const torch::Tensor& input, torch::ScalarType dtype);
}
namespace norm {
    torch::Tensor rms_norm(torch::Tensor weight, torch::Tensor input, double eps);
    torch::Tensor layer_norm(torch::Tensor input, std::optional<torch::Tensor> weight, std::optional<torch::Tensor> bias, double eps);
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
    
    // Normalization
    auto norm = m.def_submodule("norm", "Normalization kernels");
    
    norm.def("rms_norm", &omni_xpu::norm::rms_norm,
        "RMSNorm using ESIMD optimization",
        py::arg("weight"), py::arg("input"), py::arg("eps") = 1e-6);
    
    norm.def("layer_norm", &omni_xpu::norm::layer_norm,
        "LayerNorm using ESIMD optimization",
        py::arg("input"), py::arg("weight") = py::none(), py::arg("bias") = py::none(), py::arg("eps") = 1e-5);
}
