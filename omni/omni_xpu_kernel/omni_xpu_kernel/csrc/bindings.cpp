// ============================================================================
// omni_xpu_kernel - Python Bindings
// ============================================================================
// Unified Python interface for Intel XPU optimized kernels
// 
// Currently supported:
//   - GGUF Dequantization: Q4_0, Q8_0, Q4_K, Q6_K
//   - Normalization: RMSNorm, LayerNorm
// ============================================================================

#include <torch/extension.h>

// Forward declarations from kernel implementations
namespace omni_xpu {
namespace gguf {
    torch::Tensor dequantize_q4_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q4_0_comfyui(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q4_0_impl(const torch::Tensor& input, torch::ScalarType dtype, bool sequential_layout);
    torch::Tensor dequantize_q8_0(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q4_k(const torch::Tensor& input, torch::ScalarType dtype);
    torch::Tensor dequantize_q6_k(const torch::Tensor& input, torch::ScalarType dtype);
    double benchmark(const torch::Tensor& input, torch::ScalarType dtype, int warmup_iters, int bench_iters);
}
namespace norm {
    torch::Tensor rms_norm(torch::Tensor weight, torch::Tensor input, double eps);
    torch::Tensor layer_norm(torch::Tensor input, std::optional<torch::Tensor> weight, std::optional<torch::Tensor> bias, double eps);
}
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "omni_xpu_kernel - High-performance Intel XPU ESIMD kernels";
    
    // GGUF Dequantization kernels
    auto gguf = m.def_submodule("gguf", "GGUF dequantization kernels (Q4_0, Q8_0, Q4_K, Q6_K)");
    
    gguf.def("dequantize_q4_0", &omni_xpu::gguf::dequantize_q4_0,
        "Dequantize Q4_0 tensor (interleaved output)",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16
    );
    
    gguf.def("dequantize_q4_0_comfyui", &omni_xpu::gguf::dequantize_q4_0_comfyui,
        "Dequantize Q4_0 tensor (sequential output for ComfyUI)",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16
    );
    
    gguf.def("dequantize_q4_0_layout", &omni_xpu::gguf::dequantize_q4_0_impl,
        "Dequantize Q4_0 tensor with configurable layout",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16,
        py::arg("sequential_layout") = false
    );
    
    gguf.def("dequantize_q8_0", &omni_xpu::gguf::dequantize_q8_0,
        "Dequantize Q8_0 tensor",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16
    );
    
    gguf.def("dequantize_q4_k", &omni_xpu::gguf::dequantize_q4_k,
        "Dequantize Q4_K tensor",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16
    );
    
    gguf.def("dequantize_q6_k", &omni_xpu::gguf::dequantize_q6_k,
        "Dequantize Q6_K tensor",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16
    );
    
    gguf.def("benchmark", &omni_xpu::gguf::benchmark,
        "Benchmark Q4_0 dequantization",
        py::arg("input"),
        py::arg("dtype") = torch::kFloat16,
        py::arg("warmup_iters") = 10,
        py::arg("bench_iters") = 100
    );
    
    // Normalization kernels
    auto norm = m.def_submodule("norm", "Normalization kernels (RMSNorm, LayerNorm)");
    
    norm.def("rms_norm", &omni_xpu::norm::rms_norm,
        "RMSNorm using ESIMD optimization",
        py::arg("weight"),
        py::arg("input"),
        py::arg("eps") = 1e-6
    );
    
    norm.def("layer_norm", &omni_xpu::norm::layer_norm,
        "LayerNorm using ESIMD optimization",
        py::arg("input"),
        py::arg("weight") = py::none(),
        py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5
    );
}

