// Python binding for the Q4_0 XPU quantization kernel.
//
// Registered under the `custom_esimd_kernels_vllm` torch library (same name as
// the existing esimd kernels) so callers can reach it via
// torch.ops.custom_esimd_kernels_vllm.q4_0_quantize(...).

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <Python.h>
#include <sycl/sycl.hpp>

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

void q4_0_quantize_bf16_host(sycl::queue&, const bf16*, int32_t*, fp16*,
                             int64_t, int64_t);
void q4_0_quantize_fp16_host(sycl::queue&, const fp16*, int32_t*, fp16*,
                             int64_t, int64_t);

namespace {

constexpr int64_t kBlock     = 128;  // QK4_GROUP_SIZE
constexpr int64_t kPackFactor = 8;   // nibbles per int32

// Row-major Q4_0 quantization for a 2D tensor. Produces a bit-packed INT4
// tensor plus one FP16 scale per 128-element block. See q4_0_quant.sycl for
// packing details.
void q4_0_quantize(at::Tensor input,
                   at::Tensor out_qweight,
                   at::Tensor out_scale) {
    TORCH_CHECK(input.device().is_xpu(),      "input must be an XPU tensor");
    TORCH_CHECK(out_qweight.device().is_xpu(),"out_qweight must be on XPU");
    TORCH_CHECK(out_scale.device().is_xpu(),  "out_scale must be on XPU");
    TORCH_CHECK(input.dim() == 2,             "input must be 2D");
    TORCH_CHECK(input.is_contiguous(),        "input must be contiguous");
    TORCH_CHECK(out_qweight.is_contiguous(),  "out_qweight must be contiguous");
    TORCH_CHECK(out_scale.is_contiguous(),    "out_scale must be contiguous");
    TORCH_CHECK(out_qweight.scalar_type() == at::kInt,
                "out_qweight must be int32");
    TORCH_CHECK(out_scale.scalar_type() == at::kHalf,
                "out_scale must be float16");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K % kBlock == 0,
                "K must be divisible by ", kBlock, " (got ", K, ")");

    TORCH_CHECK(out_qweight.size(0) == M, "out_qweight row count mismatch");
    TORCH_CHECK(out_qweight.size(1) == K / kPackFactor,
                "out_qweight col count must be K/", kPackFactor);
    TORCH_CHECK(out_scale.size(0) == M, "out_scale row count mismatch");
    TORCH_CHECK(out_scale.size(1) == K / kBlock,
                "out_scale col count must be K/", kBlock);

    sycl::queue& q =
        c10::xpu::getCurrentXPUStream(input.device().index()).queue();

    auto* qw_ptr = reinterpret_cast<int32_t*>(out_qweight.data_ptr());
    auto* sc_ptr = reinterpret_cast<fp16*>(out_scale.data_ptr());

    if (input.scalar_type() == at::kBFloat16) {
        auto* in_ptr = reinterpret_cast<const bf16*>(input.data_ptr());
        q4_0_quantize_bf16_host(q, in_ptr, qw_ptr, sc_ptr, M, K);
    } else if (input.scalar_type() == at::kHalf) {
        auto* in_ptr = reinterpret_cast<const fp16*>(input.data_ptr());
        q4_0_quantize_fp16_host(q, in_ptr, qw_ptr, sc_ptr, M, K);
    } else {
        TORCH_CHECK(false,
                    "input dtype must be bfloat16 or float16 (got ",
                    toString(input.scalar_type()), ")");
    }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
    m.def("q4_0_quantize(Tensor input, Tensor(a!) out_qweight, "
          "Tensor(b!) out_scale) -> ()");
}

TORCH_LIBRARY_IMPL(custom_esimd_kernels_vllm, XPU, m) {
    m.impl("q4_0_quantize", &q4_0_quantize);
}

PyMODINIT_FUNC PyInit_q4_0_quant_ops() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "q4_0_quant_ops", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
