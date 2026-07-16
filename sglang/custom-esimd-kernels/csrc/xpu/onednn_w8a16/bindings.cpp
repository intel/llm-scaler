#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <vector>

#include "fp8_gemm_w8a16.h"

namespace {

// Ported from llm-scaler Omni's oneDNN W8A16 path and adapted to the
// copy-free [K, N] FP8 weight layout used by SGLang.
torch::Tensor create_output(
    const torch::Tensor& input,
    const torch::Tensor& weight) {
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "W8A16 matmul supports only 2D or 3D inputs");
  TORCH_CHECK(weight.dim() == 2, "W8A16 weight must be 2D");

  std::vector<int64_t> result_shape;
  if (input.dim() == 2) {
    result_shape = {input.size(0), weight.size(1)};
  } else {
    result_shape = {input.size(0), input.size(1), weight.size(1)};
  }

  auto result_stride = input.strides().vec();
  const auto k = input.size(-1);
  const auto n = result_shape.back();
  for (size_t i = 0; i + 1 < result_stride.size(); ++i) {
    result_stride[i] = result_stride[i] / k * n;
  }
  return at::empty_strided(result_shape, result_stride, input.options());
}

torch::Tensor onednn_fp8_gemm_w8a16(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& weight_scale,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "W8A16 input must be float16 or bfloat16");
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float8_e5m2 ||
          weight.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "W8A16 weight must be float8_e5m2 or float8_e4m3fn");
  TORCH_CHECK(
      input.size(-1) == weight.size(0),
      "W8A16 input K must match weight [K, N]");

  auto result = create_output(input, weight);
  const bool is_nt = weight.strides()[weight.dim() - 2] == 1;
  auto scale = weight_scale.has_value()
      ? weight_scale.value()
      : at::ones({1}, weight.options().dtype(input.dtype()));
  oneDNN::dnnl_matmul_w8a16_fp8(
      result, input, weight, is_nt, bias, scale);
  return result;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_sglang, m) {
  m.def(
      "onednn_fp8_gemm_w8a16(Tensor input, Tensor weight, "
      "Tensor? weight_scale, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_esimd_kernels_sglang, XPU, m) {
  m.impl("onednn_fp8_gemm_w8a16", &onednn_fp8_gemm_w8a16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
