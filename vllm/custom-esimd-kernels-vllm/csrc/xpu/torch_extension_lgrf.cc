/* torch_extension_lgrf.cc — Op registration for doubleGRF ESIMD kernels.
 * Uses TORCH_LIBRARY_FRAGMENT to add ops to the existing library namespace.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

#include "kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  m.def("esimd_gdn_conv_fused(Tensor qkvz, "
        "Tensor conv_state, Tensor conv_weight, Tensor conv_bias, "
        "Tensor conv_state_indices, "
        "Tensor A_log, Tensor dt_bias, "
        "Tensor ba, "
        "Tensor ssm_state, Tensor ssm_state_indices, "
        "Tensor output, Tensor z_out, "
        "int N, int H, int HV, int K, int V, "
        "float scale) -> Tensor");
  m.impl("esimd_gdn_conv_fused", torch::kXPU, &esimd_gdn_conv_fused);

  m.def("esimd_gdn_conv_fused_seq(Tensor qkvz, "
        "Tensor conv_state, Tensor conv_weight, Tensor conv_bias, "
        "Tensor conv_state_indices, "
        "Tensor A_log, Tensor dt_bias, "
        "Tensor ba, "
        "Tensor ssm_state, Tensor ssm_state_indices, "
        "Tensor output, Tensor z_out, "
        "int N, int H, int HV, int K, int V, "
        "float scale) -> Tensor");
  m.impl("esimd_gdn_conv_fused_seq", torch::kXPU, &esimd_gdn_conv_fused_seq);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels_lgrf() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "custom_esimd_kernels_lgrf", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
