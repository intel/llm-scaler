/* torch_extension_topk_v2.cc — Standalone registration for TopK V2 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

// Forward declaration
at::Tensor esimd_moe_topk_v2(
    at::Tensor router_logits, at::Tensor top_values, at::Tensor top_indices,
    int64_t T, int64_t num_experts, int64_t topk);

TORCH_LIBRARY(esimd_topk_v2, m) {
  m.def("topk_v2(Tensor router_logits, Tensor top_values, "
        "Tensor top_indices, int T, int num_experts, int topk) -> Tensor");
  m.impl("topk_v2", torch::kXPU, &esimd_moe_topk_v2);
}

PyMODINIT_FUNC PyInit_esimd_topk_v2() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "esimd_topk_v2", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
