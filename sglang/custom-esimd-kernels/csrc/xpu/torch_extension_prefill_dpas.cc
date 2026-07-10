/* torch_extension_prefill_dpas.cc — Op registration for the doubleGRF
 * sdp_paged_prefill_dpas kernel. Uses TORCH_LIBRARY_FRAGMENT to add
 * to the existing custom_esimd_kernels_vllm namespace.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

// Forward decl of the impl in esimd_kernel_prefill_dpas.sycl (same module).
at::Tensor esimd_sdpa_prefill_dpas(
    at::Tensor q, at::Tensor key_cache, at::Tensor value_cache,
    at::Tensor cu_seqlens_q, at::Tensor seq_lens, bool is_causal,
    std::optional<double> scale, at::Tensor block_table);

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  m.def("esimd_sdpa_prefill_dpas(Tensor q, Tensor key_cache, "
        "Tensor value_cache, Tensor cu_seqlens_q, Tensor seq_lens, "
        "bool is_causal, float? scale, Tensor block_table) -> Tensor");
  m.impl("esimd_sdpa_prefill_dpas", torch::kXPU, &esimd_sdpa_prefill_dpas);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels_prefill_dpas() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "custom_esimd_kernels_prefill_dpas", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
