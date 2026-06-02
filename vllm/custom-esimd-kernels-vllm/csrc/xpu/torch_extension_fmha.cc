/* torch_extension_fmha.cc — Op registration for prefill FMHA ESIMD kernel.
 * Uses TORCH_LIBRARY_FRAGMENT to add ops to the existing library namespace.
 */
#include <Python.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

// Forward declaration (defined in esimd_kernel_fmha.sycl)
at::Tensor esimd_prefill_fmha(
    at::Tensor output,
    at::Tensor query,
    at::Tensor key_cache,
    at::Tensor value_cache,
    at::Tensor block_table,
    at::Tensor cu_seqlens_q,
    at::Tensor seqused_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double sm_scale,
    bool is_causal);

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
  m.def("esimd_prefill_fmha(Tensor output, Tensor query, Tensor key_cache, "
        "Tensor value_cache, Tensor block_table, Tensor cu_seqlens_q, "
        "Tensor seqused_k, int max_seqlen_q, int max_seqlen_k, "
        "float sm_scale, bool is_causal) -> Tensor");
  m.impl("esimd_prefill_fmha", torch::kXPU, &esimd_prefill_fmha);
}

PyMODINIT_FUNC PyInit_custom_esimd_kernels_fmha() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "custom_esimd_kernels_fmha", nullptr, 0, nullptr
    };
    return PyModule_Create(&module);
}
