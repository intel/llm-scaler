# moe_prefill — INT4 MoE prefill kernels (DPAS-based)

**Target**: Qwen3.5-122B-A10B and similar large-M MoE models on Intel Xe GPUs.
**Coexists with**: `moe_batch/moe_int4.sycl` (decode-optimized, N-major GEMV).

See [`DESIGN.md`](./DESIGN.md) for full specification.

## Build

Uses the same `setup.py` as the rest of `custom-esimd-kernels-vllm`:

```bash
cd /llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm
python setup.py build_ext --inplace
```

Produces `python/custom_esimd_kernels_vllm/moe_int4_prefill_ops.cpython-*.so`.

## Python import

```python
from custom_esimd_kernels_vllm import moe_int4_prefill_ops as ops

# Low-level ops (one per kernel)
offsets, tokens = ops.moe_prefill_gather_v2(selected_experts, num_local_experts)
inter    = ops.moe_prefill_up_int4(x, gate_up_qw, gate_up_sc, offsets, tokens, top_k)
exp_out  = ops.moe_prefill_down_int4(inter, down_qw, down_sc, offsets, tokens)
out      = ops.moe_prefill_finalize(exp_out, routing_weights, top_k)

# Or end-to-end
out = ops.moe_prefill_forward_int4(
    x, selected_experts, routing_weights,
    gate_up_qw, gate_up_sc, down_qw, down_sc,
    top_k, num_local_experts)
```

## Dispatch from vLLM

Use `USE_ESIMD_MOE=1` env var to swap `ipex.llm.modules.GatedMLPMOE` for the
ESIMD implementation (see `llm-scaler-vllm-xpu/vllm/model_executor/layers/quantization/sym_int4.py`).
M-threshold for prefill vs decode dispatch is `moe_prefill_int4::kPrefillMThreshold` (=64).

## Status

| Phase | Kernel | Status |
|---|---|---|
| P0 | Skeleton + registration | IN PROGRESS |
| P1 | `moe_prefill_gather_v2` | todo |
| P2 | `moe_prefill_up_int4` (DPAS) | todo |
| P3 | `moe_prefill_down_int4` (DPAS) | todo |
| P4 | `moe_prefill_finalize` | todo |
| P5 | vLLM integration | todo |
| P6 | Benchmark vs ipex marlin | todo |
