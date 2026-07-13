# Changelog

## dev/kitchen_xpu (2026-07-13)

### Build System

- Kernel and Omni image development versions now share
  `omni_xpu_kernel/_version.py`, initially set to `0.1.0-b8-dev`. Wheel
  metadata uses the normalized PEP 440 version `0.1.0b8.dev0`.
- CUTE FMHA is required by default on Linux. A normal build now requires a
  complete `CUTLASS_SYCL_ROOT` and fails if `cute_fmha_torch` cannot be
  included.
- `OMNI_XPU_REQUIRE_CUTE=0` is the explicit core-only opt-out and is required
  for Windows, where the CUTE extension is not supported.
- The Omni Docker image keeps an explicit `OMNI_XPU_REQUIRE_CUTE=1` release
  guard so an incomplete CUTE build fails during image construction.

## dev/sdp-optimization (2026-03-27)

### Performance

- **SDP Flash Attention: +89-113% throughput** (38→73/85 TFLOPS on Arc B580)
  - K prefetch moved before softmax, maximizing prefetch distance (~500 cycles vs ~100)
  - This single change is responsible for the majority of the speedup
  - Verified: SLM load pipelining (0% gain), fast_exp2 polynomial (-16%, GPU EM unit conflict), tile 64→128 (infeasible, exceeds GRF)

- **FP8 GEMM: E5M2 12-17% faster than E4M3** on large matrices
  - E5M2 exponent width matches FP16, enabling simpler hardware conversion
  - decode batch=16: +44% speedup

### Correctness & Robustness

- **SDP overflow prevention** — two-layer protection:
  1. Overflow-safe fp32 compensation with clamp to [-65504, 65504]
  2. Adaptive per-head V-scaling for models with large V values (Qwen Image, etc.)
  - Cached decision: first call checks V range, subsequent calls use cache (zero overhead)

- **FP8: E5M2 weight format support** alongside existing E4M3

### Build System

- **`OMNI_XPU_DEVICE` env var** — configurable AOT target device (default: `bmg`)
  - Fixes P0: original `pvc` target caused 28+ minute JIT fallback on Arc B580
  - AOT compilation: 4.7 seconds vs 28+ minutes JIT

- **SDP template parameterization** (`sdp_config.h`) — compile-time hardware configs
  - `ConfigBMG`, `ConfigPVC`, `ConfigLNL` — switch via `-DSDP_CONFIG_PVC`
  - Zero performance overhead (verified via A/B test)

### New APIs

- `norm.fused_rms_norm_linear(input, norm_weight, proj_weight)` — chains RMSNorm + Linear in C++
- `gguf.dequantize_batch(inputs, formats, dtype)` — batch dequantization
- `linear.onednn_w8a16_fp8(x, weight, scales)` — FP8 GEMM with E4M3/E5M2 support
- `utils.h: submit_kernel` — templated to avoid `std::function` overhead

### Infrastructure

- **Unified debug logging** via `OMNI_XPU_DEBUG` env var (disabled by default)
  - `OMNI_XPU_DEBUG=1` (all), `=sdp`, `=fp8`, `=sdp,fp8` (selective)
  - Backward compatible with `OMNI_FP8_DEBUG`

### Kernel Health Summary (Arc B580)

| Kernel | TFLOPS | % Peak | vs Torch |
|--------|--------|--------|----------|
| SDP fp16 | 73 | 67% | 1.09x |
| SDP bf16 | 83 | 77% | 1.23x |
| oneDNN INT4 GEMM (M=4096) | 105 | 96% | — |
| RMSNorm | — | 65% BW | — |
| GGUF Q4_0 | — | 81% BW | — |
| FP8 GEMM E5M2 (M=4096) | 99 | 91% | — |
