# omni_xpu_kernel

High-performance Intel XPU ESIMD kernels for PyTorch.

## Modules

### sdp — Scaled Dot-Product Attention (Flash Attention)

ESIMD Flash Attention with doubleGRF, AOT-compiled for target GPU.

**Optimizations applied:**
- K prefetch moved before softmax (+89-113% vs baseline)
- Overflow-safe fp32 compensation + clamp (prevents NaN/Inf)
- Adaptive per-head V-scaling with cached decision (zero overhead on normal models)
- Template parameterization via `sdp_config.h` for hardware adaptation

**Performance on Arc B580 (vs PyTorch SDPA):**

| Config | FP16 TFLOPS | BF16 TFLOPS | vs Torch |
|--------|-------------|-------------|----------|
| flux-4096x24 | 73 | 83 | 1.09x / 1.23x |
| wan-3600x40 | 69 | 79 | 1.10x / 1.24x |

```python
from omni_xpu_kernel import sdp

# Input: [B, L, H, 128] fp16/bf16, B==1
output = sdp.sdp(q, k, v)
# V-scaling is automatic: models with large V values (e.g., Qwen Image)
# are scaled to prevent fp16 overflow, with zero overhead on normal models.
```

**AOT compilation for target GPU:**
```bash
# Default target: bmg (Arc B580)
OMNI_XPU_DEVICE=bmg pip install -e . --no-build-isolation

# For other GPUs:
OMNI_XPU_DEVICE=pvc pip install -e . --no-build-isolation   # Data Center GPU Max
```

### linear — FP8 GEMM (oneDNN W8A16)

FP8 weight × FP16/BF16 activation GEMM via oneDNN, with primitive caching.
Supports E4M3 and E5M2 weight formats. E5M2 is 12-17% faster.

```python
from omni_xpu_kernel import linear

# E4M3 or E5M2 weights accepted automatically
output = linear.onednn_w8a16_fp8(x_fp16, weight_fp8, scales_f32)
output = linear.onednn_w8a16_fp8(x_fp16, weight_fp8, scales_f32, bias=bias)

# Cache management
linear.fp8_cache_clear()
hits, misses, size = linear.fp8_cache_stats()
```

### gguf — GGUF Dequantization

| Format | Block Size | Elements |
|--------|------------|----------|
| Q4_0   | 18 bytes   | 32       |
| Q8_0   | 34 bytes   | 32       |
| Q4_K   | 144 bytes  | 256      |
| Q6_K   | 210 bytes  | 256      |

```python
from omni_xpu_kernel import gguf

output = gguf.dequantize_q4_0(quantized, torch.float16)
output = gguf.dequantize_q8_0(quantized, torch.float16)
output = gguf.dequantize_q4_k(quantized, torch.float16)
output = gguf.dequantize_q6_k(quantized, torch.float16)

# Batch dequantization (groups by format, fewer kernel launches)
outputs = gguf.dequantize_batch(
    [tensor1, tensor2, tensor3],
    ['q4_0', 'q4_0', 'q8_0'],
    torch.float16
)
```

### norm — Normalization

RMSNorm, LayerNorm, fused Add+RMSNorm, and fused RMSNorm+Linear.
Supports fp32/fp16/bf16, hidden_size <= 8192 (divisible by 32).

```python
from omni_xpu_kernel import norm

output = norm.rms_norm(weight, input, eps=1e-6)
output = norm.layer_norm(input, weight=weight, bias=None, eps=1e-5)
norm.fused_add_rms_norm(input, residual, weight, eps=1e-6)  # in-place

# Fused RMSNorm + Linear projection (chains in C++, keeps data in L3 cache)
output = norm.fused_rms_norm_linear(input, norm_weight, proj_weight, eps=1e-6)
```

### svdq — SVDQuant W4A4

INT4 weight dequantization, activation quantization, and oneDNN fused
dequant+GEMM for SVDQuant W4A4 inference.

```python
from omni_xpu_kernel import svdq

# ESIMD dequantization
dequantized = svdq.dequantize_w4(packed, scales, out_dtype=torch.bfloat16)
unpacked = svdq.unpack_int4(packed, signed=True)
packed_act, act_scales = svdq.quantize_act_int4(activation, group_size=64)

# oneDNN INT4 GEMM (pre-convert weights once, then use preconverted variant)
packed_u4, scales_f16 = svdq.prepare_onednn_weights(packed, wscales)
output = svdq.onednn_int4_gemm_preconverted(act_f16, packed_u4, scales_f16)

# Fused f16->bf16 convert + add
svdq.fused_convert_add(out_bf16, result_f16, residual_bf16)
```

### rotary — Rotary Position Embedding

Fused bf16->f32 + rotary rotation + f32->bf16 in a single ESIMD kernel.
Supports head_dim 64 and 128.

```python
from omni_xpu_kernel import rotary

output = rotary.rotary_emb(x, cos_cache, sin_cache, seq_len, heads)
```

## Requirements

- Intel oneAPI DPC++/C++ Compiler (icpx)
- PyTorch >= 2.0 with XPU support
- Intel GPU: Arc B-series (BMG), Data Center GPU Max (PVC), or compatible
- oneDNN (for INT4/FP8 GEMM; auto-detected from oneAPI)

## Installation

```bash
source /opt/intel/oneapi/setvars.sh

# Default: builds for Arc B580 (bmg)
pip install -e . --no-build-isolation

# Specify GPU target:
OMNI_XPU_DEVICE=bmg pip install -e . --no-build-isolation   # Arc B580/B770
OMNI_XPU_DEVICE=pvc pip install -e . --no-build-isolation   # Data Center GPU Max
```

On Windows, see [WHL_BUILD_INSTALL.md](WHL_BUILD_INSTALL.md).

## Debug Logging

Controlled by `OMNI_XPU_DEBUG` environment variable. **Disabled by default.**

```bash
# Enable all modules
OMNI_XPU_DEBUG=1 python your_script.py

# Enable specific modules (comma-separated)
OMNI_XPU_DEBUG=sdp python your_script.py       # SDP only
OMNI_XPU_DEBUG=fp8 python your_script.py       # FP8 only
OMNI_XPU_DEBUG=sdp,fp8 python your_script.py   # SDP + FP8

# Legacy FP8 debug (still works)
OMNI_FP8_DEBUG=1 python your_script.py
```

Log format: `[omni_xpu::<module>] <message>`

Example output:
```
[omni_xpu::sdp] call #0: V_max=4.9 threshold=256 needs_scaling=0 q=[1,4096,24,128]
[omni_xpu::fp8] cache MISS: impl=jit:gemm:any (M=4096 K=4096 N=12288 wtype=10)
```

## Tests & Benchmarks

```bash
# Correctness tests
python -m pytest tests/

# All kernel benchmarks
python -m tests.benchmarks.run_all

# Individual benchmarks
python -m tests.benchmarks.run_all --sdp
python -m tests.benchmarks.run_all --norm
python -m tests.benchmarks.run_all --gguf
python -m tests.benchmarks.run_all --onednn
python -m tests.benchmarks.run_all --rotary
```

## Architecture

### SDP Kernel Compilation

The SDP Flash Attention kernel uses ESIMD with doubleGRF and is compiled as a
separate sidecar shared library (`lgrf_sdp.so`). AOT compilation targets a
specific GPU via `-device <target>` (default: bmg).

Configuration is via `sdp_config.h`:
- `ConfigBMG` — Arc B580 (default)
- `ConfigPVC` — Data Center GPU Max
- `ConfigLNL` — Lunar Lake

To switch config at compile time: `-DSDP_CONFIG_PVC`

### Build System

The package builds multiple extension modules:
- `_C.so` — Main extension (norm, gguf, svdq, rotary, sdp loader, fp8)
- `lgrf_sdp.so` — SDP ESIMD sidecar (AOT, doubleGRF)

## License

Apache 2.0
