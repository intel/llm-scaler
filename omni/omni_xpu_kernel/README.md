# omni_xpu_kernel

High-performance Intel XPU ESIMD kernels for PyTorch.

## Modules

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
```

### norm — Normalization

RMSNorm, LayerNorm, and fused Add+RMSNorm using ESIMD.
Supports fp32/fp16/bf16, hidden_size ≤ 8192 (divisible by 32).

```python
from omni_xpu_kernel import norm

output = norm.rms_norm(weight, input, eps=1e-6)
output = norm.layer_norm(input, weight=weight, bias=None, eps=1e-5)
norm.fused_add_rms_norm(input, residual, weight, eps=1e-6)  # in-place
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

# Fused f16→bf16 convert + add
svdq.fused_convert_add(out_bf16, result_f16, residual_bf16)
```

### rotary — Rotary Position Embedding

Fused bf16→f32 + rotary rotation + f32→bf16 in a single ESIMD kernel.

```python
from omni_xpu_kernel import rotary

output = rotary.rotary_emb(x, cos_cache, sin_cache, seq_len, heads)
```

## Requirements

- Intel oneAPI DPC++/C++ Compiler (icpx)
- PyTorch ≥ 2.0 with XPU support
- oneDNN (for INT4 GEMM; auto-detected from oneAPI)

## Installation

```bash
source /opt/intel/oneapi/setvars.sh
pip install -e . --no-build-isolation
```

On Windows, see [WHL_BUILD_INSTALL.md](WHL_BUILD_INSTALL.md).

## Tests & Benchmarks

```bash
# Correctness tests
python -m pytest tests/

# All kernel benchmarks
python -m tests.benchmarks.run_all

# Individual benchmarks
python -m tests.benchmarks.run_all --svdq
python -m tests.benchmarks.run_all --onednn
python -m tests.benchmarks.run_all --rmsnorm
python -m tests.benchmarks.run_all --rotary
```

## License

Apache 2.0
