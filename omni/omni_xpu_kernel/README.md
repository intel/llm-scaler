# omni_xpu_kernel

High-performance Intel XPU ESIMD kernels for PyTorch.

## Features

### GGUF Dequantization
Optimized for ComfyUI-GGUF with up to **90x speedup** over PyTorch reference:

| Format | Block Size | Elements | Performance |
|--------|------------|----------|-------------|
| Q4_0   | 18 bytes   | 32       | ~25-47x faster |
| Q8_0   | 34 bytes   | 32       | ~4-6x faster |
| Q4_K   | 144 bytes  | 256      | ~15-90x faster |
| Q6_K   | 210 bytes  | 256      | ~20-62x faster |

### Normalization Kernels
- RMSNorm and LayerNorm
- Supports fp32, fp16, bf16
- Hidden size up to 8192

## Requirements

- Intel oneAPI DPC++/C++ Compiler
- PyTorch >= 2.0 with XPU support
- CMake >= 3.18

## Installation

```bash
# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Install
pip install . --no-build-isolation
```

## Usage

### GGUF Dequantization

```python
import torch
from omni_xpu_kernel import gguf

# Q4_0: 18 bytes -> 32 elements
quantized_q4_0 = torch.randint(0, 256, (1000 * 18,), dtype=torch.uint8, device="xpu")
output = gguf.dequantize_q4_0(quantized_q4_0, torch.float16)

# Q8_0: 34 bytes -> 32 elements
quantized_q8_0 = torch.randint(0, 256, (1000 * 34,), dtype=torch.uint8, device="xpu")
output = gguf.dequantize_q8_0(quantized_q8_0, torch.float16)

# Q4_K: 144 bytes -> 256 elements
quantized_q4_k = torch.randint(0, 256, (1000 * 144,), dtype=torch.uint8, device="xpu")
output = gguf.dequantize_q4_k(quantized_q4_k, torch.float16)

# Q6_K: 210 bytes -> 256 elements
quantized_q6_k = torch.randint(0, 256, (1000 * 210,), dtype=torch.uint8, device="xpu")
output = gguf.dequantize_q6_k(quantized_q6_k, torch.float16)
```

### Normalization

```python
import torch
from omni_xpu_kernel import norm

input = torch.randn(32, 4096, device="xpu", dtype=torch.float16)
weight = torch.ones(4096, device="xpu", dtype=torch.float16)

# RMSNorm
output = norm.rms_norm(weight, input, eps=1e-6)

# LayerNorm
output = norm.layer_norm(input, weight=weight, bias=None, eps=1e-5)
```

## API Reference

### gguf module

- `dequantize_q4_0(input, dtype=torch.float16)` - Q4_0 dequantization
- `dequantize_q8_0(input, dtype=torch.float16)` - Q8_0 dequantization
- `dequantize_q4_k(input, dtype=torch.float16)` - Q4_K dequantization
- `dequantize_q6_k(input, dtype=torch.float16)` - Q6_K dequantization

### norm module

- `rms_norm(weight, input, eps=1e-6)` - RMSNorm
- `layer_norm(input, weight=None, bias=None, eps=1e-5)` - LayerNorm

## License

Apache 2.0
