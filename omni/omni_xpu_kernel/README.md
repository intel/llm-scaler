# omni_xpu_kernel

High-performance Intel XPU kernels for PyTorch.

## Features

- **GGUF Q4_0 Dequantization**
  - ESIMD-optimized implementation
  - Supports both standard GGUF and ComfyUI memory layouts

- **Normalization Kernels**: High-performance RMSNorm and LayerNorm
  - ESIMD-optimized with shared local memory
  - Supports fp32, fp16, bf16
  - Hidden size up to 8192

## Requirements

- Intel oneAPI DPC++/C++ Compiler (icpx on Linux, icx on Windows)
- PyTorch >= 2.0 with XPU support
- CMake >= 3.18

## Installation

### Linux

```bash
# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Install (--no-build-isolation required for torch dependency)
pip install . --no-build-isolation

# Or for development
pip install -e . --no-build-isolation
```

Quick build script:
```bash
./scripts/build.sh
```

### Windows

1. Install Intel oneAPI Base Toolkit from:
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

2. Open a command prompt and set up the oneAPI environment:
```cmd
"C:\Program Files\Intel\oneAPI\setvars.bat"
```

3. Install the package:
```cmd
pip install . --no-build-isolation

REM Or for development
pip install -e . --no-build-isolation
```

Quick build script:
```cmd
scripts\build.bat
```

**Note:** On Windows, make sure you have:
- Intel oneAPI Base Toolkit 2024.2 or later
- PyTorch with XPU support
- Visual Studio Build Tools (for linking)

## Usage

### GGUF Dequantization

```python
import torch
from omni_xpu_kernel import gguf

# Create quantized data on XPU
quantized = torch.randint(0, 256, (1000 * 18,), dtype=torch.uint8, device="xpu")

# Dequantize with standard GGUF layout
output = gguf.dequantize_q4_0(quantized, torch.float16)

# Dequantize with ComfyUI-compatible layout
output_comfyui = gguf.dequantize_q4_0_comfyui(quantized, torch.float16)

# Benchmark
time_ms = gguf.benchmark(quantized, torch.float16)
print(f"Time: {time_ms:.3f} ms")
```

### Normalization

```python
import torch
from omni_xpu_kernel import norm

batch_size, hidden_size = 32, 4096
input = torch.randn(batch_size, hidden_size, device="xpu", dtype=torch.float16)
weight = torch.ones(hidden_size, device="xpu", dtype=torch.float16)
bias = torch.zeros(hidden_size, device="xpu", dtype=torch.float16)

# RMSNorm
output = norm.rms_norm(weight, input, eps=1e-6)

# LayerNorm
output = norm.layer_norm(input, weight, bias, eps=1e-5)

# LayerNorm without affine parameters
output = norm.layer_norm(input, eps=1e-5)
```

### Testing and Benchmarking

Run correctness tests:
```bash
pytest tests/ -v
```

Run performance benchmarks:
```bash
python tests/test_norm.py
```

## API Reference

### GGUF Module

#### `gguf.dequantize_q4_0(input, dtype=torch.float16)`
Dequantize Q4_0 tensor with interleaved output layout.

#### `gguf.dequantize_q4_0_comfyui(input, dtype=torch.float16)`
Dequantize Q4_0 tensor with sequential output layout (ComfyUI-compatible).

#### `gguf.benchmark(input, dtype, warmup_iters=10, bench_iters=100)`
Benchmark dequantization performance. Returns average time per iteration in milliseconds.

### Norm Module

#### `norm.rms_norm(weight, input, eps=1e-6)`
RMSNorm: `output = (input / sqrt(mean(input^2) + eps)) * weight`

#### `norm.layer_norm(input, weight=None, bias=None, eps=1e-5)`
LayerNorm: `output = ((input - mean) / sqrt(var + eps)) * weight + bias`

**Constraints:**
- Input must be 2D tensor [batch_size, hidden_size]
- hidden_size must be <= 8192 and divisible by 32
- Supports fp32, fp16, bf16

## Q4_0 Format

- Block size: 32 elements
- Block layout: `[scale (2 bytes, FP16)] [data (16 bytes, 32 x 4-bit)]`
- Total: 18 bytes per 32 elements

## License

Apache 2.0

