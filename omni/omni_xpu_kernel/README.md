# omni_xpu_kernel

High-performance Intel XPU kernels for PyTorch.

The Kernel version and the `intel/llm-scaler-omni` image version share the
single source in `omni_xpu_kernel/_version.py`. The current development version
is `0.1.0-b8-dev`; Python packaging normalizes it to `0.1.0b8.dev0` in wheel
metadata and filenames.

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

The LGRF and CUTE sidecars contain architecture-specific AOT images. Select the
target with `OMNI_XPU_DEVICE`; see [Building from Source](#building-from-source)
for the platform matrix and complete build procedure.

### cute — CUTLASS-SYCL Flash Attention

CUTLASS-SYCL fused Flash Attention with fp32 accumulation, AOT-compiled for
the target GPU. The currently validated domain is B=1, unmasked self-attention
with standard `1/sqrt(head_dim)` scaling, head dimension 128, equal Q/K/V head
counts, and fp16 or bf16 inputs in `[B, L, H, D]` layout.

```python
from omni_xpu_kernel import cute

if cute.is_available():
    output = cute.sdp(q, k, v)
```

The CUTE extension is Linux-only and required by default. `CUTLASS_SYCL_ROOT`
must point to a complete Intel `sycl-tla`/CUTLASS-SYCL source tree; otherwise
the build fails instead of silently omitting the default attention backend.

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

### fp8 — FP8 Quantization

Per-tensor quantization, dequantization, and seed-data-driven stochastic
rounding with Comfy Kitchen-compatible FP8 semantics.

```python
from omni_xpu_kernel import fp8

quantized = fp8.quantize_per_tensor(x, scale, torch.float8_e4m3fn)
restored = fp8.dequantize_per_tensor(quantized, scale, torch.bfloat16)
rounded = fp8.stochastic_rounding(x, rng, torch.float8_e4m3fn)
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

# LayerNorm followed by AdaLN scale/shift modulation
output = norm.fused_adaln(input, scale, shift, row_repeat=1, eps=1e-6)
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
packed_u4, u4_scales = svdq.quantize_act_uint4(nonnegative_activation, group_size=64)
restored_u4 = svdq.dequantize_u4(packed_u4, u4_scales)

# oneDNN INT4 GEMM (pre-convert weights once, then use preconverted variant)
packed_u4, scales_f16 = svdq.prepare_onednn_weights(packed, wscales)
output = svdq.onednn_int4_gemm_preconverted(act_f16, packed_u4, scales_f16)

# Fused f16->bf16 convert + add
svdq.fused_convert_add(out_bf16, result_f16, residual_bf16)
```

### int8 — INT8 Quantization and Linear

INT8 dynamic quantization and linear layer for ComfyUI INT8-ConvRot models.
Uses oneDNN s8×s8→s32 GEMM with ESIMD fused quantization and scale-back.

```python
from omni_xpu_kernel import int8

# Quantize weight offline
w_int8, w_scale = int8.quantize_int8_tensorwise(weight)

# INT8 linear (dynamic activation quantization + oneDNN GEMM + rescale)
output = int8.int8_linear(x_bf16, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

# Quantize once and reuse the activation across one or more Linear calls
x_int8, x_scale = int8.quantize_int8_rowwise(x_bf16)
output = int8.int8_linear_prequantized(
    x_int8, x_scale, w_int8, w_scale,
    bias=bias, out_dtype=torch.bfloat16,
)

# SwiGLU MLP: share the input quantization, then avoid the BF16 gate tensor
gate, up = int8.int8_linear_shared_input(
    x_bf16,
    w1_int8, w1_scale,
    w3_int8, w3_scale,
    out_dtype=torch.bfloat16,
)
gated_int8, gated_scale = int8.fused_silu_mul_quantize_rowwise(gate, up)
output = int8.int8_linear_prequantized(
    gated_int8, gated_scale, w2_int8, w2_scale,
    out_dtype=torch.bfloat16,
)

# ConvRot SwiGLU: remove the SiLU temporary, then reuse the XMX rotation
gate, up = int8.int8_linear_shared_input(
    x_bf16,
    w1_convrot_int8, w1_scale,
    w3_convrot_int8, w3_scale,
    out_dtype=torch.bfloat16,
    convrot=True, convrot_groupsize=256,
)
gated = int8.fused_silu_mul(gate, up)
del gate, up
rotated = int8.rotate_convrot(gated, group_size=256)
del gated
gated_int8, gated_scale = int8.quantize_int8_rowwise(rotated)
output = int8.int8_linear_prequantized(
    gated_int8, gated_scale, w2_convrot_int8, w2_scale,
    out_dtype=torch.bfloat16,
)

# With ConvRot (Hadamard rotation for improved accuracy)
output = int8.int8_linear(x, w_int8, w_scale, convrot=True, convrot_groupsize=256)

# Native ConvRot weight preparation using a cached Hadamard matrix multiplication
w_int8, w_scale = int8.quantize_int8_convrot_weight(weight, group_size=256)

# Cache management
int8.int8_cache_clear()
stats = int8.int8_cache_stats()  # {"hits": ..., "misses": ..., "size": ...}
```

### rotary — Rotary Position Embedding

Fused bf16->f32 + rotary rotation + f32->bf16 in a single ESIMD kernel.
Supports head_dim 64 and 128.

```python
from omni_xpu_kernel import rotary

output = rotary.rotary_emb(x, cos_cache, sin_cache, seq_len, heads)

# Comfy Kitchen adjacent-pair and split-half semantics
output = rotary.apply_kitchen_rope1(x, freqs_cis)
output = rotary.apply_kitchen_rope_split_half1(x, freqs_cis)
```

## Requirements

- Intel oneAPI DPC++/C++ Compiler (icpx)
- PyTorch with XPU support; the current validated container uses `2.11.0+xpu`
- Intel GPU: Arc B-series (BMG) or Panther Lake H (PTL-H)
- oneDNN 2025.3 pip runtime and development headers (for INT4/FP8/INT8 GEMM)
- Intel `sycl-tla`/CUTLASS-SYCL headers (for the default Linux CUTE FMHA)

## Building from Source

### Select the GPU target

`OMNI_XPU_DEVICE` controls the AOT ISA embedded in `lgrf_sdp.so` and
`cute_fmha_torch.so`. A wheel built for one target must not be installed on a
different GPU architecture.

| Platform | SYCL architecture check | `OMNI_XPU_DEVICE` | Status |
|---|---|---|---|
| Arc B-series / Battlemage | `intel_gpu_bmg_*` | `bmg` | Default and performance-tuned target |
| Panther Lake H / Arc B390 | `intel_gpu_ptl_h` | `ptl-h` | Build and correctness validated |

Identify the device before compiling instead of inferring the target from the
product name:

```bash
source /opt/intel/oneapi/setvars.sh --force
sycl-ls --verbose | grep -E 'Name|Architecture|Version|DeviceID'
```

For example, the validated PTL-H system reports:

```text
Name         : Intel(R) Arc(TM) B390 GPU
Version      : 30.0.4
DeviceID     : 45184
Architecture : intel_gpu_ptl_h
```

`ptl-h` and `ptl-u` are different AOT targets. The compiler accepting a target
only proves toolchain support; it does not prove that the generated image
matches the local GPU.

### Prepare CUTE / sycl-tla

The Linux build requires CUTE by default (`OMNI_XPU_REQUIRE_CUTE=1`).
`CUTLASS_SYCL_ROOT` must point to a complete Intel `sycl-tla` source tree with
the following directories:

```text
include/
tools/util/include/
examples/common/
applications/
```

The currently validated revision is:

```bash
git clone https://github.com/intel/sycl-tla.git /path/to/sycl-tla
git -C /path/to/sycl-tla checkout 2fc09973bfdf15755090fcb0e3b6ad236408a992
```

Do not update this pin without rebuilding and retesting CUTE FMHA; the templates
depend on a specific set of CUTE and Xe SYCL APIs.

### Reproducible Linux development container

The following example mounts the source tree, a writable output workspace, and
the read-only sycl-tla checkout. Set `SOURCE_DIR`, `WORKSPACE_DIR`, and
`SYCL_TLA_DIR` to host paths before running it.

```bash
export SOURCE_DIR=/path/to/llm-scaler/omni/omni_xpu_kernel
export WORKSPACE_DIR=/path/to/workspace
export SYCL_TLA_DIR=/path/to/sycl-tla

docker run -d \
  --name omni-xpu-kernel-devel \
  --device /dev/dri:/dev/dri \
  -v "$SOURCE_DIR:/src/omni_xpu_kernel" \
  -v "$WORKSPACE_DIR:/workspace" \
  -v "$SYCL_TLA_DIR:/opt/sycl-tla:ro" \
  intel/omix:0.1.0-devel-ubuntu24.04 \
  sleep infinity
```

Install the Python build environment and PyTorch 2.11 XPU:

```bash
docker exec omni-xpu-kernel-devel bash -lc '
  set -e
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev
  python3 -m venv /opt/venv
  /opt/venv/bin/python -m pip install -U pip
  /opt/venv/bin/python -m pip install \
    torch==2.11.0+xpu torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/xpu
  /opt/venv/bin/python -m pip install \
    onednn==2025.3.0 onednn-devel==2025.3.0
  /opt/venv/bin/python -m pip install wheel pytest numpy
'
```

The current resolver selects torchvision `0.26.0+xpu`, torchaudio
`2.11.0+xpu`, and `triton-xpu==3.7.0`. A `docker exec` shell does not reliably
inherit the oneAPI environment established for container PID 1, so source
`setvars.sh` explicitly in every build and test command.

### Build and install

Set `TARGET` to a value from the platform table. This example writes a wheel
instead of installing an editable tree so the exact artifact can be archived
and installed into another matching environment.

```bash
export TARGET=ptl-h  # use bmg on Arc B-series

docker exec -e TARGET="$TARGET" omni-xpu-kernel-devel bash -lc '
  set -e
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  source /opt/venv/bin/activate
  cd /src/omni_xpu_kernel
  mkdir -p /workspace/dist
  CUTLASS_SYCL_ROOT=/opt/sycl-tla \
  OMNI_XPU_REQUIRE_CUTE=1 \
  OMNI_XPU_DEVICE="$TARGET" \
  python -m pip wheel . --no-build-isolation --no-deps \
    -w /workspace/dist
'

docker exec omni-xpu-kernel-devel bash -lc '
  set -e
  source /opt/venv/bin/activate
  python -m pip install --force-reinstall --no-deps \
    /workspace/dist/omni_xpu_kernel-*.whl
'
```

`--no-build-isolation` is required because the compiler needs the headers and
libraries from the installed XPU PyTorch wheel. A normal `_C` build currently
compiles 17 C++ translation units in one serial `icpx` invocation. The build
backend captures compiler output, so several minutes without terminal output
can be normal while a `clang` child process is consuming CPU.

For a direct editable build outside the container, use the same environment:

```bash
source /opt/intel/oneapi/setvars.sh --force

CUTLASS_SYCL_ROOT=/path/to/sycl-tla \
OMNI_XPU_DEVICE=bmg \
pip install -e . --no-build-isolation

CUTLASS_SYCL_ROOT=/path/to/sycl-tla \
OMNI_XPU_DEVICE=ptl-h \
pip install -e . --no-build-isolation

# Explicit Linux core-only opt-out:
OMNI_XPU_REQUIRE_CUTE=0 \
OMNI_XPU_DEVICE=bmg \
pip install -e . --no-build-isolation
```

On Windows, CUTE is not built and the core-only path is used; see
[WHL_BUILD_INSTALL.md](WHL_BUILD_INSTALL.md).

### Verify the installed wheel

The Linux wheel should contain three native artifacts:

```bash
python -m zipfile -l /workspace/dist/omni_xpu_kernel-*.whl \
  | grep -E '(_C|lgrf_sdp|cute_fmha_torch).*\.so'
```

```text
omni_xpu_kernel/_C.cpython-312-x86_64-linux-gnu.so
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cpython-312-x86_64-linux-gnu.so
omni_xpu_kernel/cute/cute_fmha_torch.cpython-312-x86_64-linux-gnu.so
```

Run the import check outside the source directory so an unbuilt source package
cannot shadow the installed wheel:

```bash
cd /tmp
python -c '
import torch, omni_xpu_kernel as ok
from omni_xpu_kernel import cute
print(torch.__version__, torch.xpu.get_device_name(0))
print(ok.__version__, ok.is_available(), cute.is_available())
'
```

### oneDNN header/library consistency

PyTorch XPU wheels include oneDNN headers for their internal build but do not
ship an exported `libdnnl.so.3`. The extension directly calls oneDNN, so Linux
builds use the matched official pip packages by default:

```bash
pip install onednn==2025.3.0 onednn-devel==2025.3.0
```

Both packages contain oneDNN 3.9.1. Mixing the newer headers bundled under
`torch/include` with an older external library can produce an import failure
such as:

```text
undefined symbol: dnnl_primitive_attr_set_zero_points_v2
```

The build validates the header and runtime versions, passes the selected
`libdnnl` file directly to the linker, and keeps that include directory before
`torch/include`. It also removes only duplicate oneDNN entries injected through
`CPATH`, `C_INCLUDE_PATH`, or `CPLUS_INCLUDE_PATH`.

The three native extensions use `$ORIGIN`-relative ELF search paths to the
active Python prefix instead of embedding the build venv or `/opt/intel` path.
Check the result with:

```bash
nm -D /path/to/_C.so | grep dnnl_primitive_attr_set_zero_points
ldd /path/to/_C.so | grep dnnl
readelf -d /path/to/_C.so | grep -E 'RPATH|RUNPATH'
```

The validated build references `dnnl_primitive_attr_set_zero_points` and loads
`libdnnl.so.3` from the Python prefix. A non-pip development installation can
still be selected explicitly by setting both `ONEDNN_INCLUDE` and
`ONEDNN_LIB`; this is not the relocatable wheel path.

### Platform-specific notes

#### Battlemage

- `bmg` remains the default AOT target.
- LGRF `ConfigBMG` is the currently performance-tuned attention configuration.
- The PTL-specific oneDNN workaround described below is guarded by runtime
  architecture and does not change the BMG path.

#### Panther Lake H

- Use `ptl-h` only after `sycl-ls --verbose` reports `intel_gpu_ptl_h`.
- LGRF and CUTE compile to PTL-H AOT images, but their tile/WG policy currently
  reuses the BMG/Xe2 configuration; successful compilation is not a claim of
  PTL-specific performance tuning.
- oneDNN 3.9 cannot create the FP16 `M=4096, K=4096, N=4096` JIT GEMM primitive
  used as a chunk of the `N=12288` FP8 workflow shape. The implementation uses
  an `N=2048` chunk only on `intel_gpu_ptl_h`; BF16 and non-PTL paths retain the
  existing chunk selection.

The PTL-H configuration validated on 2026-07-16 was:

| Component | Version / result |
|---|---|
| Source baseline | `origin/main@1380ca7` plus PTL build fixes |
| GPU | Arc B390, `intel_gpu_ptl_h`, IP 30.0.4 |
| Container | `intel/omix:0.1.0-devel-ubuntu24.04` |
| Compiler | oneAPI DPC++/C++ 2025.3.3 |
| PyTorch | `2.11.0+xpu` |
| oneDNN | pip `onednn==2025.3.0` / oneDNN 3.9.1 |
| sycl-tla | `2fc09973bfdf15755090fcb0e3b6ad236408a992` |
| Tests | `469 passed, 2 skipped`, no failures |

The validated wheel is
`omni_xpu_kernel-0.1.0b8.dev0-cp312-cp312-linux_x86_64.whl`, SHA256
`71eb2c49e2a798ffdf00b389de756896076965b9347af9b4517084533c62f09b`.

Compiler metadata contains two LGRF images and three CUTE images. LGRF D128 and
D64 kernels reserve 256 GRF with 32 KiB and 16 KiB SLM respectively; the CUTE
FP16/BF16 main kernels reserve 256 GRF. No image reports compiler-visible
scratch, spill, or `per_thread_memory_buffers`. Inspect another build with:

```bash
TOOL=/opt/intel/oneapi/compiler/2025.3/bin/compiler/clang-offload-extract
$TOOL --stem=/tmp/lgrf /path/to/lgrf_sdp.so
$TOOL --stem=/tmp/cute /path/to/cute_fmha_torch.so

for image in /tmp/lgrf.* /tmp/cute.*; do
  readelf -p .ze_info "$image" \
    | grep -E 'name:|grf_count:|slm_size:|scratch|spill|per_thread_memory_buffers'
done
```

`grf_count=256` denotes the doubleGRF reservation and is not itself evidence of
spill. The PTL oneDNN failure happens during JIT register-bundle allocation,
before execution, and is handled by chunking rather than accepting spill.

#### Support boundary

- The documented build targets are BMG and PTL-H only. PVC has never been
  validated by this project and is not a planned support target.
- `sdp_config.h` contains legacy PVC/LNL and B770 configuration placeholders;
  their presence is not a build-support claim.
- Never distribute a BMG or PTL-H wheel as a generic cross-platform binary;
  rebuild the AOT sidecars for the matching target.

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

### Attention Kernel Compilation

The SDP Flash Attention kernel uses ESIMD with doubleGRF and is compiled as a
separate sidecar shared library (`lgrf_sdp.so`). AOT compilation targets a
specific GPU via `-device <target>` (default: bmg).

On Linux, the default build requires a valid `CUTLASS_SYCL_ROOT` and produces
the CUTLASS-SYCL attention sidecar (`cute_fmha_torch.so`). Set
`OMNI_XPU_REQUIRE_CUTE=0` only for an explicit core-only build. The remaining
native operations are built into the main `_C` extension.

`sdp_config.h` contains the active `ConfigBMG` configuration plus legacy
PVC/LNL configuration placeholders. Only the BMG path is performance-tuned;
the PTL-H build currently reuses its tile/WG policy. The legacy definitions do
not imply that those platforms are supported.

### Build System

The package builds multiple extension modules:
- `_C.so` — Main extension (norm, gguf, svdq, rotary, sdp loader, fp8, int8)
- `lgrf_sdp.so` — SDP ESIMD sidecar (AOT, doubleGRF)
- `cute_fmha_torch.so` — CUTLASS-SYCL FMHA sidecar (Linux, AOT, required by default)

## License

Apache 2.0
