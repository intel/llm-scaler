# omni_xpu_kernel

Native Intel XPU kernels used by llm-scaler image and video workloads.

The package combines SYCL/ESIMD kernels, oneDNN-backed quantized GEMM, and a
CUTLASS-SYCL attention backend behind a small PyTorch API. Linux wheels are
compiled for one Torch minor and one GPU architecture; they are not portable
across those native ABI boundaries.

## Package contents

| Module | Main functionality |
|---|---|
| `sdp` | ESIMD scaled dot-product attention |
| `cute` | CUTLASS-SYCL fused attention |
| `cute.sdp_bhld_d128` | BMG batched/rectangular D128 BHLD attention |
| `cute.sdp_minimax_h3_vae_d64` | Structural BMG MiniMax H3 VideoVAE D64 tile attention |
| `cute.sdp_wan22_cross` | Exact BMG Wan 2.2 14B T2V Turbo cross-attention |
| `cute.sol_attn` | BMG sparse Sol-Attn for BF16 BTHD D128 tensors |
| `linear` | oneDNN FP8 weight-only GEMM |
| `fp8` | FP8 quantization, dequantization, and stochastic rounding |
| `gguf` | Q4_0, Q4_1, Q8_0, Q4_K, and Q6_K dequantization |
| `norm` | RMSNorm, LayerNorm, and fused normalization operations |
| `svdq` | SVDQuant W4A4 dequantization, INT4 GEMM, and post-processing |
| `int8` | INT8 quantization, linear, fused GELU/SwiGLU, and ConvRot operations |
| `rotary` | Rotary embedding and Comfy Kitchen-compatible RoPE operations |

The exact native symbols available in an installed artifact can be inspected
without relying on a hard-coded capability list:

```python
import omni_xpu_kernel as omni

print(omni.native_capabilities())
```

## Artifact identity

The package and `intel/llm-scaler-omni` image versions share the source in
[`omni_xpu_kernel/_version.py`](omni_xpu_kernel/_version.py). A source build
derives its native identity from the active Torch installation and
`OMNI_XPU_DEVICE`.

The packaging layer recognizes Torch XPU minors 2.10, 2.11, 2.12, and 2.13. Each
Torch/GPU pair still requires its own build and runtime validation; recognizing
a version is not a validation claim. The generated wheel uses a PEP 440 local
version such as:

```text
omni_xpu_kernel-0.2.0b2+torch213.bmg
omni_xpu_kernel-0.2.0b2+torch213.ptlh
```

Build and install a different wheel for every Torch/GPU pair. The wheel
metadata pins the exact public Torch version used at build time.

After installation, these values come from the wheel's own metadata:

```python
import omni_xpu_kernel as omni

print(omni.__version__)
print(omni.__torch_version__)
print(omni.__xpu_target__)
```

On Linux, `core_aot_target()` reads the architecture marker embedded in the
loaded `_C` extension. It must equal `__xpu_target__`; an empty or different
value indicates an old, JIT-only, or stale native artifact.

```python
assert omni.is_available()
assert omni.core_aot_target() == omni.__xpu_target__
```

## Build targets

`OMNI_XPU_DEVICE` selects the AOT ISA and architecture-level policy. Unknown
values are rejected before compilation. One BMG wheel contains both B60 and
B70 kernel profiles and selects between them from the exact runtime PCI Device
ID:

| PCI Device ID | Runtime BMG profile |
|---|---|
| `0xE210`, `0xE211` | `b60` |
| `0xE223` | `b70` |
| other BMG ID | `generic-bmg` (the shipped B70-compatible defaults) |

Use `omni_xpu_kernel.device.info(index)` to inspect the detected ID, selected
profile, and concrete policy values.

| GPU architecture | `sycl-ls --verbose` architecture | `OMNI_XPU_DEVICE` |
|---|---|---|
| Intel Arc B-series / Battlemage | `intel_gpu_bmg_*` | `bmg` |
| Intel Panther Lake H | `intel_gpu_ptl_h` | `ptl-h` |

Identify the device before building:

```bash
source /opt/intel/oneapi/setvars.sh --force
sycl-ls --verbose | grep -E 'Name|Architecture|Version|DeviceID'
```

Do not infer the AOT target only from a product name. In particular, `ptl-h`
and `ptl-u` are different compiler targets, and a wheel built for BMG must not
be installed on PTL-H.

## Build requirements

- Python 3.9 or newer development environment
- Intel oneAPI DPC++/C++ Compiler (`icpx`)
- PyTorch XPU 2.13.x for the current validated build
- `onednn==2026.0.0` and `onednn-devel==2026.0.0` (oneDNN 3.11.2) for the
  package's direct oneDNN calls on Linux
- A matched oneAPI 2026.0 oneDNN 3.11.2 development installation on Windows.
  The build vendors its `dnnl.dll` and redistribution notices into the wheel.
- Intel [`sycl-tla`](https://github.com/intel/sycl-tla) headers for the
  default Linux CUTE build or explicit experimental Windows BMG CUTE build

Torch and oneDNN are intentionally not listed as isolated build dependencies.
Install the target runtime first, then build with `--no-build-isolation` so the
compiler uses the same Torch headers and libraries as the final environment.

### Build through the llm-scaler image

The canonical Linux integration path is the llm-scaler Omni Docker build. It
pins the base environment, `sycl-tla`, oneDNN, Torch, Python, and the target
architecture:

```bash
cd /path/to/llm-scaler/omni

OMNI_IMAGE_REPOSITORY=llm-scaler-omni \
XPU_TARGET=bmg bash build.sh
# or
OMNI_IMAGE_REPOSITORY=llm-scaler-omni \
XPU_TARGET=ptl-h bash build.sh
```

See the [Omni image documentation](../README.md) for image tags,
runtime setup, and acceptance checks.

### Build a standalone Linux wheel

Prepare a matching Torch XPU environment and the pinned `sycl-tla` source:

```bash
python3 -m venv /opt/venv
source /opt/venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install \
  torch==2.13.0+xpu torchvision==0.28.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
python -m pip install onednn==2026.0.0 onednn-devel==2026.0.0

git clone https://github.com/intel/sycl-tla.git /opt/sycl-tla
git -C /opt/sycl-tla checkout 2fc09973bfdf15755090fcb0e3b6ad236408a992
```

There is no Torch-2.13-matched `torchaudio` wheel on the official XPU index.
It is not required by `omni_xpu_kernel`; the complete ComfyUI image separately
keeps its existing `2.11.0+xpu` audio wheel as a validated workflow
compatibility exception.

Build the wheel from this directory:

```bash
source /opt/intel/oneapi/setvars.sh --force

CUTLASS_SYCL_ROOT=/opt/sycl-tla \
OMNI_XPU_REQUIRE_CUTE=1 \
OMNI_XPU_DEVICE=bmg \
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Replace `bmg` with `ptl-h` only when building on the matching target. CUTE is
required by default on Linux; the build fails if `CUTLASS_SYCL_ROOT` is absent
or incomplete. `OMNI_XPU_REQUIRE_CUTE=0` is an explicit core-only build and
must not be mistaken for the default image artifact.

For Windows build and installation details, see
[`WHL_BUILD_INSTALL.md`](WHL_BUILD_INSTALL.md).

Windows wheels remain core-only by default even when a sycl-tla checkout is
present. Set both `CUTLASS_SYCL_ROOT=<clean-sycl-tla-v0.8-checkout>` and
`OMNI_XPU_REQUIRE_CUTE=1` to include the experimental BMG CUTE `.pyd`. Runtime
routing is a separate opt-in: ComfyUI continues to use PyTorch SDPA unless
`OMNI_ATTN_BACKEND=cute` is set before launch.

### oneDNN consistency

The native extensions call oneDNN directly. The Linux `2026.0.0` pin belongs to
`omni_xpu_kernel`; it is not inherited from the selected Torch wheel. Using
headers from one oneDNN release with a library from another can produce
missing-symbol errors during import. The default Linux path therefore uses the
matched pip runtime and development packages shown above for every recognized
Torch minor. A new Torch minor is accepted only after rebuilding and testing
that complete combination.

Torch 2.13 pins its Intel runtime packages to 2026.0.0. oneDNN 2026.0.0 is the
matching package release: later 2026.0.x oneDNN wheels require 2026.1 runtimes
and cannot satisfy this exact Torch environment. The Windows Torch 2.13 build
uses the matching oneAPI 2026.0 oneDNN 3.11.2 headers, import library, and
runtime.

For a non-pip development installation, set both variables to the same oneDNN
installation:

```bash
ONEDNN_INCLUDE=/path/to/include \
ONEDNN_LIB=/path/to/lib \
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Setting only one variable is rejected.

## Verify an installed wheel

Install the wheel without resolving a different Torch build:

```bash
python -m pip install --force-reinstall --no-deps dist/omni_xpu_kernel-*.whl
```

Run the import check outside the source directory so the local package cannot
shadow the installed wheel:

```bash
cd /tmp
python - <<'PY'
import torch
import omni_xpu_kernel as omni

print("torch:", torch.__version__)
print("device:", torch.xpu.get_device_name(0))
print("package:", omni.__version__)
print("built torch:", omni.__torch_version__)
print("metadata target:", omni.__xpu_target__)
print("core AOT target:", omni.core_aot_target())
print("runtime kernel profile:", omni.device.info(0))
print("available:", omni.is_available())

assert omni.is_available()
assert omni.core_aot_target() == omni.__xpu_target__
PY
```

A default Linux wheel contains:

```text
omni_xpu_kernel/_C.cpython-312-x86_64-linux-gnu.so
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cpython-312-x86_64-linux-gnu.so
omni_xpu_kernel/cute/cute_fmha_torch.cpython-312-x86_64-linux-gnu.so
```

## API examples

### Attention

```python
from omni_xpu_kernel import cute, sdp

# q, k, v use [B, L, H, D] layout.
output = sdp.sdp(q, k, v)

if cute is not None and cute.is_available():
    output = cute.sdp(q, k, v)

# PTL-H and BMG wheels expose a separate dense-BHLD D120 capability.
if cute is not None and cute.supports_d120_bhld():
    output = cute.sdp_bhld_d120(q_bhld, k_bhld, v_bhld)

# BMG wheels expose batched self/cross attention for dense packed-BHLD,
# BLHD-backed BHLD, or the B1/H56 MiniMax H3 QKV-backed D128 layout.
if cute is not None and cute.supports_d128_bhld():
    output = cute.sdp_bhld_d128(q_bhld, k_bhld, v_bhld)

# BMG wheels expose the structural MiniMax H3 VideoVAE FP16 D64 tile family.
if cute is not None and cute.supports_minimax_h3_vae_d64():
    output = cute.sdp_minimax_h3_vae_d64(q_bhld, k_bhld, v_bhld)

# BMG wheels expose the exact official Wan 2.2 14B T2V Turbo 720p
# tuned cross-attention contract separately from the general BHLD API.
if cute is not None and cute.supports_wan22_cross():
    output = cute.sdp_wan22_cross(q_blhd, k_blhd, v_blhd)

# BMG builds expose sparse Sol-Attn for the validated BF16 BTHD D128
# self-attention contract. Routing thresholds remain explicit call policy.
if cute is not None and cute.supports_sol_attn():
    output = cute.sol_attn(q_bthd, k_bthd, v_bthd, tau=1.3)
```

The legacy BLHD `cute.sdp` entry point accepts unmasked self-attention with
`B=1`. The BMG `cute.sdp_bhld_d128` entry point accepts positive batch, head,
query-length, and key/value-length dimensions; matching Q/K/V batch, head,
dtype, and head dimension; dense packed-BHLD, BLHD-backed BHLD, or the B1/H56
MiniMax H3 QKV-backed layout; D128; standard `1/sqrt(head_dim)` scaling; and
FP16 or BF16. Neither entry point
accepts masks, causal mode, GQA, or custom scaling. API capability does not
imply that every shape is faster than PyTorch; callers must retain a
performance-qualified fallback policy.

`sdp_wan22_cross` remains an exact MMA-K16 specialization. It accepts only
dense FP16 BMG tensors with Q `[1, 75600, 40, 128]` and K/V
`[1, 512, 40, 128]`; other structurally supported BHLD contracts use
`sdp_bhld_d128`.

`sdp_minimax_h3_vae_d64` accepts the MiniMax H3 VideoVAE tile family: FP16
Q/K/V `[1, 32, S, 64]`, where `S` varies with the decoder's temporal and
spatial tile extent. Q/K use the runtime-derived `H*D` sequence stride and V
retains the three-wide QKV projection stride. Other D64 layouts remain with
the caller's fallback.

`sol_attn` is BMG-only and accepts matching XPU BF16 Q/K/V in BTHD layout,
with non-empty sequence length, D128, and contiguous head dimension. It does
not accept masks, causal mode, GQA, or cross-attention. The approximation
policy is controlled by `tau`, `sink_blocks`, and `sink_q`; callers must not
substitute it for dense attention unless their model has selected Sol-Attn.

### Quantized linear operations

```python
from omni_xpu_kernel import int8, linear

output = linear.onednn_w8a16_fp8(
    activation, fp8_weight, weight_scales, bias=bias
)

w_int8, w_scale = int8.quantize_int8_tensorwise(weight)
output = int8.int8_linear(
    activation,
    w_int8,
    w_scale,
    bias=bias,
    out_dtype=activation.dtype,
)

# Fold a concatenated [gate | up] SwiGLU into rowwise quantization. The
# activated width is half the input width and must match the INT8 weight.
output = int8.int8_linear(
    gate_up,
    w_int8,
    w_scale,
    out_dtype=gate_up.dtype,
    input_act="swiglu",
)

# GELU-tanh is fused directly into rowwise INT8 quantization for the validated
# profitable BMG row range when ConvRot is disabled. Larger rows retain the
# faster materialized XPU route.
output = int8.int8_linear(
    activation,
    w_int8,
    w_scale,
    out_dtype=activation.dtype,
    input_act="gelu_tanh",
)

x_int8, x_scale = int8.quantize_int8_rowwise(activation)
output = int8.int8_linear_prequantized(
    x_int8,
    x_scale,
    w_int8,
    w_scale,
    out_dtype=activation.dtype,
)
```

### FP8 and GGUF

```python
from omni_xpu_kernel import fp8, gguf

quantized = fp8.quantize_per_tensor(x, scale, torch.float8_e4m3fn)
restored = fp8.dequantize_per_tensor(
    quantized, scale, torch.bfloat16
)
rounded = fp8.stochastic_rounding(
    x, rng, torch.float8_e4m3fn
)

q4 = gguf.dequantize_q4_0(packed_q4, torch.float16)
q8 = gguf.dequantize_q8_0(packed_q8, torch.float16)
outputs = gguf.dequantize_batch(
    [packed_q4, packed_q8],
    ["q4_0", "q8_0"],
    torch.float16,
)
```

### Normalization and SVDQuant

```python
from omni_xpu_kernel import norm, svdq

output = norm.rms_norm(weight, x, eps=1e-6)
output = norm.layer_norm(x, weight=weight, bias=bias, eps=1e-5)
norm.fused_add_rms_norm(x, residual, weight, eps=1e-6)
output = norm.fused_rms_adaln(x_2d, scale_2d, shift_2d, row_repeat, eps=1e-6)

unpacked = svdq.unpack_int4(packed_weight, signed=True)
dequantized = svdq.dequantize_w4(
    packed_weight, scales, out_dtype=torch.bfloat16
)
prepared_weight, prepared_scales = svdq.prepare_onednn_weights(
    packed_weight, scales
)
output = svdq.onednn_int4_gemm_preconverted(
    activation, prepared_weight, prepared_scales
)
```

### Comfy Kitchen RoPE

```python
from omni_xpu_kernel import rotary

output = rotary.apply_kitchen_rope1(x, freqs_cis)
output = rotary.apply_kitchen_rope_split_half1(x, freqs_cis)

# The fused pair API supports strided packed-QKV views, in-place output, and a
# partial split-half rotary prefix while RMSNorm still covers the full head.
q, k = rotary.rms_kitchen_rope_split_half_(
    q,
    k,
    freqs_cis,
    q_scale,
    k_scale,
    epsilon=1e-5,
    rot_dim=96,
)

if rotary.kitchen_rope_fast_supported(x, freqs_cis):
    output = rotary.apply_kitchen_rope1(x, freqs_cis)
```

Callers should use the capability query before selecting a specialized native
route and preserve the established PyTorch fallback.

## Debug logging

Native logging is disabled by default. Enable all modules or a comma-separated
subset with `OMNI_XPU_DEBUG`:

```bash
OMNI_XPU_DEBUG=1 python your_script.py
OMNI_XPU_DEBUG=sdp,fp8 python your_script.py
```

Messages use this format:

```text
[omni_xpu::<module>] <message>
```

`OMNI_FP8_DEBUG=1` remains available for compatibility.

## Tests and benchmarks

Run source tests from this directory in a matching XPU build environment:

```bash
python -m pytest tests
```

Benchmarks live outside `tests/` so pytest cannot collect performance
workloads. Run the available groups explicitly:

```bash
python -m benchmarks.run_all --fp8
python -m benchmarks.run_all --gguf
python -m benchmarks.run_all --norm
python -m benchmarks.run_all --sdp
```

See [`benchmarks/README.md`](benchmarks/README.md) for workload-specific
programs and measurement boundaries.

Benchmark results are device-, driver-, Torch-, shape-, and power-state
specific. Do not treat a number from one target as validation for another.

## Native layout

The default Linux build produces three extension components:

- `_C.so`: main AOT extension for normalization, quantization, GGUF, SVDQuant,
  rotary, and oneDNN-backed operations;
- `lgrf_sdp.so`: target-specific ESIMD attention sidecar;
- `cute_fmha_torch.so`: target-specific CUTLASS-SYCL attention sidecar.

The default Windows build contains `_C.pyd` and `lgrf_sdp.pyd`. An explicitly
enabled BMG CUTE build adds `cute_fmha_torch.pyd`, including the packaged
Sol-Attn operators; it does not enable either runtime route automatically.

`setup.py` derives one architecture macro from `OMNI_XPU_DEVICE` so wheel
metadata, core AOT ISA, and sidecars identify the same target. BMG core and
CUTE components query the exact runtime Device ID and share the B60/B70 policy
table.

## License

Apache 2.0.
