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
| `linear` | oneDNN FP8 weight-only GEMM |
| `fp8` | FP8 quantization, dequantization, and stochastic rounding |
| `gguf` | Q4_0, Q8_0, Q4_K, and Q6_K dequantization |
| `norm` | RMSNorm, LayerNorm, and fused normalization operations |
| `svdq` | SVDQuant W4A4 dequantization, INT4 GEMM, and post-processing |
| `int8` | INT8 quantization, linear, SwiGLU, and ConvRot operations |
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

The packaging layer recognizes Torch XPU minors 2.10, 2.11, and 2.12. Each
Torch/GPU pair still requires its own build and runtime validation; recognizing
a version is not a validation claim. The generated wheel uses a PEP 440 local
version such as:

```text
omni_xpu_kernel-0.1.0b9.dev0+torch211.bmg
omni_xpu_kernel-0.1.0b9.dev0+torch211.ptlh
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

`OMNI_XPU_DEVICE` selects both the AOT ISA and target-specific kernel policy.
Unknown values are rejected before compilation.

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
- A packaging-supported PyTorch XPU minor: 2.10.x, 2.11.x, or 2.12.x
- `onednn==2025.3.0` and `onednn-devel==2025.3.0` for the package's direct
  oneDNN calls on Linux
- Intel [`sycl-tla`](https://github.com/intel/sycl-tla) headers for the
  default Linux CUTE build

Torch and oneDNN are intentionally not listed as isolated build dependencies.
Install the target runtime first, then build with `--no-build-isolation` so the
compiler uses the same Torch headers and libraries as the final environment.

### Build through the llm-scaler image

The canonical Linux integration path is the llm-scaler Omni Docker build. It
pins the base environment, `sycl-tla`, oneDNN, Torch, Python, and the target
architecture:

```bash
cd /path/to/llm-scaler/omni

XPU_TARGET=bmg bash build.sh
# or
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
  torch==2.11.0+xpu torchvision==0.26.0+xpu torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
python -m pip install onednn==2025.3.0 onednn-devel==2025.3.0

git clone https://github.com/intel/sycl-tla.git /opt/sycl-tla
git -C /opt/sycl-tla checkout 2fc09973bfdf15755090fcb0e3b6ad236408a992
```

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

### oneDNN consistency

The native extensions call oneDNN directly. The `2025.3.0` pin belongs to
`omni_xpu_kernel`; it is not inherited from the selected Torch wheel. Using
headers from one oneDNN release with a library from another can produce
missing-symbol errors during import. The default Linux path therefore uses the
matched pip runtime and development packages shown above for every recognized
Torch minor. A new Torch minor is accepted only after rebuilding and testing
that complete combination.

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

# PTL-H wheels expose a separate dense-BHLD D120 capability.
if cute is not None and cute.supports_d120_bhld():
    output = cute.sdp_bhld_d120(q_bhld, k_bhld, v_bhld)
```

The CUTE D128 path accepts unmasked self-attention with `B=1`, equal Q/K/V
head counts, standard `1/sqrt(head_dim)` scaling, and FP16 or BF16 inputs.
Callers must retain their fallback for unsupported shapes and layouts.

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

The Linux build produces three extension components:

- `_C.so`: main AOT extension for normalization, quantization, GGUF, SVDQuant,
  rotary, and oneDNN-backed operations;
- `lgrf_sdp.so`: target-specific ESIMD attention sidecar;
- `cute_fmha_torch.so`: target-specific CUTLASS-SYCL attention sidecar.

`setup.py` derives one architecture macro from `OMNI_XPU_DEVICE` so wheel
metadata, core AOT ISA, sidecars, and compile-time kernel policy identify the
same target.

## License

Apache 2.0
