# ComfyUI-OmniXPU

Intel XPU acceleration for upstream ComfyUI via [omni_xpu_kernel](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel).

All optimizations are applied transparently at startup — no workflow changes needed.

## Install

Bundled with the `llm-scaler-omni` Docker image. No manual installation needed.

Requires `omni_xpu_kernel` installed. Without it the node loads silently with no patches applied.

## What it does

| Patch | Target |
|-------|--------|
| Auto-routed cute/ESIMD Attention | `optimized_attention` |
| ESIMD RoPE | `_apply_rope1` / `apply_rope1` / `apply_rope` (flux.math dual-tensor) |
| ESIMD LayerNorm/RMSNorm | `LayerNorm.forward` / `RMSNorm.forward` / `rms_norm()` |
| FP8 GEMM | `fp8_linear` / `mixed_precision_ops` |
| INT8 Linear | `comfy_kitchen::int8_linear` (oneDNN s8 GEMM) |
| Fused INT8 SwiGLU FFN | Eligible Lumina/Z-Image `FeedForward` blocks |
| FP8 Negative Zero Fix | `manual_stochastic_round_to_float8` |
| Interpolate Fix | `F.interpolate` |
| Median Fix | `torch.median` / `torch.nanmedian` (XPU dim-reduction) |

## Environment Variables

All patches enabled by default. Disable with `=0`:

```bash
OMNIXPU_ENABLE=0            # Master switch — disable everything
OMNIXPU_ATTENTION=0         # Disable the XPU attention patch only
OMNIXPU_ROPE=0              # Disable ESIMD RoPE only
OMNIXPU_NORM=0              # Disable ESIMD LayerNorm/RMSNorm only
OMNIXPU_KREA2_RMSNORM=0     # Disable the Krea2-specific local RMSNorm hook only
OMNIXPU_FP8_GEMM=0          # Disable FP8 GEMM only
OMNIXPU_INT8=0              # Disable all INT8 routes
OMNIXPU_INT8_FFN=0          # Disable fused Lumina/Z-Image INT8 FFN only
OMNIXPU_FP8_NEG_ZERO_FIX=0  # Disable FP8 negative zero fix only
OMNIXPU_INTERPOLATE_FIX=0   # Disable interpolate workaround only
OMNIXPU_MEDIAN_FIX=0        # Disable median workaround only
```

Set `OMNIXPU_DEBUG=1` before starting ComfyUI to log the XPU kernels that are
actually selected. Wrapper calls that fall back to another implementation are
not reported as Omni XPU kernel executions. Tensor shapes, dtypes, and devices
are included without printing values or synchronizing the device:

```bash
OMNIXPU_DEBUG=1 python main.py
```

Example:

```text
[OmniXPU DEBUG] stage=kernel op=int8_linear backend=omni_xpu tensors=x(shape=(1, 4160, 3840), dtype=torch.bfloat16, device=xpu:0), weight(shape=(10240, 3840), dtype=torch.int8, device=xpu:0), weight_scale(shape=(10240, 1), dtype=torch.float32, device=xpu:0)
[OmniXPU DEBUG] stage=kernel op=int8_swiglu_mlp backend=omni_xpu route=shared_up+fused_swiglu+convrot+quant+prequant_down up_convrot=True down_convrot=True tensors=input(shape=(1, 4160, 3840), dtype=torch.bfloat16, device=xpu:0), ...
```

For dispatch and fallback analysis, use the verbose flag instead. It is a
superset of normal debug logging, so setting both flags is unnecessary:

```bash
OMNIXPU_DEBUG_VERBOSE=1 python main.py
```

Verbose output adds the high-level dispatch stage, including the quantization
format and layout where available:

```text
[OmniXPU DEBUG] stage=dispatch op=mixed_precision.Linear quant_format=int8_tensorwise layout=TensorWiseINT8Layout tensors=input(shape=(1, 4160, 3840), dtype=torch.bfloat16, device=xpu:0)
[OmniXPU DEBUG] stage=kernel op=int8_linear backend=omni_xpu tensors=x(shape=(1, 4160, 3840), dtype=torch.bfloat16, device=xpu:0), weight(shape=(10240, 3840), dtype=torch.int8, device=xpu:0), weight_scale(shape=(10240, 1), dtype=torch.float32, device=xpu:0)
```

Set either flag before ComfyUI startup. Changing tracing flags in a running
process is unsupported; restart ComfyUI after changing them.

The fused INT8 FFN route shares activation quantization and optional ConvRot
between Lumina `w1` and `w3`. For an unrotated `w2`, it fuses
`SiLU(w1(x)) * w3(x)` directly into rowwise INT8 storage. For a ConvRot `w2`,
including Z-Image INT8 ConvRot, it writes one fused floating SwiGLU result,
reuses the existing XMX ConvRot, quantizes it, and feeds the result to the
prequantized projection. This avoids the separate floating SiLU temporary
without replacing the faster XMX rotation with a slower custom transform.

The route is selected only for resident `TensorWiseINT8Layout` XPU weights
with matching dtypes and supported ConvRot settings. LoRA or other weight
functions, offloaded weights, bias, training, transposed weights,
full-precision overrides, and unsupported shapes retain the original ComfyUI
forward path. Use `OMNIXPU_DEBUG_VERBOSE=1` to see the fallback reason.

The current PTL-H wheel guards the native dynamic rowwise INT8 quantizer. On
that target, `comfy_kitchen::int8_linear` uses the Comfy Kitchen eager backend
and the fused Lumina INT8 FFN route is skipped. Prequantized oneDNN INT8 APIs
remain available because they do not enter the guarded quantizer. BMG keeps
the native dispatcher and fused FFN routes. The status node reports native and
fallback call counts and the reason `ptl_h_dynamic_rowwise_quantize`.

Attention routing is selected independently:

```bash
OMNI_ATTN_BACKEND=auto   # default: cute d128 self-attn, then ESIMD, then PyTorch
OMNI_ATTN_BACKEND=cute   # force cute where supported; otherwise PyTorch
OMNI_ATTN_BACKEND=esimd  # force ESIMD where supported; otherwise PyTorch
OMNI_ATTN_BACKEND=torch  # keep the original PyTorch attention path
```

With `auto`, CUTE handles its validated B=1, unmasked, standard-scale d128
self-attention domain. Supported d64 and cross-attention calls use ESIMD.
Masked attention, other batch sizes or head dimensions, GQA, custom scaling,
and any unsupported shape fall back to the original PyTorch implementation.
Explicit `cute` and `esimd` select only that fused backend and still use the
safe PyTorch fallback outside its supported domain.

`OMNIXPU_MEDIAN_STRICT_INDICES=1` makes the median workaround reproduce
`torch.median`'s exact tie-break indices (values are always bit-exact).

> **Note:** the XPU median slowdown this works around has only been verified on Intel Arc B60/B70 with torch 2.10. It should be re-checked on other hardware or torch versions before relying on it there.

## Diagnostics

Add the **OmniXPU Status** node to any workflow to see:

```
=== ComfyUI-OmniXPU Status ===
  GPU: Intel(R) Arc(TM) B580 Graphics (11605 MB)
  omni_xpu_kernel: 0.1.0b8.dev0+torch211
    available: sdp, norm, rotary, linear_fp8

  [+] interpolate_fix: applied
  [+] median_fix: applied
  [+] fp8_neg_zero_fix: applied
  [+] norm: applied
  [+] rope: applied
  [+] fp8_gemm: applied
  [+] attention: applied
```

## Startup Log

When loaded successfully, ComfyUI logs:

```
[OmniXPU] omni_xpu_kernel 0.1.0b8.dev0+torch211 — available: sdp, norm, rotary, linear_fp8
[OmniXPU] interpolate_fix: applied
[OmniXPU] median_fix: applied
[OmniXPU] fp8_neg_zero_fix: applied
[OmniXPU] norm: applied
[OmniXPU] rope: applied
[OmniXPU] fp8_gemm: applied
[OmniXPU] attention[cute]: rebound 45 by-value imports across sys.modules
[OmniXPU] attention: applied
[OmniXPU] INT8: registered XPU impl for comfy_kitchen::int8_linear
[OmniXPU] int8: applied
[OmniXPU] INT8 FFN: routed eligible Lumina FeedForward through fused kernels
[OmniXPU] int8_ffn: applied
```

## How it works

The node monkey-patches ComfyUI internals at import time. Each patch:

1. Checks if the corresponding `omni_xpu_kernel` submodule is available (via centralized probe)
2. Verifies the target function/class exists in the current ComfyUI version
3. Wraps the original with an XPU-accelerated version that falls back to the original for non-XPU tensors or unsupported shapes
4. Records status for the diagnostics node

No ComfyUI core files are modified. Works with unmodified upstream ComfyUI.

## Compatibility

- ComfyUI >= 0.18.x (>= 0.27.0 for INT8 ConvRot model support)
- PyTorch >= 2.7 with XPU support
- An `omni_xpu_kernel` wheel matching the installed Torch minor; the current
  Omni image uses `0.1.0b8.dev0+torch211` with Torch 2.11
- `comfy_kitchen` >= 0.2.8 (for INT8 custom ops)
