# ComfyUI-OmniXPU

Intel XPU acceleration for upstream ComfyUI via [omni_xpu_kernel](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel).

All optimizations are applied transparently at startup — no workflow changes needed.

## Install

Bundled with the `llm-scaler-omni` Docker image. No manual installation needed.

Requires `omni_xpu_kernel` installed. Without it the node loads silently with no patches applied.

## What it does

| Patch | Target |
|-------|--------|
| ESIMD Flash Attention | `optimized_attention` |
| ESIMD RoPE | `_apply_rope1` / `apply_rope1` |
| ESIMD LayerNorm/RMSNorm | `LayerNorm.forward` / `RMSNorm.forward` / `rms_norm()` |
| FP8 GEMM | `fp8_linear` / `mixed_precision_ops` |
| INT8 Linear | `comfy_kitchen::int8_linear` (oneDNN s8 GEMM) |
| FP8 Negative Zero Fix | `manual_stochastic_round_to_float8` |
| Interpolate Fix | `F.interpolate` |
| Median Fix | `torch.median` / `torch.nanmedian` (XPU dim-reduction) |

## Environment Variables

All patches enabled by default. Disable with `=0`:

```bash
OMNIXPU_ENABLE=0            # Master switch — disable everything
OMNIXPU_ATTENTION=0         # Disable ESIMD Flash Attention only
OMNIXPU_ROPE=0              # Disable ESIMD RoPE only
OMNIXPU_NORM=0              # Disable ESIMD LayerNorm/RMSNorm only
OMNIXPU_FP8_GEMM=0          # Disable FP8 GEMM only
OMNIXPU_INT8=0              # Disable INT8 Linear only
OMNIXPU_FP8_NEG_ZERO_FIX=0  # Disable FP8 negative zero fix only
OMNIXPU_INTERPOLATE_FIX=0   # Disable interpolate workaround only
OMNIXPU_MEDIAN_FIX=0        # Disable median workaround only
```

`OMNIXPU_MEDIAN_STRICT_INDICES=1` makes the median workaround reproduce
`torch.median`'s exact tie-break indices (values are always bit-exact).

> **Note:** the XPU median slowdown this works around has only been verified on Intel Arc B60/B70 with torch 2.10. It should be re-checked on other hardware or torch versions before relying on it there.

## Diagnostics

Add the **OmniXPU Status** node to any workflow to see:

```
=== ComfyUI-OmniXPU Status ===
  GPU: Intel(R) Arc(TM) B580 Graphics (11605 MB)
  omni_xpu_kernel: 0.1.0
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
[OmniXPU] omni_xpu_kernel 0.1.0 — available: sdp, norm, rotary, linear_fp8
[OmniXPU] interpolate_fix: applied
[OmniXPU] median_fix: applied
[OmniXPU] fp8_neg_zero_fix: applied
[OmniXPU] norm: applied
[OmniXPU] rope: applied
[OmniXPU] fp8_gemm: applied
[OmniXPU] attention: applied
[OmniXPU] INT8: registered XPU impl for comfy_kitchen::int8_linear
[OmniXPU] int8: applied
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
- `omni_xpu_kernel` >= 0.1.0
- `comfy_kitchen` >= 0.2.8 (for INT8 custom ops)
