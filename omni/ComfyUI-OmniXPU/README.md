# ComfyUI-OmniXPU

Thin Intel XPU integration for upstream ComfyUI.

The runtime is deliberately split into three layers:

1. `omni_xpu_kernel` supplies native XPU kernels.
2. `comfy_kitchen` owns generic operator APIs, capability checks, dispatch,
   and safe eager fallback.
3. `ComfyUI-OmniXPU` only adapts ComfyUI call sites that do not yet expose a
   Kitchen entry point, plus a small set of opt-in legacy correctness fixes.

No workflow or model-pipeline replacement is required.

## Ownership

| Layer | Current responsibility |
|---|---|
| Kitchen XPU backend | INT8/QTensor operations, FP8 QDQ and stochastic rounding, SVDQuant, AdaLN, four RoPE APIs, and ConvRot |
| ComfyUI adapter | Attention routing, LayerNorm/RMSNorm class integration, the remaining FP8 model/factory bridge, and fused Lumina/Z-Image INT8 FFN wiring |
| Legacy fix | Global `F.interpolate` and `torch.median`/`torch.nanmedian` workarounds; disabled by default |

RoPE, generic INT8 linear dispatch, and the old FP8 negative-zero wrapper are
not registered by this custom node. Duplicating those registrations here can
override Kitchen's constraints and fallback policy.

## Install

The node is bundled with the `llm-scaler-omni` ComfyUI image. It requires:

- an `omni_xpu_kernel` wheel built for the active XPU target and Torch minor;
- the pinned `comfy_kitchen` XPU integration;
- upstream ComfyUI.

If an Intel XPU is unavailable, initialization is skipped.

## Components and switches

Adapters are enabled by default and always retain the original ComfyUI route
for unsupported inputs:

```bash
OMNIXPU_ENABLE=0            # Disable every custom-node component
OMNIXPU_ATTENTION=0         # Disable the attention adapter
OMNIXPU_NORM=0              # Disable the norm adapter
OMNIXPU_FP8_GEMM=0          # Disable the temporary FP8 model/factory adapter
OMNIXPU_INT8_FFN=0          # Disable fused Lumina/Z-Image INT8 FFN wiring
```

Validated sub-routes can be disabled independently:

```bash
OMNI_ATTN_BACKEND=auto      # auto, cute, esimd, or torch; Windows defaults to torch
OMNIXPU_NONCONTIG_RMSNORM=0
OMNIXPU_H120_RMSNORM=0
OMNIXPU_KREA2_RMSNORM=0
```

For diagnostics, the per-call CUTE output scan can be enabled explicitly. It
is disabled by default because validated CUTE routes accumulate in FP32 and a
full output scan adds a shape-proportional temporary allocation. Explicit
ESIMD FP16 routing retains its overflow scan regardless of this setting.

```bash
OMNIXPU_VALIDATE_ATTENTION_OUTPUT=1
```

The two global workarounds are opt-in:

```bash
OMNIXPU_INTERPOLATE_FIX=1
OMNIXPU_MEDIAN_FIX=1
OMNIXPU_MEDIAN_STRICT_INDICES=1
```

`OMNIXPU_MEDIAN_STRICT_INDICES=1` reproduces the exact tie-break indices. The
median workaround was only verified on BMG with Torch 2.10 and remains
disabled by default on other configurations.

## Debugging and diagnostics

Kernel-only tracing:

```bash
OMNIXPU_DEBUG=1 python main.py
```

Dispatch decisions and fallback reasons:

```bash
OMNIXPU_DEBUG_VERBOSE=1 python main.py
```

Set tracing variables before startup. The **OmniXPU Status** node reports:

- GPU and `omni_xpu_kernel` capabilities;
- each component's kind (`adapter` or `legacy_fix`) and apply status;
- attention and fused INT8 FFN routing counters.

Kitchen backend ownership can be inspected independently:

```bash
python -c 'import comfy_kitchen as ck; print(ck.list_backends()["xpu"])'
```
