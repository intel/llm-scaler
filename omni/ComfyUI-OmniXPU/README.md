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
| Memory adapter | Cached whole-LoRA model budgets plus optional DynamicVRAM per-layer XPU staging measurements |
| SeedVR2 capacity | Guarded Ada broadcast plus byte-bounded RMSNorm, SwiGLU, and window-attention materialization |
| SeedVR2 native adapters | Validated BMG FP16 GroupNorm and causal-prefix cat-pad routing |
| Large-video preprocessing | Source-guarded, bounded CPU materialization for PIL Lanczos resize, SeedVR input padding, and XPU VAE input staging |
| Legacy fix | Global `F.interpolate` and `torch.median`/`torch.nanmedian` workarounds; disabled by default |

RoPE, generic INT8 linear dispatch, and the old FP8 negative-zero wrapper are
not registered by this custom node. Duplicating those registrations here can
override Kitchen's constraints and fallback policy.

## Install

The node is bundled with the `llm-scaler-omni` ComfyUI image. It requires:

- an `omni_xpu_kernel` wheel built for the active XPU target and Torch minor;
- the official `comfy-kitchen` and `comfy-aimdo` distributions;
- matching `comfy-kitchen-xpu-runtime` and `comfy-aimdo-xpu-runtime`
  provider wheels;
- upstream ComfyUI.

If an Intel XPU is unavailable, initialization is skipped.

## Official packages and XPU providers

The provider distributions use private top-level package names and do not own
any `comfy_kitchen/*` or `comfy_aimdo/*` file. The official packages can
therefore be reinstalled or upgraded without overwriting the XPU runtime.
ComfyUI's launcher and Python entry point are unchanged.

During the normal custom-node prestartup phase, OmniXPU discovers lightweight
provider metadata before importing PyTorch. It verifies the official package
version, exact Torch XPU build, platform and image target, source revision,
source-wheel hash, and every vendored runtime file hash. Kitchen is then routed
only when its canonical package is first imported.

AIMDO takeover additionally requires explicit DynamicVRAM enablement, an
unimported PyTorch runtime on Linux, and an official `comfy_aimdo.control`
module with no device context or allocator. ComfyUI calls official AIMDO init
before custom-node prestartup; on an XPU Torch build that can leave only its
pre-device CUDA DSO state live. OmniXPU permits exactly that reversible state,
calls the official public `deinit()`, and verifies that it returned to a
pristine module before takeover. Any device or allocator state is rejected.
OmniXPU then executes the provider control implementation in that same module
object so the reference imported by ComfyUI remains valid. A reversible
provider failure restores the deinitialized official module. A failure after
provider allocator or native state becomes live stops startup because
allocator ownership cannot be rolled back safely.

Provider routing defaults to `auto` and can be controlled without changing the
launcher:

```bash
OMNIXPU_PROVIDER_BOOTSTRAP=off       # Keep every official runtime
OMNIXPU_PROVIDER_BOOTSTRAP=auto      # Use each compatible XPU provider
OMNIXPU_PROVIDER_BOOTSTRAP=required  # Fail unless both providers activate
```

After an official package upgrade, an incompatible provider is skipped in
`auto` mode instead of being forced into a new API contract. Upgrade the
corresponding provider wheel to restore XPU routing.

## Components and switches

Adapters are enabled by default and always retain the original ComfyUI route
for unsupported inputs:

```bash
OMNIXPU_ENABLE=0            # Disable every custom-node component
OMNIXPU_ATTENTION=0         # Disable the attention adapter
OMNIXPU_NORM=0              # Disable the norm adapter
OMNIXPU_FP8_GEMM=0          # Disable the temporary FP8 model/factory adapter
OMNIXPU_INT8_FFN=0          # Disable fused Lumina/Z-Image INT8 FFN wiring
OMNIXPU_DYNAMIC_VRAM_BOUNDARY_TRIM=0  # Disable Windows XPU model-boundary trim
OMNIXPU_LORA_MEMORY=0       # Disable cached whole-LoRA budgets and staging logs
OMNIXPU_SEEDVR_ADA_RESHAPE=0  # Disable the guarded SeedVR2 Ada reshape patch
OMNIXPU_SEEDVR_CAPACITY=0     # Disable bounded SeedVR2 activation scheduling
OMNIXPU_SEEDVR_CAT_PAD=0      # Disable validated BMG causal-prefix cat-pad routing
OMNIXPU_LARGE_VIDEO_PREPROCESS=0  # Disable bounded large-video CPU preprocessing
```

On Windows XPU, the boundary trim turns an unmet DynamicVRAM minimum-memory
budget into an explicit partial VBAR reclaim before model loading. It preserves
loaded models and is enabled by default; the environment variable above is the
A/B-test escape hatch.

Validated sub-routes can be disabled independently:

```bash
OMNI_ATTN_BACKEND=auto      # auto, cute, esimd, or torch; Windows defaults to torch
OMNIXPU_NONCONTIG_RMSNORM=0
OMNIXPU_H120_RMSNORM=0
OMNIXPU_KREA2_RMSNORM=0
OMNIXPU_SEEDVR_GROUPNORM=0
```

On Windows, CUTE is never selected implicitly. A wheel built explicitly with
`OMNI_XPU_REQUIRE_CUTE=1` still uses PyTorch SDPA by default; set
`OMNI_ATTN_BACKEND=cute` before launching ComfyUI to enable the CUTE routes.

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

LoRA weights are measured once when the LoRA node executes. Unique tensor sizes
are cached in a `ModelPatcher` attachment, inherited by clones, accumulated for
stacked LoRAs, and added to both `memory_required` and an explicitly supplied
`minimum_memory_required`. The base model's `model_size()` semantics stay
unchanged. Model loads read the cached attachment instead of rescanning patches.
DynamicVRAM layer scanning is disabled by default. To diagnose every LoRA
staging operation, including its XPU state and any failure, enable:

```bash
OMNIXPU_LORA_MEMORY_TRACE=1 python main.py
```

Set tracing variables before startup. The **OmniXPU Status** node reports:

- GPU and `omni_xpu_kernel` capabilities;
- runtime-provider activation, skip, and rejection reasons;
- each component's kind (`adapter`, `compatibility_patch`, or `legacy_fix`)
  and apply status;
- attention and fused INT8 FFN routing counters.

Kitchen backend ownership can be inspected independently:

```bash
python -c 'import comfy_kitchen as ck; print(ck.list_backends()["xpu"])'
```
