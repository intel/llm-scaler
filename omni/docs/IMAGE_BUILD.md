# Omni image build and acceptance

This document describes the source build implemented by `omni/build.sh`. The
default output is the ComfyUI-focused image.

## Build inputs

Run builds from `omni/`:

```bash
XPU_TARGET=bmg bash build.sh
XPU_TARGET=ptl-h bash build.sh
```

The supported environment overrides are:

| Variable | Purpose | Default |
|---|---|---|
| `XPU_TARGET` | Native GPU build target | `bmg` |
| `OMNI_IMAGE_REPOSITORY` | Local image repository | `intel/llm-scaler-omni` |
| `OMNI_BASE_IMAGE` | OMIX development base | `intel/omix:0.1.0-devel-ubuntu24.04` |
| `MAX_JOBS` | Native build parallelism | `8` |
| `COMFY_KITCHEN_REPOSITORY` | Kitchen source repository | pinned in `build.sh` |
| `COMFY_KITCHEN_COMMIT` | Kitchen source revision | pinned in `build.sh` |
| `COMFY_KITCHEN_VERSION` | Expected Kitchen wheel version | pinned in `build.sh` |

Kitchen repository, commit, and version must be updated together. The kernel
source is copied from `omni/omni_xpu_kernel` in the current llm-scaler
checkout.

## Focused-image build graph

The focused Dockerfile separates the frequently changed native projects:

| Stage | Contents |
|---|---|
| `os-base`, `python-base` | OS, Torch XPU, and oneDNN dependencies |
| `comfyui-deps` | Pinned ComfyUI and third-party custom nodes |
| `sycl-tla` | Pinned native headers |
| `kernel-wheel` | Target-specific `omni_xpu_kernel` wheel |
| `kitchen-wheel` | Pinned Comfy Kitchen wheel |
| `builder-comfyui` | Wheel installation and local ComfyUI integration |
| `runtime-comfyui` | Final labels, environment, and runtime metadata |

BuildKit is enabled by `build.sh`. Normal incremental builds should preserve
the cache. The `kernel-wheel` and `kitchen-wheel` targets are diagnostics;
image acceptance must use the default final target.

## Source and artifact identity

For focused images, `build.sh` records the full llm-scaler Git revision and
whether `omni/` had uncommitted changes. The final image also records:

- image version and flavor;
- selected XPU target;
- Kitchen version and commit;
- SYCL-TLA commit.

Build from a clean commit before release acceptance. A device-less Docker
build can verify packaging, but it cannot prove that Torch or Kitchen can use
the destination XPU.

## Acceptance

Run the validator inside the final container:

```bash
IMAGE=intel/llm-scaler-omni:0.1.0-b9-dev-comfyui-bmg

docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    python /llm/tools/validate_comfyui_image.py
```

The release check requires a real XPU and clean source metadata. The
`--allow-no-xpu` and `--allow-dirty-source` switches are intended only for
explicit diagnostics and do not replace device-backed acceptance.
