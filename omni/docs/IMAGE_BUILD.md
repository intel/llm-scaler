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
| `COMFYUI_REPOSITORY` | ComfyUI source repository | pinned in `build.sh` |
| `COMFYUI_COMMIT` | ComfyUI source revision | pinned in `build.sh` |
| `COMFYUI_VERSION` | Expected ComfyUI version | pinned in `build.sh` |
| `COMFYUI_FRONTEND_VERSION` | ComfyUI frontend package version | pinned in `build.sh` |
| `COMFYUI_WORKFLOW_TEMPLATES_VERSION` | Official workflow-template bundle version | pinned in `build.sh` |
| `COMFYUI_MANAGER_VERSION` | Integrated Node Manager package version | pinned in `build.sh` |
| `COMFY_KITCHEN_REPOSITORY` | Kitchen source repository | pinned in `build.sh` |
| `COMFY_KITCHEN_COMMIT` | Kitchen source revision | pinned in `build.sh` |
| `COMFY_KITCHEN_VERSION` | Expected Kitchen wheel version | pinned in `build.sh` |
| `COMFY_AIMDO_REPOSITORY` | AIMDO source repository identity | pinned in `build.sh` |
| `COMFY_AIMDO_COMMIT` | AIMDO source revision | pinned in `build.sh` |
| `COMFY_AIMDO_VERSION` | Expected AIMDO wheel version | pinned in `build.sh` |
| `COMFY_GGUF_REPOSITORY` | GGUF custom-node source repository | pinned in `build.sh` |
| `COMFY_GGUF_COMMIT` | GGUF custom-node source revision | pinned in `build.sh` |
| `COMFY_NUNCHAKU_REPOSITORY` | Combined Nunchaku custom-node/runtime repository | pinned in `build.sh` |
| `COMFY_NUNCHAKU_COMMIT` | Combined Nunchaku source revision | pinned in `build.sh` |
| `COMFY_NUNCHAKU_VERSION` | Expected combined distribution version | pinned in `build.sh` |

ComfyUI repository, commit, and version must be updated together. Kitchen and
AIMDO repository, commit, and version pins are independently checked against
their package metadata. GGUF repository and commit must be updated together.
The same rule applies to the combined Nunchaku repository, commit, and
distribution version. The kernel source is copied from
`omni/omni_xpu_kernel` in the current llm-scaler checkout.

The AIMDO revision must be reachable from its pinned remote. The build fetches
and checks out that exact full commit before compiling the Level Zero backend;
branch names are not used as image identity.

The focused image installs the version-pinned integrated `comfyui-manager` package and
does not clone the legacy Manager custom node. Frontend, workflow templates,
and Manager are explicit build inputs; the final image also records a complete
`pip freeze --all` dependency snapshot at
`/llm/manifests/comfyui-python-freeze.txt`.

## Focused-image build graph

The focused Dockerfile separates the frequently changed native projects:

| Stage | Contents |
|---|---|
| `os-base`, `python-base` | OS, Torch XPU, and oneDNN dependencies |
| `comfyui-deps` | Pinned ComfyUI and third-party custom nodes |
| `sycl-tla` | Pinned native headers |
| `kernel-wheel` | Target-specific `omni_xpu_kernel` wheel |
| `kitchen-wheel` | Pinned Comfy Kitchen wheel |
| `aimdo-wheel` | Pinned AIMDO wheel with the Level Zero XPU backend |
| `builder-comfyui` | Wheel installation and local ComfyUI integration |
| `runtime-comfyui` | Final labels, environment, and runtime metadata |

BuildKit is enabled by `build.sh`. Normal incremental builds should preserve
the cache. The `kernel-wheel`, `kitchen-wheel`, and `aimdo-wheel` targets are
diagnostics; image acceptance must use the default final target.

## Source and artifact identity

For focused images, `build.sh` records the full llm-scaler Git revision and
whether `omni/` had uncommitted changes. The final image also records:

- image version and flavor;
- selected XPU target;
- ComfyUI version and commit;
- ComfyUI frontend, workflow-template, and integrated Manager versions;
- Kitchen version and commit;
- AIMDO version and commit;
- GGUF custom-node commit;
- combined Nunchaku custom-node/runtime version and commit;
- SYCL-TLA commit.

Build from a clean commit before release acceptance. A device-less Docker
build can verify packaging, but it cannot prove that Torch or Kitchen can use
the destination XPU.

## Acceptance

Run the validator inside the final container:

```bash
IMAGE=intel/llm-scaler-omni:0.1.0-b9-dev1-comfyui-bmg

sudo docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    python /llm/tools/validate_comfyui_image.py
```

The release check requires a real XPU and clean source metadata. The
`--allow-no-xpu` and `--allow-dirty-source` switches are intended only for
explicit diagnostics and do not replace device-backed acceptance. The same
validator also requires exact Kitchen, AIMDO, GGUF, and combined Nunchaku
source revisions; the installed AIMDO XPU backend; the
GGUF/SentencePiece/Protobuf imports; the bundled `nunchaku_torch` runtime; and
the managed Kitchen GGUF/W4A16 capabilities.
