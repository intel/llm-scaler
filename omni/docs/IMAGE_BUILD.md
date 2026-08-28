# Omni image build and acceptance

This document describes the source build implemented by `omni/build.sh`. The
default output is the ComfyUI-focused image.

## Build inputs

Run builds from `omni/`:

```bash
OMNI_IMAGE_REPOSITORY=llm-scaler-omni \
XPU_TARGET=bmg bash build.sh
```

The current image supports Intel Arc B-series/Battlemage GPUs. `build.sh`
assigns the `-comfyui-bmg` suffix to local builds because their native binaries
are AOT-compiled for BMG. Published BMG releases use
`intel/llm-scaler-omni:<version>` without a flavor or target suffix, but
`0.2.0-b2` is currently available only as a source build. The command above
produces `llm-scaler-omni:0.2.0-b2-comfyui-bmg`.

The supported environment overrides are:

| Variable | Purpose | Default |
|---|---|---|
| `XPU_TARGET` | Native GPU build target | `bmg` |
| `OMNI_IMAGE_REPOSITORY` | Local image repository | `intel/llm-scaler-omni` |
| `OMNI_BASE_IMAGE` | OMIX development base | `intel/omix:0.3.0-devel-ubuntu24.04` at the digest pinned in `build.sh` |
| `OMNI_TORCH_VERSION` | PyTorch XPU wheel | `2.13.0+xpu` |
| `OMNI_TORCHVISION_VERSION` | torchvision XPU wheel | `0.28.0+xpu` |
| `OMNI_TORCHAUDIO_VERSION` | ComfyUI audio compatibility wheel | `2.11.0+xpu` |
| `OMNI_ONEDNN_VERSION` | Matched oneDNN runtime/development wheels | `2026.0.0` (oneDNN `3.11.2`) |
| `OMNI_ONEDNN_SOURCE_REPOSITORY` | oneDNN source repository | official oneDNN repository pinned in `build.sh` |
| `OMNI_ONEDNN_SOURCE_COMMIT` | oneDNN release source revision | oneDNN `v3.11.2` commit pinned in `build.sh` |
| `OMNI_ONEDNN_PATCH_SHA256` | Reviewed BF16/INT4 compatibility patch identity | SHA256 pinned in `build.sh` |
| `MAX_JOBS` | Native build parallelism | `8` |
| `COMFYUI_REPOSITORY` | ComfyUI source repository | pinned in `build.sh` |
| `COMFYUI_COMMIT` | ComfyUI source revision | pinned in `build.sh` |
| `COMFYUI_VERSION` | Expected ComfyUI version | pinned in `build.sh` |
| `COMFYUI_FRONTEND_VERSION` | ComfyUI frontend package version | pinned in `build.sh` |
| `COMFYUI_WORKFLOW_TEMPLATES_VERSION` | Official workflow-template bundle version | pinned in `build.sh` |
| `COMFYUI_MANAGER_VERSION` | Integrated Node Manager package version | pinned in `build.sh` |
| `COMFY_KITCHEN_REPOSITORY` | Kitchen XPU provider source repository | pinned in `build.sh` |
| `COMFY_KITCHEN_COMMIT` | Kitchen XPU provider source revision | pinned in `build.sh` |
| `COMFY_KITCHEN_VERSION` | Official Kitchen and matching provider version | pinned in `build.sh` |
| `COMFY_AIMDO_REPOSITORY` | AIMDO XPU provider source repository | pinned in `build.sh` |
| `COMFY_AIMDO_COMMIT` | AIMDO XPU provider source revision | pinned in `build.sh` |
| `COMFY_AIMDO_VERSION` | Official AIMDO and matching provider version | pinned in `build.sh` |
| `COMFY_GGUF_REPOSITORY` | GGUF custom-node source repository | pinned in `build.sh` |
| `COMFY_GGUF_COMMIT` | GGUF custom-node source revision | pinned in `build.sh` |
| `COMFY_NUNCHAKU_REPOSITORY` | Combined Nunchaku custom-node/runtime repository | pinned in `build.sh` |
| `COMFY_NUNCHAKU_COMMIT` | Combined Nunchaku source revision | pinned in `build.sh` |
| `COMFY_NUNCHAKU_VERSION` | Expected combined distribution version | pinned in `build.sh` |
| `COMFY_SOL_ATTN_REPOSITORY` | Sol-Attn custom-node repository | pinned in `build.sh` |
| `COMFY_SOL_ATTN_COMMIT` | Sol-Attn custom-node source revision | pinned in `build.sh` |

ComfyUI repository, commit, and version must be updated together. Kitchen and
AIMDO official distribution versions and matching XPU provider source pins are
independently checked against their package and provider manifests. GGUF
repository and commit must be updated together.
The same rule applies to the combined Nunchaku repository, commit, and
distribution version. The Sol-Attn custom node is pinned independently from
the kernel package and contains no image-local XPU build path. The kernel source is copied from
`omni/omni_xpu_kernel` in the current llm-scaler checkout.

Each provider revision must be reachable from its pinned remote. The build
fetches and checks out those exact full commits before constructing the private
provider wheels; branch names are not used as image identity. The normal XPU
wheel is only an intermediate input. Its canonical package tree is re-homed
below the distinct `comfy-kitchen-xpu-runtime` or
`comfy-aimdo-xpu-runtime` distribution, so it cannot overwrite the official
top-level package.

The AIMDO Unified Runtime hook is compiled against
`/opt/venv/include/unified-runtime/ur_api.h`, which belongs to the
Torch-matched runtime family used by the final image. OMIX 0.3 supplies the
oneAPI 2026.1 compiler, but its compiler include tree is not used as the AIMDO
hook ABI contract. The runtime entrypoint likewise places the venv Unified
Runtime loader before the OMIX build-toolchain libraries. The final image also
exports that directory as `UR_INCLUDE_DIR` so AIMDO's maintained native-hook
helper build uses the same explicit ABI contract during installed-image
validation.

This focused image is single-XPU and does not copy the legacy
`libsycl-native-*.spv` multi-XPU blobs used by older platform images. OMIX 0.3
and oneAPI 2026.1 do not ship those files; the card-specific BMG binaries are
embedded in the AOT kernel wheel instead.

The focused image installs the version-pinned integrated `comfyui-manager` package and
does not clone the legacy Manager custom node. Frontend, workflow templates,
and Manager are explicit build inputs; the final image also records a complete
`pip freeze --all` dependency snapshot at
`/llm/manifests/comfyui-python-freeze.txt`.

The image includes `/llm/configs/comfyui_host_models.yaml` for registering a
read-only host model tree mounted at `/models/host`, plus external runtime data
directories below `/data`. Keeping model, input, output, and user mounts
outside the ComfyUI checkout preserves upstream's tracked files, so
in-container ComfyUI upgrades retain a clean Git worktree.

The official `comfy-kitchen` and `comfy-aimdo` packages remain installed as
ordinary ComfyUI dependencies. Matching XPU runtime providers use private
package names and are discovered by ComfyUI-OmniXPU during the normal custom
node prestartup phase. The image retains the two exact provider wheels and
their SHA256 values in `/llm/manifests/xpu-runtime-providers.sha256`. No
launcher, Python entry point, or global `sitecustomize` hook is modified.
The runtime also exports a narrow pip constraint file for Torch, torchvision,
torchaudio, `omni_xpu_kernel`, and the two provider distributions. Normal
ComfyUI dependency upgrades can move the official packages, but cannot
silently replace the Torch ABI or provider artifacts.

The official XPU index does not provide a Torch-2.13-matched `torchaudio`
wheel, while ComfyUI requires torchaudio and the maintained workflows include
audio. The focused image therefore keeps the existing `2.11.0+xpu` wheel as
an explicit compatibility exception. That XPU wheel has no exact Torch
dependency in its package metadata; the installed-image validator still
requires its exact version and performs an XPU resample with shape and finite
output checks. The canonical audio workflows remain the milestone-level gate.

The `onednn` and `onednn-devel` wheels provide the 2026.0 runtime package
layout and SYCL/Unified Runtime dependencies expected by Torch 2.13. The
focused build replaces only `libdnnl` with the exact official oneDNN 3.11.2
release source plus the checked-in, SHA256-pinned two-site BF16/INT4
dequantization compatibility patch. The patch restores the behavior removed
by upstream commit `0d99c32b03614d6943b993974f91d419e3c3a0f6`; it does not
downgrade the rest of the Torch/oneAPI stack. The build records source,
patch, and installed-library identities in
`/llm/manifests/onednn-runtime.env`.

OMIX 0.3 supplies the 2026.1 compiler toolchain, while Torch 2.13 pins its
packaged SYCL and Unified Runtime libraries to 2026.0. The final image keeps
OMIX initialization for compiler and tool discovery, then restores
`/opt/venv/lib` and the discovered Torch library directory to the front of
`LD_LIBRARY_PATH` through `entrypoints/omix_torch_runtime.sh`. This prevents a process
from mixing the Torch wheel's `libsycl` with OMIX's newer `libur_loader`; the
installed-image validator checks the libraries actually mapped by the process.

## Focused-image build graph

The focused Dockerfile separates the frequently changed native projects:

| Stage | Contents |
|---|---|
| `os-base`, `python-base` | OS, Torch XPU, and oneDNN dependencies |
| `comfyui-deps` | Pinned ComfyUI and third-party custom nodes |
| `sycl-tla` | Pinned native headers |
| `kernel-wheel` | Target-specific `omni_xpu_kernel` wheel |
| `kitchen-wheel` | Pinned Kitchen XPU source wheel and co-installable provider wheel |
| `aimdo-wheel` | Pinned AIMDO Level Zero source wheel and co-installable provider wheel |
| `builder-comfyui` | Wheel installation and local ComfyUI integration |
| `runtime-comfyui` | Final labels, environment, and runtime metadata |

BuildKit is enabled by `build.sh`. Normal incremental builds should preserve
the cache. The `kernel-wheel`, `kitchen-wheel`, and `aimdo-wheel` targets are
diagnostics; image acceptance must use the default final target.

## Source and artifact identity

For focused images, `build.sh` records the full llm-scaler Git revision and
whether `omni/` had uncommitted changes. The final image also records:

- image version and flavor;
- exact OMIX base reference, Torch, torchvision, torchaudio, and oneDNN package versions;
- exact oneDNN source repository and revision, compatibility-patch SHA256,
  and installed `libdnnl.so.3.11` SHA256;
- selected XPU target;
- ComfyUI version and commit;
- ComfyUI frontend, workflow-template, and integrated Manager versions;
- official Kitchen version and Kitchen XPU provider commit;
- official AIMDO version and AIMDO XPU provider commit;
- retained XPU provider wheel SHA256 values;
- GGUF custom-node commit;
- combined Nunchaku custom-node/runtime version and commit;
- Sol-Attn custom-node repository and commit;
- SYCL-TLA commit.

Build from a clean commit before release acceptance. A device-less Docker
build can verify packaging, but it cannot prove that Torch or Kitchen can use
the destination XPU.

## Acceptance

Run the validator inside the final container:

```bash
IMAGE=llm-scaler-omni:0.2.0-b2-comfyui-bmg

sudo docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    python /llm/tools/validate_comfyui_image.py
```

The acceptance check requires a real XPU and clean source metadata. The
`--allow-no-xpu` and `--allow-dirty-source` switches are intended only for
explicit diagnostics and do not replace device-backed acceptance. The same
validator also requires exact Kitchen, AIMDO, GGUF, and combined Nunchaku
provider source revisions; both official distributions; both disjoint provider
distributions; conditional routing; the installed AIMDO XPU backend and Linux
allocator takeover after a verified unwind of official AIMDO's reversible
pre-device state; the
GGUF/SentencePiece/Protobuf imports; the bundled `nunchaku_torch` runtime; and
the managed Kitchen GGUF/W4A16 capabilities. It additionally fails closed if
the oneDNN manifest, checked-in patch, installed runtime DSO, or
`libdnnl.so.3` symlink does not match the identities recorded by the image.
It also requires the exact Sol-Attn custom-node checkout, its explicit XPU
gate, and a loadable packaged `omni-cute` Sol-Attn DSO; the image does not
install Triton or build a custom-node-local XPU sidecar.

This source-built image supports BMG.

Run the upgrade lifecycle check in a disposable container, passing an exact
official ComfyUI revision accepted for the milestone:

```bash
docker run --rm "$IMAGE" \
    python /llm/tools/validate_comfyui_upgrade.py \
        --comfyui-revision <full-40-character-commit> \
        --upgrade-kitchen-version <resolved-official-version> \
        --upgrade-aimdo-version <resolved-official-version>
```

This fetches that ComfyUI revision, installs its normal Python requirements,
upgrades to the explicitly resolved official Kitchen and AIMDO versions, and
verifies that compatible providers activate while incompatible providers are
safely left unwired. It then restores the image's accepted official versions
and requires both XPU providers to activate. Every provider-owned file is
hashed across each phase. The test mutates the container's writable layer and
therefore does not replace clean-image acceptance.
