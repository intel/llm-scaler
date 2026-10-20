# LLM Scaler Omni

LLM Scaler Omni provides Intel XPU images for generative media workloads. The
default image is a single-XPU ComfyUI environment with target-specific
`omni_xpu_kernel` binaries, the XPU-enabled Comfy Kitchen backend, and a thin
ComfyUI integration layer.

> [!IMPORTANT]
> The current `0.2.0-b3` beta preview is available only as a source build. The
> published ComfyUI image is `intel/llm-scaler-omni:0.2.0-b2`. This source build is
> experimental and focused on single-XPU ComfyUI workloads, and does not replace
> the broader b8 image. For SGLang Diffusion, Raylight, or other multi-XPU
> scenarios, use the published
> [`intel/llm-scaler-omni:0.1.0-b8`](https://github.com/intel/llm-scaler/releases/tag/omni-0.1.0-b8)
> image.

## Getting Started with the Omni Docker Image

Build from the `omni` directory:

```bash
cd omni

# Intel Arc B-series / Battlemage
OMNI_IMAGE_REPOSITORY=llm-scaler-omni \
XPU_TARGET=bmg bash build.sh
```

The current image supports Intel Arc B-series/Battlemage GPUs. Its native wheel
is AOT-compiled for BMG, and the source-build command above assigns this tag to
local images:

```text
llm-scaler-omni:<version>-comfyui-bmg
```

Published BMG releases use the version as the image tag:

```text
intel/llm-scaler-omni:<version>
```

The published BMG image is
[`intel/llm-scaler-omni:0.2.0-b2`](https://hub.docker.com/layers/intel/llm-scaler-omni/0.2.0-b2/images/sha256-cc127efc23f8d6033d00e9d759468241ef478562ba63a56bb1dd8e83bbab46e3).
The source-build command above produces the development tag
`llm-scaler-omni:0.2.0-b3-comfyui-bmg`; `0.2.0-b3` has not been published.
The component pins below describe this source build. Published tags are listed
in [Releases](../Releases.md). The development version is defined in
`omni_xpu_kernel/omni_xpu_kernel/_version.py`.

### Validate the image

Run the supplied acceptance script against the final image with the GPU device
exposed:

```bash
IMAGE=llm-scaler-omni:0.2.0-b3-comfyui-bmg

sudo docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    /llm/entrypoints/validate_comfyui_image.sh
```

The check verifies package identity, the Torch ABI, native AOT target, clean
source provenance, dependencies, XPU availability, and required Kitchen
capabilities. This source-built image supports BMG.

### Run ComfyUI

Mount the existing ComfyUI model directory rather than copying models into the
image:

```bash
IMAGE=llm-scaler-omni:0.2.0-b3-comfyui-bmg
CONTAINER_NAME=comfyui
COMFYUI_MODEL_DIR=/path/to/comfyui_models
COMFYUI_INPUT_DIR=/path/to/comfyui_input
COMFYUI_OUTPUT_DIR=/path/to/comfyui_output
COMFYUI_USER_DIR=/path/to/comfyui_user

sudo docker run -itd \
    --device=/dev/dri \
    --network=host \
    --shm-size=64g \
    --name="$CONTAINER_NAME" \
    --workdir=/llm/ComfyUI \
    -v "$COMFYUI_MODEL_DIR":/models/host:ro \
    -v "$COMFYUI_INPUT_DIR":/data/input \
    -v "$COMFYUI_OUTPUT_DIR":/data/output \
    -v "$COMFYUI_USER_DIR":/data/user \
    "$IMAGE" \
    python main.py \
        --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
        --input-directory /data/input \
        --output-directory /data/output \
        --user-directory /data/user
```

Open `http://127.0.0.1:8188`. This direct ComfyUI launch is recommended by
default because it avoids weight-staging overhead when the workflow fits in
XPU memory. Append `--listen 0.0.0.0` when the server must accept remote
connections. The matching `comfyui-manager` Python package is installed in the
image; append `--enable-manager` when Node Manager is needed.

Use the supplied entrypoint only for workflows with a known or observed XPU
out-of-memory risk. It enables DynamicVRAM with the pinned AIMDO XPU backend,
reserves 4 GiB of XPU memory, and enables Node Manager. This lets resident
model weights be staged, unloaded, or reloaded to preserve activation
headroom, but the additional memory management can reduce performance for
workflows that already fit in memory:

```bash
sudo docker run -itd \
    --device=/dev/dri \
    --network=host \
    --name="$CONTAINER_NAME" \
    -v "$COMFYUI_MODEL_DIR":/models/host:ro \
    -v "$COMFYUI_INPUT_DIR":/data/input \
    -v "$COMFYUI_OUTPUT_DIR":/data/output \
    -v "$COMFYUI_USER_DIR":/data/user \
    "$IMAGE" \
    /llm/entrypoints/start_comfyui.sh
```

Override `OMNI_COMFYUI_RESERVE_VRAM_GB` only when the workload requires a
different reserve. The supplied entrypoint automatically loads
`/llm/configs/comfyui_host_models.yaml`.

The Linux provider defaults to `native_hook`; the entrypoint prepares its
verified preload so Torch retains its native XPU caching allocator. Set
`AIMDO_XPU_ALLOCATOR_MODE=global` to select the Linux pluggable allocator.

Keep host models and mutable runtime data outside `/llm/ComfyUI`. Mounting over
its `models`, `input`, or `output` directories hides files tracked by upstream
ComfyUI, which makes the checkout appear modified and causes
`tools/update_comfyui.sh` to refuse an upgrade. The external mounts, supplied
extra-model-paths config, and explicit data-directory arguments preserve model
discovery, generated data, and a clean, upgradable ComfyUI checkout.

For model placement, upstream templates, optional nodes, and runtime switches,
see [ComfyUI usage](docs/COMFYUI.md).

## Image contents

The focused source build selects:

- upstream [ComfyUI v0.42.1](https://github.com/Comfy-Org/ComfyUI/tree/v0.42.1),
  pinned to `63be2f9b82ad0dac88a7ceaa28a9cf3154125a18`;
- `omni_xpu_kernel`, built for the selected Torch minor and XPU target;
- official `comfy-kitchen==0.2.39` plus the matching co-installable XPU runtime provider
  from [`comfy-kitchen-xpu` revision](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/3c9c495f3c648166ec4b0c69b676904f09b53267),
  including the managed GGUF and Nunchaku W4A16 routes;
- official `comfy-aimdo==0.5.6` plus the co-installable XPU runtime provider
  from [`comfy-aimdo-xpu` revision](https://github.com/xiangyuT/comfy-aimdo-xpu/commit/96e6503c01124ce62bc4f5215458371a650c539f),
  built with its Level Zero backend and native allocator hook;
- [`ComfyUI-GGUF-XPU`](https://github.com/analytics-zoo/ComfyUI-GGUF-XPU/commit/39671fe73117ba97de7011e7e06e32599dcda06d),
  with GGUF, SentencePiece, and Protobuf dependencies installed from the same
  pinned checkout's requirements;
- [`ComfyUI-nunchaku-XPU==1.2.1+xpu.3`](https://github.com/xiangyuT/ComfyUI-nunchaku-XPU/commit/9f5604445f56154dd44e8b4acfed1aef868003c5),
  with its `nunchaku_torch` runtime bundled in the same pinned checkout;
- ComfyUI native **Model Sparse Attention** (SOL, SLA and VSA), backed by
  Kitchen and the packaged `omni_xpu_kernel` XPU operators; see
  [usage and legacy-node migration](docs/SPARSE_ATTENTION.md);
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md);
- ComfyUI v0.42.1 integrated Node Manager plus pinned VideoHelperSuite,
  Easy-Use, KJNodes, CacheDiT, and ControlNet auxiliary nodes;
- an exact installed Python dependency snapshot at
  `/llm/manifests/comfyui-python-freeze.txt`.

The focused image does not include Xinference, SGLang Diffusion, the disabled
audio/3D node bundle, repository workflow snapshots, or example input files.
Use ComfyUI's Template Browser for maintained upstream workflows.

## Build and component documentation

- [Image build and acceptance](docs/IMAGE_BUILD.md)
- [ComfyUI usage](docs/COMFYUI.md)
- [Native sparse attention and legacy-node migration](docs/SPARSE_ATTENTION.md)
- [Windows Intel XPU ComfyUI Portable deployment](docs/WINDOWS_PORTABLE.md)
- [Omni XPU kernel](omni_xpu_kernel/README.md)
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md)
- [Standalone examples](standalone_examples/)
