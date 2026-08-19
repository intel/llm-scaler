# LLM Scaler Omni

LLM Scaler Omni provides Intel XPU images for generative media workloads. The
default image is a single-XPU ComfyUI environment with target-specific
`omni_xpu_kernel` binaries, the XPU-enabled Comfy Kitchen backend, and a thin
ComfyUI integration layer.

> [!IMPORTANT]
> The current 0.2.0 beta preview is experimental and focused on single-XPU
> ComfyUI workloads. It does not replace the broader b8 image. For SGLang Diffusion,
> Raylight, or other multi-XPU scenarios, use the published
> [`intel/llm-scaler-omni:0.1.0-b8`](https://github.com/intel/llm-scaler/releases/tag/omni-0.1.0-b8)
> image.

## Getting Started with the Omni Docker Image

Build from the `omni` directory:

```bash
cd omni

# Intel Arc B-series / Battlemage
XPU_TARGET=bmg bash build.sh
```

The current image supports Intel Arc B-series/Battlemage GPUs. Its native wheel
is AOT-compiled for BMG, and `build.sh` assigns this tag to local images:

```text
intel/llm-scaler-omni:<version>-comfyui-bmg
```

We publish the BMG image using the version as the image tag:

```text
intel/llm-scaler-omni:<version>
```

Version `0.2.0-b1` is tagged `intel/llm-scaler-omni:0.2.0-b1`. Published tags
are listed in [Releases](../Releases.md). The development version is defined
in `omni_xpu_kernel/omni_xpu_kernel/_version.py`.

### Validate the image

Run the supplied acceptance script against the final image with the GPU device
exposed:

```bash
IMAGE=intel/llm-scaler-omni:0.2.0-b1

sudo docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    python /llm/tools/validate_comfyui_image.py
```

The check verifies package identity, the Torch ABI, native AOT target, clean
source provenance, dependencies, XPU availability, and required Kitchen
capabilities. The release image supports BMG.

### Run ComfyUI

Mount the existing ComfyUI model directory rather than copying models into the
image:

```bash
IMAGE=intel/llm-scaler-omni:0.2.0-b1
CONTAINER_NAME=comfyui
COMFYUI_MODEL_DIR=/path/to/comfyui_models
COMFYUI_OUTPUT_DIR=/path/to/comfyui_output

sudo docker run -itd \
    --privileged \
    --device=/dev/dri \
    --network=host \
    --shm-size=64g \
    --name="$CONTAINER_NAME" \
    --workdir=/llm/ComfyUI \
    -v "$COMFYUI_MODEL_DIR":/llm/ComfyUI/models \
    -v "$COMFYUI_OUTPUT_DIR":/llm/ComfyUI/output \
    "$IMAGE" \
    python main.py
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
    --privileged \
    --device=/dev/dri \
    --network=host \
    --name="$CONTAINER_NAME" \
    -v "$COMFYUI_MODEL_DIR":/llm/ComfyUI/models \
    "$IMAGE" \
    /llm/entrypoints/start_comfyui.sh
```

Override `OMNI_COMFYUI_RESERVE_VRAM_GB` only when the workload requires a
different reserve.

For model placement, upstream templates, optional nodes, and runtime switches,
see [ComfyUI usage](docs/COMFYUI.md).

## Image contents

The focused image contains:

- upstream [ComfyUI v0.31.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.31.0),
  pinned to `43cb4fffc89bba20ab7bd61467a36d0339338dab`;
- `omni_xpu_kernel`, built for the selected Torch minor and XPU target;
- `comfy-kitchen==0.2.31` from the XPU-enabled
  [`comfy-kitchen-xpu` revision](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/74aead36b68cc67f156d8b6724ed7f6acb05a528),
  including the managed GGUF and Nunchaku W4A16 routes;
- `comfy-aimdo==0.4.13` from the XPU-enabled
  [`comfy-aimdo` fork](https://github.com/xiangyuT/comfy-aimdo-xpu) at revision
  `8600b938a517af7329cbf290d6e70a52ae78187d`, built with its Level Zero
  allocator backend;
- [`ComfyUI-GGUF-XPU`](https://github.com/analytics-zoo/ComfyUI-GGUF-XPU/commit/39671fe73117ba97de7011e7e06e32599dcda06d),
  with GGUF, SentencePiece, and Protobuf dependencies installed from the same
  pinned checkout's requirements;
- [`ComfyUI-nunchaku-XPU==1.2.1+xpu.3`](https://github.com/xiangyuT/ComfyUI-nunchaku-XPU/commit/cc0f6236b6c329178ad4ef58452a874e774c7b8e),
  with its `nunchaku_torch` runtime bundled in the same pinned checkout;
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md);
- ComfyUI v0.31 integrated Node Manager plus pinned VideoHelperSuite,
  Easy-Use, KJNodes, CacheDiT, and ControlNet auxiliary nodes;
- an exact installed Python dependency snapshot at
  `/llm/manifests/comfyui-python-freeze.txt`.

The focused image does not include Xinference, SGLang Diffusion, the disabled
audio/3D node bundle, repository workflow snapshots, or example input files.
Use ComfyUI's Template Browser for maintained upstream workflows.

## Build and component documentation

- [Image build and acceptance](docs/IMAGE_BUILD.md)
- [ComfyUI usage](docs/COMFYUI.md)
- [Windows Intel XPU ComfyUI Portable deployment](docs/WINDOWS_PORTABLE.md)
- [Omni XPU kernel](omni_xpu_kernel/README.md)
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md)
- [Standalone examples](standalone_examples/)
