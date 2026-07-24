# LLM Scaler Omni

LLM Scaler Omni provides Intel XPU images for generative media workloads. The
default image is a single-XPU ComfyUI environment with target-specific
`omni_xpu_kernel` binaries, the XPU-enabled Comfy Kitchen backend, and a thin
ComfyUI integration layer.

The optional `full` image also contains Xinference and SGLang Diffusion. It is
not the default build and is maintained separately from the focused ComfyUI
runtime.

## Getting Started with the Omni Docker Image

Build from the `omni` directory:

```bash
cd omni

# Intel Arc B-series / Battlemage
XPU_TARGET=bmg bash build.sh

# Intel Panther Lake H
XPU_TARGET=ptl-h bash build.sh
```

`XPU_TARGET` is required to match the destination GPU because the native wheel
is AOT-compiled for that target. Supported values are `bmg` and `ptl-h`.

The generated image tag includes the image flavor and target:

```text
intel/llm-scaler-omni:<version>-comfyui-bmg
intel/llm-scaler-omni:<version>-comfyui-ptl-h
```

See [Releases](../Releases.md) for published image tags. Development tags are
read from `omni_xpu_kernel/omni_xpu_kernel/_version.py`.

### Validate the image

Run the supplied acceptance script against the final image with the GPU device
exposed:

```bash
IMAGE=intel/llm-scaler-omni:0.1.0-b9-dev-comfyui-bmg

docker run --rm \
    --device=/dev/dri \
    "$IMAGE" \
    python /llm/tools/validate_comfyui_image.py
```

The check verifies package identity, the Torch ABI, native AOT target, clean
source provenance, dependencies, XPU availability, and required Kitchen
capabilities. A BMG image must not be renamed or reused for PTL-H, or vice
versa.

### Run ComfyUI

Mount the existing ComfyUI model directory rather than copying models into the
image:

```bash
IMAGE=intel/llm-scaler-omni:0.1.0-b9-dev-comfyui-bmg
COMFYUI_MODEL_DIR=/path/to/comfyui_models
COMFYUI_OUTPUT_DIR=/path/to/comfyui_output

docker run --rm -it \
    --device=/dev/dri \
    --network=host \
    --shm-size=64g \
    -v "$COMFYUI_MODEL_DIR":/llm/ComfyUI/models \
    -v "$COMFYUI_OUTPUT_DIR":/llm/ComfyUI/output \
    "$IMAGE" \
    /llm/entrypoints/start_comfyui.sh
```

Open `http://127.0.0.1:8188`. Additional ComfyUI arguments can be appended to
the command.

The entrypoint reserves 4 GiB of XPU memory by default so a resident diffusion
model can be offloaded before an XPU text encoder is executed again. Override
the reserve only when required by the workload:

```bash
docker run --rm -it \
    --device=/dev/dri \
    --network=host \
    -e OMNI_COMFYUI_RESERVE_VRAM_GB=6 \
    -v "$COMFYUI_MODEL_DIR":/llm/ComfyUI/models \
    "$IMAGE" \
    /llm/entrypoints/start_comfyui.sh
```

For model placement, upstream templates, optional nodes, and runtime switches,
see [ComfyUI usage](docs/COMFYUI.md).

## Image contents

The focused image contains:

- a pinned upstream ComfyUI checkout;
- `omni_xpu_kernel`, built for the selected Torch minor and XPU target;
- `comfy-kitchen==0.2.18` from the XPU-enabled
  [`comfy-kitchen-xpu` main merge](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/acdf65deace1b0ca3b436f45e560ed44f0c0d08f);
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md);
- pinned ComfyUI Manager, VideoHelperSuite, Easy-Use, KJNodes, CacheDiT,
  GGUF-XPU, Nunchaku-XPU, and ControlNet auxiliary nodes.

The focused image does not include Xinference, SGLang Diffusion, the disabled
audio/3D node bundle, repository workflow snapshots, or example input files.
Use ComfyUI's Template Browser for maintained upstream workflows.

## Optional full image

Build the full image only when Xinference, SGLang Diffusion, or the optional
workflow bundle is needed:

```bash
OMNI_IMAGE_FLAVOR=full XPU_TARGET=bmg bash build.sh
```

The resulting tag is:

```text
intel/llm-scaler-omni:<version>-full-<target>
```

The full image retains a broader dependency set and repository workflows. See
the [SGLang Diffusion guide](docs/SGLang_Diffusion_Guide.md) and
[SGLang Diffusion ComfyUI guide](docs/SGLang_Diffusion_ComfyUI_Guide.md).

## Build and component documentation

- [Image build and acceptance](docs/IMAGE_BUILD.md)
- [ComfyUI usage](docs/COMFYUI.md)
- [Omni XPU kernel](omni_xpu_kernel/README.md)
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md)
- [Windows portable setup](comfyui_windows_setup/README.md)
- [Standalone examples](standalone_examples/)
