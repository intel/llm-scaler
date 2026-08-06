# LLM Scaler Omni

LLM Scaler Omni provides Intel XPU images for generative media workloads. The
default image is a single-XPU ComfyUI environment with target-specific
`omni_xpu_kernel` binaries, the XPU-enabled Comfy Kitchen backend, and a thin
ComfyUI integration layer.

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
the command. The entrypoint enables ComfyUI v0.30's integrated Node Manager;
the matching `comfyui-manager` Python package is installed in the image.

The entrypoint enables ComfyUI DynamicVRAM with the pinned AIMDO XPU backend
and reserves 4 GiB of XPU memory by default. This lets resident model weights
be evicted under allocator pressure before an XPU text encoder is executed
again. Override the reserve only when required by the workload:

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

- upstream [ComfyUI v0.30.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.30.0),
  pinned to `b1693ecba9f5b65f8c80ab36b195ab963ec92413`;
- `omni_xpu_kernel`, built for the selected Torch minor and XPU target;
- `comfy-kitchen==0.2.26` from the XPU-enabled
  [`comfy-kitchen-xpu` revision](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/f7250fa44cb6f593969ba869be803e7d03c80ec8),
  including the managed GGUF and Nunchaku W4A16 routes;
- `comfy-aimdo==0.4.13` from the XPU-enabled
  [`comfy-aimdo` fork](https://github.com/xiangyuT/comfy-aimdo-xpu) at revision
  `6fda6e619e1647134d4ced4370e5fad488779d62`, built with its Level Zero
  allocator backend;
- [`ComfyUI-GGUF-XPU`](https://github.com/analytics-zoo/ComfyUI-GGUF-XPU/commit/39671fe73117ba97de7011e7e06e32599dcda06d),
  with GGUF, SentencePiece, and Protobuf dependencies installed from the same
  pinned checkout's requirements;
- [`ComfyUI-nunchaku-XPU==1.2.1+xpu.3`](https://github.com/xiangyuT/ComfyUI-nunchaku-XPU/commit/5cf4fa9886f45abff102d1dd91af5247b4950148),
  with its `nunchaku_torch` runtime bundled in the same pinned checkout;
- [ComfyUI-OmniXPU](ComfyUI-OmniXPU/README.md);
- ComfyUI v0.30 integrated Node Manager plus pinned VideoHelperSuite,
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
