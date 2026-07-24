# ComfyUI usage

The default Omni image runs upstream ComfyUI on one Intel XPU. Models are not
bundled in the image.

## Start the server

Mount an existing ComfyUI model directory and invoke the supplied entrypoint:

```bash
IMAGE=intel/llm-scaler-omni:0.1.0-b9-dev-comfyui-bmg

docker run --rm -it \
    --device=/dev/dri \
    --network=host \
    --shm-size=64g \
    -v /path/to/comfyui_models:/llm/ComfyUI/models \
    "$IMAGE" \
    /llm/entrypoints/start_comfyui.sh
```

The entrypoint listens on port `8188`. Extra arguments are forwarded to
ComfyUI:

```bash
/llm/entrypoints/start_comfyui.sh --disable-smart-memory
```

The default 4 GiB XPU reserve can be changed with
`OMNI_COMFYUI_RESERVE_VRAM_GB`. Keeping a nonzero reserve is important for
workflows that execute an XPU text encoder again after diffusion weights are
resident.

## Models and workflows

Place model files under the standard `/llm/ComfyUI/models` subdirectories used
by their loader nodes. Use the model's official ComfyUI documentation for the
exact file names and directory:

- [ComfyUI documentation](https://docs.comfy.org/)
- [ComfyUI Template Browser](https://docs.comfy.org/interface/features/template)
- [ComfyUI model tutorials](https://docs.comfy.org/tutorials)

The focused image deliberately does not copy `omni/workflows` or
`omni/example_inputs`. This prevents stale workflow snapshots from replacing
maintained upstream templates.

## Included custom nodes

The focused image installs pinned revisions of:

- ComfyUI Manager;
- VideoHelperSuite;
- Easy-Use;
- KJNodes;
- CacheDiT;
- ComfyUI-GGUF-XPU;
- ComfyUI-nunchaku-XPU;
- ControlNet auxiliary nodes;
- ComfyUI-OmniXPU.

The Dockerfile is the source of truth for exact revisions. Installing or
updating nodes through ComfyUI Manager changes the running container and is
not part of the reproducible image build.

## Omni XPU switches

ComfyUI-OmniXPU adapters are enabled by default and fall back to the original
ComfyUI path when a capability or input is unsupported. Common switches are:

```bash
OMNIXPU_ENABLE=0
OMNIXPU_ATTENTION=0
OMNIXPU_NORM=0
OMNIXPU_FP8_GEMM=0
OMNIXPU_INT8_FFN=0
```

See [ComfyUI-OmniXPU](../ComfyUI-OmniXPU/README.md) for adapter behavior,
diagnostics, and opt-in legacy workarounds.

## Outputs

Mount `/llm/ComfyUI/output` when generated files must survive container
removal:

```bash
-v /path/to/comfyui_output:/llm/ComfyUI/output
```

Input files can similarly be mounted at `/llm/ComfyUI/input`.
