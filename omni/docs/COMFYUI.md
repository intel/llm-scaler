# ComfyUI usage

The default Omni image runs upstream ComfyUI on one Intel XPU. Models are not
bundled in the image.

## Start the server

Mount an existing ComfyUI model directory and start ComfyUI directly. This is
the recommended default when the workflow fits in XPU memory. The following
uses the local image produced by the source-build command in
[`../README.md`](../README.md); `0.2.0-b2` is not currently available as a
published image:

```bash
IMAGE=llm-scaler-omni:0.2.0-b2-comfyui-bmg
CONTAINER_NAME=comfyui

sudo docker run -itd \
    --device=/dev/dri \
    --network=host \
    --shm-size=64g \
    --name="$CONTAINER_NAME" \
    --workdir=/llm/ComfyUI \
    -v /path/to/comfyui_models:/models/host:ro \
    -v /path/to/comfyui_input:/data/input \
    -v /path/to/comfyui_output:/data/output \
    -v /path/to/comfyui_user:/data/user \
    "$IMAGE" \
    python main.py \
        --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
        --input-directory /data/input \
        --output-directory /data/output \
        --user-directory /data/user
```

This source-built image supports Intel Arc B-series/Battlemage GPUs.

The default server is available at `http://127.0.0.1:8188`. Append
`--listen 0.0.0.0` when remote access is required, and append
`--enable-manager` when the integrated Node Manager is needed.

## DynamicVRAM for memory-constrained workflows

Use the supplied entrypoint only when a workflow has a known or observed XPU
out-of-memory risk:

```bash
/llm/entrypoints/start_comfyui.sh
```

The entrypoint enables ComfyUI DynamicVRAM, backed by the image's pinned AIMDO
XPU/Level Zero allocator, enables Node Manager, and reserves 4 GiB of XPU
memory. DynamicVRAM stages, unloads, and reloads model weights to preserve
activation headroom during model switching or text re-encoding. This can avoid
OOM failures, but the additional weight-management work can reduce performance
when the workflow already fits in XPU memory. The reserve can be changed with
`OMNI_COMFYUI_RESERVE_VRAM_GB` when required by a specific workload.

Additional ComfyUI arguments are forwarded by the entrypoint. For example:

```bash
/llm/entrypoints/start_comfyui.sh --disable-smart-memory
```

## Models and workflows

Organize the host directory with the standard ComfyUI model subdirectories and
mount it read-only at `/models/host`. The supplied
`/llm/configs/comfyui_host_models.yaml` registers those directories with the
loader nodes. Use the model's official ComfyUI documentation for the exact
file names and directory:

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
- ComfyUI-SolAttn;
- ControlNet auxiliary nodes;
- ComfyUI-OmniXPU.

The Dockerfile is the source of truth for exact revisions. Installing or
updating nodes through ComfyUI Manager changes the running container and is
not part of the reproducible image build.

The image enables the Sol-Attn XPU adapter and uses the Sol-Attn implementation
packaged in `omni_xpu_kernel`; it does not install Triton for the XPU path. Add
**Patch Sol-Attn** after the model loader to opt a workflow into sparse
attention. Unsupported tensor contracts retain the original dense path.

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

Kitchen and AIMDO are installed as their official distributions. Their XPU
implementations are separately named runtime-provider distributions, activated
only during the normal ComfyUI custom-node prestartup lifecycle when the
official version, Torch XPU build, platform, target, and provider integrity all
match. No launcher change is required:

```bash
OMNIXPU_PROVIDER_BOOTSTRAP=auto      # use each compatible XPU provider
OMNIXPU_PROVIDER_BOOTSTRAP=off       # retain the official implementations
OMNIXPU_PROVIDER_BOOTSTRAP=required  # fail unless both providers activate
```

Reinstalling or upgrading an official Kitchen or AIMDO distribution does not
overwrite provider-owned files. If the official version no longer matches the
installed provider, `auto` mode leaves that provider inactive until its
matching XPU provider wheel is installed. AIMDO XPU activation additionally
requires DynamicVRAM; Kitchen routing does not.

The image exports a pip constraint for the Torch/XPU ABI and the provider
distributions. Upgrade the detached ComfyUI checkout with:

```bash
bash /llm/tools/update_comfyui.sh
```

The helper refuses tracked local edits, fetches ComfyUI `master` by default, and
installs its normal requirements under that runtime constraint. Set
`COMFYUI_UPGRADE_REF` to an exact commit for a reproducible upgrade. Run such
changes in a new container or preserve the container explicitly; they do not
modify the source image.

Do not mount host directories over `/llm/ComfyUI/models`, `input`, or `output`
when the checkout must remain upgradable. Upstream tracks files below those
directories; replacing them with bind mounts makes the files appear deleted,
so the helper correctly rejects the checkout as modified. Mount host models at
`/models/host:ro`, mount mutable state below `/data`, and use the supplied
configuration and path arguments as in the startup command above.

See [ComfyUI-OmniXPU](../ComfyUI-OmniXPU/README.md) for adapter behavior,
diagnostics, and opt-in legacy workarounds.

## Outputs

Mount `/data/output` and select it with `--output-directory` when generated
files must survive container removal:

```bash
-v /path/to/comfyui_output:/data/output
```

Input files and user state can similarly be mounted at `/data/input` and
`/data/user`, selected with `--input-directory` and `--user-directory`.
