# Native sparse attention in ComfyUI

With ComfyUI 0.35.0 or later and a matching Omni XPU stack, use the built-in
**Model Sparse Attention** node for sparse attention. Search for that name
under **model/patch**; its workflow and API class ID is `BlockSparseAttention`.

The legacy `ComfyUI-SolAttn_xpu` custom node and **Patch Sol-Attn** workflow
entry point are deprecated for this Omni integration. The focused Dockerfile
no longer installs the legacy custom node.

## Requirements

Use the matched ComfyUI, Kitchen XPU provider, `omni_xpu_kernel` and
ComfyUI-OmniXPU sources selected by this checkout's
[image build](IMAGE_BUILD.md). The current build pins ComfyUI 0.35.0 and
Kitchen 0.2.33. Installing a newer ComfyUI alone does not provide the XPU
backend; an older native wheel may expose legacy Sol operators while lacking
the complete API needed by the built-in node.

ComfyUI owns the node, model patch, sparse selection and MiniMax-H3 layout.
ComfyUI-OmniXPU enables eligible XPU calls, Kitchen dispatches them, and
`omni_xpu_kernel` supplies the native operators. No separate Sol custom node,
Triton installation or `SOL_ATTN_XPU_EXPERIMENTAL=1` switch is needed for this
XPU route.

## Connect the node

1. Start with the model's maintained ComfyUI template and load its matching
   diffusion model, text encoder and video/audio VAE.
2. Apply any required model LoRA and model sampling/shift nodes first.
3. Add **Model Sparse Attention** after those model changes. Connect its
   `model` output to the model input consumed by the sampler or guider.
4. Select the method and explicitly set its parameters. Keep the template's
   conditioning, latent and output connections.

For MiniMax-H3 the model path is:

```text
Model loader → optional model LoRA → MiniMaxH3SigmaShift
             → Model Sparse Attention → guider/sampler
```

The sparse node converts its start/end percentages using the incoming model's
sampling configuration, so apply the shift before the sparse patch. To use
dense attention, bypass the sparse node and connect the unpatched model to
the guider/sampler. Use the base model's dense recipe for this comparison;
disabling the sparse node on a VSA checkpoint also removes its learned coarse
branch.

## Choose a method and matching weights

These are starting configurations exercised with MiniMax-H3 at 864×480 and
1344×768, with a 5s duration input. The table combines model-author settings,
ComfyUI defaults and explicit integration choices; their sources are explained
below. These are complete model recipes, not interchangeable kernel switches.

| Setting | SOL | SLA | VSA |
| --- | --- | --- | --- |
| UI `method` / API `selection` | `sol-attn` | `sla` | `vsa` |
| Model | Base MiniMax-H3 | Base + matching Turbo-SLA LoRA, strength 1.0 | Matching FastH3 VSA checkpoint with learned coarse gates |
| Steps | 20 | 4 | 4 |
| Sampler / scheduler | `res_multistep` / `simple` | `euler` / `simple` | `euler` / `simple` |
| Guidance | `BasicGuider` (single conditioning) | `BasicGuider` (single conditioning) | `BasicGuider` (single conditioning) |
| Video / audio shift | 12 / 3 | 6 / 3 | 12 / 3 |
| Selection parameter | `tau=1.3` | `keep_percent=15.0` | `keep_percent=10.0` |
| `start_percent` / `end_percent` | 0.2 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| `extra_tokens` | 256 | 256 | 0 |
| `sink_conditioning` | `exact_kv_and_rows` | `exact_kv_and_rows` | Native VSA prefix handling |
| `min_tokens` | 12288 | 12288 | 12288 |
| `dense_blocks` | Empty | Empty | Empty |

### Parameter sources

- **SOL base recipe:** the official
  [MiniMax-H3 template in workflow-templates 0.11.57](https://github.com/Comfy-Org/workflow_templates/blob/v0.11.57/templates/video_minimax_h3_t2v.json)
  uses 20 steps with Turbo disabled, `res_multistep`, `simple` and
  `BasicGuider`. Its steps input is connected to a switch, so the scheduler's
  stored widget value alone is not the effective step count. The
  [ComfyUI MiniMaxH3 model definition](https://github.com/Comfy-Org/ComfyUI/blob/40c4fcdf513a4523e39d54a9d391908af8df8171/comfy/supported_models.py)
  sets the base video/audio shifts to 12/3. SOL retains this base recipe.
- **SLA model settings:** the
  [Turbo-SLA model card](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/10ade67cd15ff7a135fa35c2a0673ea96c839247/README.md)
  specifies four-step distillation and 85% sparsity, mapped here to
  `keep_percent=15.0`. Its
  [LightX2V configuration](https://github.com/ModelTC/LightX2V/blob/ca181cab7f454f804ab9c23f8811316a242b1e6c/configs/minimax_h3/dmd/minimax_h3_fp8_4step_5090_with_fp8_vae_sla.json)
  supplies LoRA strength 1.0, video/audio shifts 6/3 and disabled CFG.
- **VSA model settings:** the matching step-1300
  [FastH3 inference configuration](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/blob/5ea076f35b84da4c3c82217112fa733d8eea2ae1/fastvideo_inference.json)
  specifies four transformer forwards, guidance 1.0, tile size 64 and
  90% sparsity, mapped here to `keep_percent=10.0`. The same checkpoint's
  [video scheduler](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/blob/5ea076f35b84da4c3c82217112fa733d8eea2ae1/scheduler/scheduler_config.json)
  and [audio scheduler](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/blob/5ea076f35b84da4c3c82217112fa733d8eea2ae1/audio_scheduler/scheduler_config.json)
  set shifts 12/3.
- **Native node defaults and behavior:** the
  [ComfyUI sparse node](https://github.com/Comfy-Org/ComfyUI/blob/40c4fcdf513a4523e39d54a9d391908af8df8171/comfy_extras/nodes_sparse_attention.py)
  defines the method names, SOL `tau=1.3`, the default 0.2–1.0 window,
  `min_tokens=12288`, empty `dense_blocks`, `extra_tokens=256` and
  `sink_conditioning=exact_kv_and_rows`. It forces VSA augmentation to zero
  and implements its prefix handling. These are node defaults or semantics,
  not model-author tuning recommendations. SLA's table value of 15% overrides
  the node's default 10% to follow the selected model.
- **ComfyUI integration choices:** SLA/VSA use `start_percent=0.0` here to
  enable sparse attention throughout their four-step recipes. `BasicGuider`
  expresses sampling without CFG. `euler`/`simple` is the ComfyUI sampler
  combination used for these recipes; the
  [community VSA port](https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo/blob/d610a06f7dab47f7d6329a772990bbf175b0da3c/README.md#workflow)
  also recommends that combination. The SLA configuration names
  `training_euler` and sets `infer_steps=5`; FastH3 records both
  `num_inference_steps=5` and `transformer_forwards=4`. These framework fields
  are not copied literally into ComfyUI's four sampler steps, and scheduler
  numerical equivalence is not established. The full sparse window and this
  sampler mapping are integration choices, not a shared official SOL/SLA/VSA
  preset.

### Weight selection and parameter meaning

SOL uses an adaptive threshold without sparse-specific training. Higher `tau`
requests more sparsity; it is not an exact keep percentage. SLA needs weights
trained for its selection pattern. VSA needs a matching FastH3 checkpoint and
its `to_gate_compress` layers; a missing-gate warning is not a successful VSA
configuration. Do not select SLA/VSA on arbitrary dense weights just to reduce
the step count.

For SLA, use `minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors`
with `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, following the
[MiniMax-H3 Turbo-SLA LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA/blob/10ade67cd15ff7a135fa35c2a0673ea96c839247/README.md).
For VSA, use the
[FastH3 VSA model recipe](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/blob/5ea076f35b84da4c3c82217112fa733d8eea2ae1/README.md)
and its matching ComfyUI conversion. The exercised INT8 conversion is
`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`
from [Kijai's MiniMax-H3 conversions](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/README.md).
Place the full VSA diffusion checkpoint under `models/diffusion_models` and
the SLA LoRA under `models/loras`; retain the matching template's other weights.
The four-step entries above use ComfyUI's sampling schedule; they do not claim
numerical identity with the training frameworks' scheduler implementations.

`keep_percent` uses percent units: enter `15.0` for 15%, not `0.15`.
Sinks, diagonal blocks, augmentation and the VSA coarse branch add work beyond
that selection percentage. VSA uses 4×4×4 video cubes and ignores
`extra_tokens`; leave it at zero. Its native prefix handling is independent of
the sink selector. Keep `exact_kv_and_rows` for the SOL/SLA audio-video recipes.

The node's default window starts at 0.2 and its default SLA keep is 10%; set
the trained recipes above explicitly. Sequence length is derived from the
actual inputs. Short sequences below `min_tokens`, excluded blocks, or steps
outside the sparse window remain dense. Other resolutions and durations need
their own quality and capacity checks, without a fixed sequence-length route.

## Migrate a legacy workflow

1. Save a copy of the original workflow. Replace **Patch Sol-Attn** with
   **Model Sparse Attention**, then reconnect the model path described above.
   Old node IDs and saved widget lists are not automatically compatible.
2. Select `sol-attn` for a legacy SOL recipe and review `tau`, the sparse
   window, minimum tokens, sinks and augmentation explicitly. Do not transfer
   the old CUDA-oriented `int8_qk` or `use_tma` widgets: the native backend
   selects its own implementation. Output equivalence is not implied.
3. Remove the old plugin from the active `custom_nodes` directory after
   migrating dependent workflows; keep a backup outside that directory.
   Remove legacy `SOL_ATTN` and `SOL_ATTN_XPU_EXPERIMENTAL` launcher settings.
   Apply only the intended sparse model patch.
4. Restart ComfyUI and validate a generated clip with the migrated workflow.

## Confirm sparse execution

- Keep `OMNIXPU_ENABLE` and `OMNIXPU_SPARSE_ATTENTION` enabled (both default to
  enabled). The **OmniXPU Status** node should report
  `sparse_attention_adapter` as applied. Setting
  `OMNIXPU_SPARSE_ATTENTION=0` disables the XPU eligibility adapter; it is not
  a method selector.
- Enable the sparse node's advanced `verbose` option for the first run.
  Inspect `BlockSparseAttention` logs for sparse execution or the reason a
  call remains dense. Dense calls outside the selected window are expected.
- If the node is missing, check the ComfyUI revision and built-in node import
  logs. If the adapter is skipped, check the matched Kitchen provider and
  complete native sparse API. A newer upstream eligibility contract may also
  require an adapter update.
- Check the whole output for coherent motion, expected dimensions/duration,
  and usable audio. Successful node import alone does not verify a sparse
  route, output quality or a speedup.

The focused image validator checks the native node's source/schema, adapter
compatibility, Kitchen XPU capabilities and the packaged sparse library.
Actual node registration, sparse execution and output quality are checked by
running the workflow. Compare performance using the same recipe and exclude
cold loading and warmup; differences between the four-step and twenty-step
recipes cannot be attributed solely to attention.

The node schema and behavior come from the pinned
[ComfyUI source](https://github.com/Comfy-Org/ComfyUI/blob/40c4fcdf513a4523e39d54a9d391908af8df8171/comfy_extras/nodes_sparse_attention.py).
Native XPU implementation is maintained in
[Intel llm-scaler](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel).
