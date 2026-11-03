
# 01. System Hang During Ubuntu 25.04 Installation with B60 Card Plugged In
The issue is caused by an outdated GPU GuC firmware bundled in the official Ubuntu 25.04 Desktop ISO image.

Workaround: Remove the B60 card before starting the Ubuntu installation, and plug it back in once the installation is complete.
We are also working with the Ubuntu team to address this issue upstream.

# 02. Limited 33 GB/s Bi-Directional P2P Bandwidth with 1x GPU Card
When using a single GPU card over a x16 PCIe connection without a PCIe switch, the observed bi-directional P2P bandwidth is limited to 33 GB/s.

Workaround: Change the PCIe slot configuration in BIOS from Auto/x16 to x8/x8.
With this change, over 40 GB/s bi-directional P2P bandwidth can be achieved.
Root cause analysis is still in progress.

# 03. Gemma 4 LoRA Adapters with Fused 3D Expert Weights Return Wrong Output in Graph Mode
With gemma-4-26B-A4B-it and an adapter that uses fused 3D expert weights (`is_3d_lora_weight=true`), graph mode returns incorrect output for every request once more than one request is batched. Regular adapters are not affected.

Workaround: Serve these adapters with `--enforce-eager`.

# 04. Qwen3.6-35B-A3B Expert LoRA Not Applied in Mixed Batches
For an adapter that also trains the routed experts, the expert part of the adapter may not be applied when its requests are batched with requests for the base model or other adapters. The output can then match the base model. Adapters that do not train the experts are not affected.

Root cause analysis is in progress.
