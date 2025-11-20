
# 01. System Hang During Ubuntu 25.04 Installation with B60 Card Plugged In
The issue is caused by an outdated GPU GuC firmware bundled in the official Ubuntu 25.04 Desktop ISO image.

Workaround: Remove the B60 card before starting the Ubuntu installation, and plug it back in once the installation is complete.
We are also working with the Ubuntu team to address this issue upstream.

# 02. Limited 33 GB/s Bi-Directional P2P Bandwidth with 1x GPU Card
When using a single GPU card over a x16 PCIe connection without a PCIe switch, the observed bi-directional P2P bandwidth is limited to 33 GB/s.

Workaround: Change the PCIe slot configuration in BIOS from Auto/x16 to x8/x8.
With this change, over 40 GB/s bi-directional P2P bandwidth can be achieved.
Root cause analysis is still in progress.

# 03. Container OOM killed (and vllm performance drop) when starting container not by /bin/bash and not run `source /opt/intel/oneapi/setvars.sh`

When using `--enable-auto-tool-choice` and deploy container by docker-compose without `source /opt/intel/oneapi/setvars.sh`, the LD_LIBRARY_PATH will be different and cause the container OOM (or performance drop). It can be reproduced by this two command:

```bash
docker run --rm  --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c env | grep LD_LIBRARY_PATH
 
docker run --rm --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c "source /opt/intel/oneapi/setvars.sh --force && env | grep LD_LIBRARY_PATH"
```

So we need to run `source /opt/intel/oneapi/setvars.sh --force` to ensure some configurations are consistent.
