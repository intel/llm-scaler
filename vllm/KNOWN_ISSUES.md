
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

# 04. SYCL "No device of requested type available" with Tensor Parallelism (TP > 1)

When launching vLLM with `--tensor-parallel-size` > 1, worker processes crash with:

```
sycl::_V1::exception: No device of requested type available
```

This does not occur with TP=1 because single-worker deployments never use Intel oneCCL
for inter-process collective operations.

### Cause 1 (most common): `CCL_ZE_IPC_EXCHANGE=pidfd` blocked by container seccomp

When TP > 1, workers use Intel oneCCL for collective communication. oneCCL's default
IPC exchange mechanism (`pidfd`) requires the `pidfd_getfd` syscall, which is blocked
by many container runtimes' seccomp profiles. This causes the Level Zero device handle
transfer to fail, and the SYCL runtime reports "No device of requested type available".

**Fix:**
```bash
export CCL_ZE_IPC_EXCHANGE=sockets
```
The `start_vllm_openai.sh` entrypoint sets this automatically.

### Cause 2: Wrong multiprocessing start method

The Level Zero runtime is not fork-safe. If workers are started with `fork` (Python's
default), child processes inherit the parent's SYCL/Level Zero device context. That
context is invalid in the child, causing the device to appear unavailable.

**Fix:**
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```
The Docker image sets this via `ENV VLLM_WORKER_MULTIPROC_METHOD=spawn` in the Dockerfile.

### Cause 3: Missing OneAPI environment initialization

Without `source /opt/intel/oneapi/setvars.sh --force`, `LD_LIBRARY_PATH` may be
incomplete and the SYCL runtime cannot locate the Level Zero ICD loader (see also
Known Issue #03).

**Fix:**
```bash
source /opt/intel/oneapi/setvars.sh --force
```

### Required environment for TP > 1

```bash
source /opt/intel/oneapi/setvars.sh --force
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_TRANSPORT=ofi
```
