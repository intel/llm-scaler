# llama.cpp vs vLLM on the MSI Claw 8 AI+: which one uses memory better (2026-11-11)

Method: one research pass (Opus 5.5) over the two repos' measurements and a clone of llama.cpp master
(`f3b407c`, 2026-11-11). GGUF sizes come from web-search snippets because huggingface.co was unreachable;
treat them as approximate. Nothing here was re-measured on the console.

## TL;DR

llama.cpp is the better fit for memory on this box, by about 1.5–3 GiB at the same model and context: its
GGUFs quantize the embedding and output tensors, its KV cache can be stored at q8_0 or q4_0, it allocates
exactly what you ask for with no ~2 GiB Python layer and no profiling peak, it can push experts to the CPU,
and the repos record no memory leak for it. vLLM is the better choice per GiB of throughput: it decodes
1.4–2.4× faster per stream (about 18–23 tok/s against 9.5–13.3 tok/s; both sides stale: llama.cpp March build, vLLM v0.19; verdict pending a current-build re-bench), so for one resident model that fits
under about 21 GiB, vLLM still gives more tokens per byte.

## Side-by-side

| Row | llama.cpp (Vulkan / SYCL, `-ngl 99`) | vLLM v0.32 XPU (eager) |
|---|---|---|
| Weights resident | Whole GGUF in driver buffers; `token_embd` always on the CPU (`llama-model.cpp:1781`). Qwen3-Coder Q4_K_M ≈ 17.3 GiB; IQ4_XS ≈ 15.3 GiB | Qwen3-Coder compressed-tensors gs32 ≈ 17 GiB; Qwen3-30B GPTQ gs128 measured 15.56 GiB; GPT-OSS-20B MXFP4 12.87 GiB |
| Page-cache double copy | Vulkan on an iGPU does not mmap at all (`mmap_support = !is_integrated_gpu`, `ggml-vulkan.cpp:15636`); SYCL mmaps, copies, then unmaps. Either way a reclaimable page-cache copy remains | safetensors read fills the page cache (Finding 5); reclaimable; driver counts usable memory from MemAvailable since compute-runtime 26.18 |
| KV per token, Qwen3-Coder (48 layers × 4 KV heads × 128) | f16 96 KiB; q8_0 51 KiB; q4_0 27 KiB | bf16 96 KiB; fp8_e4m3 48 KiB (unvalidated here) |
| KV reservation | all of `-c` allocated at startup, shared across slots; exact | fixed pool from `--kv-cache-memory-bytes`, else util × total minus the profiled peak |
| Runtime overhead | compute buffer 0.5–1 GiB est.; process RSS 0.3–0.5 GiB est.; **prompt cache defaults to 8 GiB of host RAM** (`--cache-ram`) | ~2 GiB Python/torch host side; activations 0.5–1.0 GiB; scratch ~0.3 GiB (Finding 7); profile peak ~8 GiB on v0.19 unless the KV budget is pinned |
| Partial offload | `-ngl N`, `-ot`, `--cpu-moe`, `--n-cpu-moe`, `--moe-cache-mib` all work | `--cpu-offload-gb` only moves accounting on UMA |
| Crash / leak recovery | nothing recorded; Vulkan buffers freed when the DRM fd closes (est.); SYCL untested against the leak | Level Zero shared-memory leak even on clean exits (Finding 3): 28 → 7–8 GiB free, drop_caches sometimes recovers, reboot after 2–3 crashes |
| Load / switch | not measured; ~5–15 s est. | 36–40 s measured |
| Measured tok/s here | Qwen3.5-35B IQ4_XS 13.3; Q4_K_XL 9.5; GPT-OSS-20B 9.5; Nemotron-30B-A3B 16.9 (all Vulkan; no SYCL numbers on Lunar Lake) | Qwen3-Coder gs32 18.2 (v0.19.0); Qwen3.6-35B ~19 (27.7 with MTP k=2; v0.30, MXFP4); GPT-OSS-20B ~23 (v0.19.0) |
| Second request | no extra KV: `-c` is split or shared across slots; hybrids add ~60 MiB state per slot | shares the paged pool; costs only the blocks it uses; much better aggregate throughput |

## Weight bytes

Qwen3-30B-A3B-class: 30.5B parameters, 28.99B in experts. GGUF Q4_K_M ≈ 4.9 bpw (half of attn_v/ffn_down and
the output tensor at Q6_K), Q4_K_S ≈ 4.6, IQ4_XS ≈ 4.3; GGUF also quantizes `token_embd` and the output.
vLLM compressed-tensors gs32 is 4.5 bpw on quantized tensors with embedding, `lm_head`, router and norms in
bf16 (~1.16 GiB), ≈ 16.8 GiB total; GPTQ gs128 ≈ 4.16 bpw (15.56 GiB measured); MXFP4 4.25 bpw on experts
only. At similar quality IQ4_XS beats gs32 by ~1.5 GiB, but gs32 is what vLLM needs for speed (gs128 ≈ 10 tok/s).

## Where the weights live

Both runtimes put the weights in driver-allocated system RAM (shmem-backed buffers on the xe driver) and
both leave a reclaimable copy in the page cache. Brainstorm idea S20's conclusion (mmap does not shrink the
resident set with `-ngl 99`) holds, with two qualifications: Vulkan on an iGPU never mmaps, so there is
nothing to unmap; and an mmap-backed, evictable path for experts kept on the CPU does exist (`-ot exps=CPU`,
`--cpu-moe`, `--moe-cache-mib`). llama.cpp has no 21 GiB preflight: Qwen3.5-35B Q4_K_XL (19.3 GiB) ran at 32K
with ~1 GB headroom, so its practical ceiling is ~22–23 GiB. Neither repo has a `free -g` reading after a
llama.cpp load; that is a gap worth measuring.

## KV cache at 32K, Qwen3-Coder

llama.cpp q8_0 1.59 GiB, q4_0 0.84 GiB; vLLM bf16 3.0 GiB, fp8 1.5 GiB (unvalidated). GPT-OSS: 48 KiB/token if
every layer is stored at full length, ~24 KiB with sliding-window layers capped (llama.cpp does that by
default, `swa_full=false`; vLLM est.). Qwen3.6-35B-A3B: ~20 KiB/token (10 full-attention layers × 2 KV heads
× 256) plus a fixed GDN state per sequence.

## Worked budget: Qwen3-Coder-30B-A3B at 32K (MemTotal 30.9 GiB)

| Item | vLLM gs32, bf16 KV | vLLM, fp8 KV (est.) | llama.cpp Vulkan Q4_K_M, q8_0 KV | llama.cpp IQ4_XS, q8_0 KV |
|---|---|---|---|---|
| Weights | 17.0 | 17.0 | 17.3 | 15.3 |
| KV at 32K | 3.0 | 1.5 | 1.59 | 1.59 |
| Activations, compute, scratch | 0.8–1.3 | 0.8–1.3 | 0.5–1.0 est. | 0.5–1.0 est. |
| **Device subtotal** | **20.8–21.3 (at the ceiling)** | 19.3–19.8 | 19.4–19.9 | 17.4–17.9 |
| Host process | ~2.0 | ~2.0 | 0.3–0.5 est. (+ up to 8 GiB prompt cache unless `--cache-ram` is set) | 0.3–0.5 |
| Driver guard (0.06 × MemTotal + 450 MB): limits XPU allocation, not consumed RAM | 2.3 | 2.3 | 2.3 | 2.3 |
| Desktop session (est.) | 4 | 4 | 4 | 4 |
| **XPU free left (est.; = MemAvailable left − 2.29, which is why the guard row is subtracted)** | **~1.3–1.8** | ~2.8–3.3 | ~3.7–4.4 | ~5.7–6.4 |

## Throughput per GiB (measured tok/s ÷ resident GiB)

vLLM GPT-OSS 1.78; llama.cpp GPT-OSS 0.79; vLLM Qwen3-Coder 1.07; vLLM Qwen3.6 1.0 (1.46 with MTP);
llama.cpp Qwen3.5 IQ4_XS 0.82. The Qwen3.5/3.6 rows compare different model versions; there is no llama.cpp
number for Qwen3-Coder on this box.

## When to use which

**vLLM:** one long-lived agent model that fits under ~21 GiB (Qwen3-Coder gs32, Qwen3.6 with MTP, GPT-OSS,
Gemma 4); speed matters; more than one concurrent client; the box is rarely restarted.

**llama.cpp (Vulkan build, not SYCL until the leak question is tested):** running next to other apps, a
game or a desktop session; contexts ≥ 32K where q8_0/q4_0 KV pays off; models of 21–25 GiB via `-ot` /
`--n-cpu-moe`; frequent model switching; the fallback whenever vLLM's Level Zero leak has wedged the box.
Set `--cache-ram 0` (or ~2048) on this box.

## Corrections to the repo docs

1. `llm-scaler/LUNAR_LAKE_COMPATIBILITY.md` said "KV ~40–48 KiB/token for the 30B-class MoEs": that is
   GPT-OSS; Qwen3-Coder is 96 KiB/token at bf16 (32K = 3.0 GiB). Fixed in the same commit as this file.
2. `OpenClaw-on-MSI-Claw-8/README.md` §Performance Tuning claims `run-model.sh` applies `-fa`, `-b 4096`,
   `-ub 1024`, `--mlock`; the script passes none of them, `--mlock`/`--no-mmap` no longer exist on llama.cpp
   master (`--load-mode mlock|none|dio` replaces them), and `--vulkan` is not a flag of the script.
3. README "GPT-OSS-20B (F16) … 20B dense": GPT-OSS-20B is a MoE (~3.6B active, 32 experts, top-4); the "F16"
   GGUF keeps the experts in MXFP4.
4. README "`--parallel 1` prevents memory pressure": extra slots never multiplied KV (`-c` is split or
   shared, `llama-context.cpp:428`); the saving is only per-slot recurrent state on hybrid models.
5. `OpenClaw-on-MSI-Claw-8/LUNAR_LAKE_COMPATIBILITY.md` "VRAM Usage ~23.7GB" for LFM2-24B (a 15.75 GB file)
   is the Vulkan heap size, not usage; the XPU itself sees 28.57 GiB.
6. README calls SYCL the default backend, but every Lunar Lake number is from Vulkan; the SYCL results are
   Meteor Lake only.

## Sources

llm-scaler: `LUNAR_LAKE_COMPATIBILITY.md` §Current status; `issues/vllm-v19-lunar-lake-findings.md`
(Findings 2, 3, 5, 7; load tables; benches); `issues/vllm-xpu-status-2026-09-02.md:103`;
`issues/vllm-xpu-status-2026-11-01.md:189-190`; `issues/stack-work-plan-2026-10-29.md` (S06, S10, S12);
`issues/service-design-brainstorm-2026-10-23.md` (S20).
OpenClaw-on-MSI-Claw-8: `README.md` (benchmarks, GPU memory table, performance tuning),
`LUNAR_LAKE_COMPATIBILITY.md`, `scripts/run-model.sh`.
llama.cpp master `f3b407c`: `src/llama-model-loader.cpp:564,1366,1795-1836,1954-1961`;
`src/llama-model.cpp:1689-1716,1781`; `ggml/src/ggml-vulkan/ggml-vulkan.cpp:15631-15637`;
`ggml/src/ggml-sycl/ggml-sycl.cpp:7305-7310`; `common/arg.cpp`; `common/common.h:472-657`;
`tools/server/server.cpp:126-130`; `src/llama-context.cpp:428-440`.
Web (search snippets, HF blocked): Qwen3-30B-A3B config.json; Qwen3.5-MoE transformers docs; GGUF sizes
from prithivMLmods / AlexAtomic / AtomicChat Qwen3-Coder-30B-A3B GGUF listings.
