# vLLM on Linux vs llama.cpp on Windows on the MSI Claw 8 AI+ (2026-11-11)

**Labels used for numbers:** **(here)** means measured on this console (Linux, from the owner's repos). **(other)** means measured on other hardware, which is named. **(est.)** means an estimate or my arithmetic.
**Labels used for claims:** [confirmed] passed the skeptic check. [corrected] was changed, and the corrected text is the one shown. [unverified] could not be checked. **Nobody has measured anything on Windows on this console.**

---

## 1. Verdict

- **Keep Linux as the service OS for J1 and J2.** No Windows stack beats vLLM on this box for the job that matters most, agentic coding with long prompts. vLLM prefill is 756-1000 tok/s at ~1K prompts (here), and a cold 16K prompt took 20.3 s (here, GPT-OSS-20B on an older stack). The only long-prompt Windows figure on a 140V is Ollama Vulkan at 83 tok/s on an 11.7K prompt (other: 258V, 2026-10). That is roughly 9x slower [confirmed].
- **"Windows consumes more memory": true, but by less than assumed, and it is not the real problem.** Idle Windows on Claw 8 AI+ units uses 6.2-8.6 GB (other: ETA Prime 7.3 desktop / 6.2 Xbox Full Screen Experience; MSI's own screenshots 8.6 / 7.8). Linux KDE plus a browser on this box uses about 3.9 GiB (here, inferred from psutil.available). The gap is about 2-3.5 GB [corrected; it was "2-5 GB"]. llama.cpp's saving of 1.5-3 GiB over vLLM roughly cancels that gap, so on RAM alone, Windows+llama.cpp vs Linux+vLLM is close to even (est.). The real Windows cost is the iGPU budget. Intel's default caps it at 57% (~18 GiB). It needs an admin-only override plus a reboot to reach ~27.3 GiB (other: 258V at 87%), and driver updates have reset it to 57% [confirmed].
- **"Windows is more GUI-friendly": mostly false for these jobs.** LM Studio, Jan and Ollama ship for Linux too, and on this iGPU they all run the same llama.cpp Vulkan engine. Open WebUI and Jan can sit in front of the existing vLLM endpoint. What is truly Windows-only is small: Intel AI Playground (beta, no documented API), the touch keyboard that pops up by itself, and Full Screen Experience. J1 and J2 are headless API jobs where no GUI helps.
- **Gaming is the one real reason to boot Windows.** Kernel anti-cheat titles are Denied on Linux (Fortnite, Valorant, Apex, Destiny 2, GTA V Online) [confirmed]. Secure-Boot-gated titles (BF6, BO7, Valorant) will not run on the Windows side of this dual boot either, because Secure Boot is off for Nobara [confirmed in owner notes; the per-game list is unverified].
- **If you leave vLLM, the evidence points to llama.cpp on Linux, not on Windows.** It keeps Linux's predictable 28.57 GiB XPU budget (here), avoids vLLM's Level Zero leak and its 36-40 s loads (here), and offers router and unload-on-idle. That makes it the lowest-ops and most gaming-friendly option.
- **Decode speed is closer than the repos suggest.** The box's llama.cpp numbers (9.5-16.9 tok/s, here) come from a March 2026 build. Current builds reach ~25 tok/s on A3B MoEs: 25.8 tok/s on Qwen3.6-35B-A3B with Linux SYCL (other: 288V), and ~25 tok/s on Qwen3-30B-A3B with Windows Vulkan in partial offload at 1K context (other: Claw 8). vLLM's clear decode lead is now MTP (27.7 tok/s, here), along with prefill and batching [corrected].
- **The one Windows stack that beats vLLM on decode is OpenVINO Model Server, not llama.cpp.** It reached a 38 tok/s median on Qwen3-Coder-30B-A3B INT4 (other: 258V, Win11, 87% override, ~2.2K context). That gain comes from the runtime, not the OS: OVMS also runs on Linux in Docker [corrected; the mean, P95 and 480 tok/s prefill figures are not visible in the source].

## 2. Decision matrix

Columns: **A** = Linux + vLLM XPU (today). **B** = Linux + llama.cpp (Vulkan). **C** = Windows + llama.cpp (Vulkan; SYCL is flakier). **D** = Windows + OVMS (OpenVINO). D is the strongest Intel-native Windows option the research supports. WSL2 vLLM has no published working run on an Intel iGPU, and AI Playground is a GUI, not a server.

| Row | A Linux + vLLM | B Linux + llama.cpp | C Windows + llama.cpp | D Windows + OVMS |
|---|---|---|---|---|
| OS idle memory | ~3.9 GiB KDE+browser (here, inferred); headless less (est.) | same as A | 6.2-8.6 GB (other: Claw units) | same as C |
| Usable iGPU budget | 28.57 GiB to Level Zero (here); ~21 GiB practical (here) | ANV heap ~23.7 GB (here); ~22-23 GiB practical (here, derived) | default 57% ≈ 18 GiB (other: 18409 MB on a Claw 8); 85-87% ≈ 27.3 GiB (other: 258V) | same cap as C |
| Largest model + context | Qwen3.6-35B INT4 18-20 GiB; Qwen3-Coder gs32 at 32K at the ceiling (est.) | Qwen3.5-35B Q4_K_XL 19.3 GiB at 32K, ~1 GB headroom (here) | default: ~15-16 GiB (est.); override: ~20-22 GiB (est.) [unverified] | 30B MoE fits at 87%; 30B + 14B do not co-reside (other) |
| Qwen3-Coder-30B-A3B at 32K | device 20.8-21.3 + host 2.0 GiB (est. from parts measured here) | IQ4_XS 17.4-17.9 / Q4_K_M 19.4-19.9 + 0.3-0.5 GiB (est.) | same as B (est.) | ~16 GB weights; "19 GB+ VRAM" recommended (vendor doc) |
| Decode tok/s | 18 Coder gs32 (v0.19); 19 / 27.7 MTP Qwen3.6 (v0.30, MXFP4); 23 GPT-OSS (v0.19); 30 Gemma 4 (here) | 9.5-16.9, old build (here); 25.8 Qwen3.6 SYCL (other: 288V) | ~25 Qwen3-30B partial offload, c=1024 (other: Claw 8); 17.2 Ollama (other: 258V) | 38 median, 31 accuracy-safe (other: 258V); 15.7 at ~11.7K (other: OpenVINO, 16 GB 258V) |
| 16K agent prompt TTFT | cold 20.3 s, warm 0.4 s (here, GPT-OSS, older stack); Qwen3.6+MTP re-prefills every turn | ~70-95 s cold (est. from 159-235 tok/s here); prompt cache when warm | ~3 min (est. from 83 tok/s, other) | ~60 s (est. from 260 tok/s at 11.7K, other) |
| Concurrency | paged batching: 5 streams 49 tok/s aggregate (here, older stack) | continuous batching, unified KV; not measured | same code; Windows read-back ~10x slow on slot swaps (other: 140T) | continuous batching; not measured on a 140V |
| Model switch | 36-40 s load (here, v0.19) + leak risk | ~5-15 s (est.); router, `/models/unload`, sleep-idle | same as B (est.) + reboot to change OS | not found |
| GUI apps | Open WebUI / Jan via OpenAI API | LM Studio, Jan, Ollama Linux builds + Open WebUI | LM Studio (iGPU off by default), Jan, Ollama (`OLLAMA_IGPU_ENABLE=1`), AI Playground | AI Playground (beta); Open WebUI against OVMS |
| Agent harnesses | native; `qwen3_coder`/`gemma4`/`openai` parsers | native; `--jinja` default, Qwen3-Coder XML parser | pi needs Git Bash; OpenClaw recommends a WSL2 gateway; nanobot native | same as C; parsers `qwen3coder`, `gptoss`, `gemma4` |
| Always-on reliability | systemd; Level Zero leak after crashes, no restart loops, local patches | systemd; no leak recorded | Win11 Home: monthly reboots, 35-day pause; single-reporter TDR/garbage bugs; override resets | same Windows ops; ovms.exe can run as a service |
| Gaming coexistence | worst: must stop the engine; leak may hold memory | good: unload/sleep on idle | anti-cheat titles available (except Secure-Boot-gated); system-wide override | same as C |
| Ops burden | high (patches, 1.5-2 h kernel builds) | low | medium-high | medium-high (IR exports, override) |

## 3. Memory: worked budget, one 30B-A3B at 32K on 32 GB

Model: Qwen3-Coder-30B-A3B. KV is 96 KiB/token at bf16 (est., from config; confirm with the startup line), so 3.0 GiB at 32K, or 1.59 GiB at q8_0 (est.). MemTotal is 30.9 GiB (here); I assume the same usable total on Windows (est.; the ~1.1 GiB firmware reserve applies to both OSes). "OS + desktop" uses the inferred 3.9 GiB on Linux, with the repo's conservative 6.3 GiB (2.3 reserve + 4 desktop) shown in brackets.

| Item (GiB) | A vLLM gs32, bf16 KV | B llama.cpp IQ4_XS, q8_0 KV | C Windows llama.cpp IQ4_XS, q8_0 KV | D Windows OVMS int4-ov, u8 KV |
|---|---|---|---|---|
| Weights | 17.0 (here) | 15.3 (est., GGUF size) | 15.3 | ~15-16 (vendor disk size) |
| KV at 32K | 3.0 | 1.59 | 1.59 | ~1.5 (est.) |
| Activations / compute | 0.8-1.3 (est.) | 0.5-1.0 (est.) | 0.5-1.0 (est.) | not found |
| Host process | ~2.0 (here) | 0.3-0.5 (est.); set `--cache-ram`, default 8 GiB | 0.3-0.5 (est.) | not found |
| OS + desktop | 3.9 [6.3] | 3.9 [6.3] | 6.2-8.6 (other) | 6.2-8.6 (other) |
| **Left (est.)** | **3.3-4.2 [1.3-1.8]** | **8.1-9.3 [5.7-6.4]** | **3.9-7.0** | **~3-7** |
| GPU-cap check | 28.57 limit: OK | ~23.7 heap: OK | device subtotal 17.4-17.9 vs ~18 GiB default: marginal; Q4_K_M (19.4-19.9) needs the override | "19 GB+" > 57% default: override needed |

Takeaways:
- **The Windows shared-GPU limit decides what fits, not idle RAM.** Microsoft's own formula gives ~50% on 32 GB [confirmed]. Intel's Lunar Lake default is 57% [confirmed]. The Shared GPU Memory Override (Intel Graphics Software run as admin, driver 32.0.101.6987+, reboot) raises it. One 258V reached a 27.3 GiB window at 87% [confirmed], about 1.3 GiB below Linux. The maximum the slider offers on a 32 GB Claw is [unverified]. The registry value is `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager\SystemPartitionCommitLimitPercentage`, and driver 32.0.101.9150 reset it to 57 (other) [confirmed].
- Whether the "(16GB)" in llama.cpp #18946 is a 16 GB cap or just the device-name string is disputed [unverified]. Measure `--list-devices` on Windows.
- Windows also counts allocations against the commit limit (RAM + pagefile), so keep the pagefile system-managed. The claim that Windows pages out an idle llama-server's Vulkan weights is [unverified]: VidMm, not working-set trimming, manages those allocations.
- Linux trap: zram makes INT4 weights thrash (owner note, here). Keep the 16 GB disk swapfile from the post-install script, and keep at least 2 GiB MemAvailable.
- Headless Linux (`systemctl isolate multi-user.target`) has no measured figure. Headless gain is TBD (1.5–3.9 GiB est.) until measured; 3.9 GiB is the inferred OS + desktop total, not the desktop share alone. The bracketed [6.3] figures above count the 2.29 GiB driver guard as consumed RAM; it is not.

## 4. Speed and features by job

**J1 coding agent (pi, Qwen Code over SSH).** Prefill decides this job. vLLM: 756 tok/s on Qwen3-Coder at ~1K (here). A cold 16K prompt took 20.3 s and a warm prefix cache 0.4 s (here, GPT-OSS, older stack); all vLLM figures above 1K for Qwen3-Coder are extrapolations [corrected]. Qwen3.6 with MTP (27.7 tok/s) requires `--no-enable-prefix-caching`, so a 20K context costs about 20-27 s on every turn (est.). For coding, that makes Qwen3-Coder gs32 with prefix caching the better default. llama.cpp's cold long-prompt prefill is 3-9x slower on any measured build and OS. Its prompt cache helps append-only turns, but not first turns or compaction. Windows adds no feature for J1 and adds harness friction (Git Bash).

**J2 two Telegram bots.** Concurrency and uptime matter. vLLM's paged batching is measured here. llama-server has continuous batching with unified KV on both OSes, but nobody has measured it here. On Windows, slot swaps and checkpoints were ~10x slower on device-to-host reads (other: 140T, open issue) [confirmed for the 140T only]. #18946 also reported output corruption after ~10 requests at the 57% setting (other: Claw 8, single unconfirmed report). That is a bad sign for always-on bots [unverified as a general defect].

**J3 voice/OCR.** Windows has Foundry Local / the OpenVINO EP for the NPU, and Whisper-base ran well on a Lunar Lake NPU (other). OpenVINO runs on Linux too, but the llama.cpp OpenVINO backend failed on K-quants on this box (here). This is a mild plus for Windows, and it does not justify moving the LLM.

**Gaming.** Windows unlocks the anti-cheat titles, and Lunar Lake game FPS is about even across OSes on geomean (other: Phoronix 2025, snippet only). Running a game and inference together collapses the resident model's decode on either OS: Qwen3-14B dropped from 19.8 to 0.19-3.27 tok/s while other GPU work ran (other: 258V). Plan on handing the bots off during play, not on sharing the GPU.

**Corrections and rejections:** "Windows Vulkan prefill beats Linux SYCL (865 vs 536)": [unverified]. The 865 row is labeled only "Core Ultra 200 Series", with no OS given. "SYCL decode collapses on Lunar Lake": [corrected] to "SYCL in llama-server is flaky on Intel iGPUs". The 3x collapse was on Meteor Lake; on Lunar Lake the failure was empty output after the first request. "MTP is slower in llama.cpp on 140V Windows": [unverified]; the source could not be fetched. Windows-specific Vulkan code paths are [corrected] to tuning choices, except the FWHT crash gate for drivers 101.8509-101.8860.

## 5. GUI: what the advantage really is

What it consists of: (1) one-click apps whose Windows builds are more polished, though the same apps (LM Studio AppImage, Jan, Ollama) exist on Linux with the same engine. (2) Intel AI Playground: every recent release is a beta (v3.3.0, 2026-10-30), it has an experimental Ubuntu-only Linux build, no documented OpenAI API and few KV/context controls [confirmed]. (3) The touch keyboard popping up on text fields. On Nobara Desktop Mode you have to press Steam+X. (4) Full Screen Experience: it is not Insider-only. MSI shipped it for all Claws, and it became widely available around 2025-11-21. It saved ~1.1 GB on a Claw 8 AI+ (other), not the headline ~2 GB. The claim that it was Insider-only is **refuted**.

What counts against Windows: LM Studio 0.4.17+ leaves Vulkan iGPUs disabled by default (one toggle) [confirmed]. Ollama needs `OLLAMA_IGPU_ENABLE=1`, seen only on AMD so far [corrected]. Every GUI hides the shared-memory override, which is exactly where #18946 failed. The dual boot runs with Secure Boot off, so BitLocker asks for the 48-digit recovery key on every boot unless device encryption is off (owner README) [confirmed].

How to get a GUI on Linux + vLLM:
- **Open WebUI** in Podman on the Claw or a Spark: Admin Settings → Connections → `http://<claw>:8000/v1`. Install it as a PWA on the phone and laptop over Tailscale.
- **Jan** custom OpenAI provider (base URL ending in `/v1`) on the laptop.
- For handheld use, add Open WebUI to Steam as a non-Steam shortcut through a Firefox kiosk wrapper, and type with Steam+X.

## 6. Recommended setup (hybrid)

**Role split.** Nobara is the always-on service: vLLM for J1 and J2, with llama.cpp on Linux as the low-ops fallback and gaming-day engine. Windows is for anti-cheat games only. During any Windows or gaming session the bots fail over to a DGX Spark (already in the owner's design). Optionally run a local llama-server on Windows for offline chat, reading GGUF files from a shared partition.

**Linux, vLLM (primary):**
```
SYCL_UR_USE_LEVEL_ZERO_V2=0   # never source oneAPI setvars at runtime
vllm serve <Qwen3-Coder-30B-A3B sym INT4 gs32> --enforce-eager \
  --max-model-len 24576 --kv-cache-memory-bytes 2415919104 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```
- Qwen3-Coder: 24576 with bf16 KV until FP8 KV is validated (work plan S12a), then `--max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1610612736`. Do not size KV from `--gpu-memory-utilization`.
- For Qwen3.6 with MTP, use the owner's existing MTP k=2 config and add `--no-enable-prefix-caching`. Use it for bots and short-context work, not long coding sessions.
- Unit: `Restart=on-failure`, `RestartSec=60`, `StartLimitBurst=2`, a graceful stop signal (SIGINT/SIGTERM, never SIGKILL), `TimeoutStopSec=60`, and an `ExecStartPre` check that refuses to start if XPU free memory is below ~22 GiB.
- System: zram off, 16 GB disk swapfile, sleep masked on AC, the WiFi D3cold units from the post-install script, and game-watch that stops vLLM and reroutes the bots.

**Linux, llama.cpp (fallback / gaming-day):** `llama-server` Vulkan in router mode: `--models-dir /shared/models/gguf --models-max 1 --models-memory-margin 2048 --sleep-idle-seconds 600`. Per-model preset: `-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 32768 --cache-ram 2048 --jinja`. Use this whenever the leak wedges vLLM, or on days the box is mostly a game console.

**Windows (gaming sessions only):**
- Intel Graphics Software as Administrator: set Shared GPU Memory Override to 85%, reboot, and re-check the registry value after every driver update.
- Pagefile system-managed. Defender exclusion for the model folder and `llama-server.exe`. Fast Startup off, so Linux can mount the shared NTFS partition read-write.
- `llama-server` from the b12110 win-vulkan zip: `-m D:\models\<model>.gguf -ngl 99 -c 32768 -ctk q8_0 -ctv q8_0 -fa on --cache-ram 2048 --jinja --host 127.0.0.1`. Use `--fit off` only for testing.
- If the driver resets (TDR): `set GGML_VK_DISABLE_COOPMAT=1`. If output is garbage: `-b 2048 -ub 512`. Record the driver version with every result. Unload before launching a game.

**Shared model partition:** NTFS, mounted with ntfs3 on Nobara, holding GGUF files only. vLLM's safetensors are useless on Windows, so they stay on btrfs. exFAT works too, but breaks the symlinked Hugging Face cache layout.

**Switching OS:** `grub2-reboot "Windows Boot Manager"` with `GRUB_DEFAULT=saved`, so a power cycle returns to Nobara.

## 7. Measurements to take on the device (in order)

1. **Windows GPU budget.** At the default and again at 85%: `llama-server --list-devices` (win-vulkan), `vulkaninfo --summary`, the registry value, and Task Manager "Shared GPU memory". *Pass:* at 85%, a device of ≥26 GiB, and it survives a reboot.
2. **Idle RAM, like for like.** After a cold boot plus 5 min: Windows desktop and FSE (Task Manager In use / Committed); Nobara KDE, Gaming Mode and multi-user.target (`free -m`, MemAvailable). *Pass:* the gap is known within ±0.5 GiB. If it is ≤3 GiB, the memory half of the premise is settled as minor.
3. **Same-GGUF speed, both OSes.** Same llama.cpp commit, Qwen3-Coder-30B-A3B IQ4_XS: `llama-bench -ngl 99 -fa 1 -p 512,8192,16384 -n 128 -r 3` with Windows Vulkan, Nobara Vulkan and Nobara SYCL. *Pass:* every run completes. Record the decode delta and pp16384, and compare against vLLM's 18 tok/s and ~756 tok/s.
4. **Full fit on Windows.** `-c 32768 -ngl 99 --fit off`, Q4_K_M and IQ4_XS. *Pass:* both load with no ErrorOutOfDeviceMemory or DeviceLost, and peak Committed stays below 90% of the commit limit.
5. **Agent TTFT.** Send a captured ~20K pi or Qwen Code request twice, the second time with ~500 tokens appended, to vLLM (Coder gs32; Qwen3.6 with MTP on and off) and to llama-server on each OS. *Pass:* vLLM warm TTFT ≤2 s and cold ≤30 s. Windows llama.cpp is acceptable for J1 only if cold ≤60 s.
6. **Stability soak, Windows llama-server.** 30 min of agent prompts at `-ub 512` and `-ub 128`; then 2 h of bot-style traffic, with and without `GGML_VK_DISABLE_COOPMAT=1`. *Pass:* no Event 4101 TDR, no NaN perplexity, no corruption after 10+ requests.
7. **vLLM clean-exit leak on v0.32.** `systemctl stop`, then read XPU free memory and MemAvailable, then repeat after drop_caches. *Pass:* ≥26 GiB XPU free without a reboot. If it fails, use llama.cpp on gaming days.
8. **Concurrency.** `vllm bench serve --max-concurrency 1,2,3` (2048 in / 512 out) against vLLM and against llama-server with `-np 3 --kv-unified -c 49152`. *Pass:* the engine chosen for J2 holds TPOT ≤100 ms at 3 streams.
9. **OVMS on both OSes.** Qwen3-Coder int4-ov on ovms.exe at 85%, and the same in Docker on Nobara (`--device /dev/dri`). *Pass:* decode >25 tok/s at 8K and ≥20 harness tool calls without malformed output. If Linux matches Windows, OVMS stays on Linux.
10. **Gaming coexistence.** One mid-weight game, with the model unloaded and then resident. Record fps, OOM kills and tok/s, then time the full GRUB → Windows → Nobara → service-ready cycle. *Pass:* no OOM; the service is back in under 3 min.
11. **Windows always-on (only if 3-6 pass).** Boot to the lock screen with an S4U boot task and Tailscale "Run unattended". *Pass:* the endpoint answers over the tailnet without logon, and the 140V is visible from that session.
12. **Your game list.** Check each title against AreWeAntiCheatYet and its Secure Boot requirement. *Pass:* a list that says how often you actually need Windows.

## 8. What could not be verified

- Any Windows measurement on this console, so every Windows cell above is other-hardware or est.
- The maximum override the slider offers on a 32 GB Claw. The 93% figure is a single source. Whether "(16GB)" in #18946 is a cap or a device name.
- Whether Windows pages out an idle llama-server's Vulkan weights.
- vLLM prefill above 1K on Qwen3-Coder and Qwen3.6 (only GPT-OSS at 16K was measured, on an older stack). Qwen3.6 prefill was never measured.
- llama.cpp numbers on a current build on this box. GPT-OSS "F16 dense" is probably a mislabel, and the Q4_K_XL 9.5 tok/s run had only ~1 GB of headroom.
- OVMS mean/P95, the 480 tok/s prefill, the INT4 group size and long-context behaviour. The NoLlama 260 tok/s figure is OpenVINO, not OVMS, on a 16 GB laptop.
- vLLM-XPU in WSL2 on any Intel iGPU: no published run; community reports say it fails.
- llama.cpp MTP speed on Lunar Lake on either OS. The vLLM XPU sleep mode (probe S07) has not been run.
- Whether the Windows-only 140V bugs (#20554 coopmat TDR, #28648 garbage output) persist on drivers after 8531/8992. Both are single reporters, closed as stale.
- Idle RAM for Nobara Gaming Mode and headless; MSI Center M's share of Windows RAM; the Claw's reboot time; a battery charge limit on either OS.
- Several sources were reachable only as search snippets: intel.com, notebookcheck, pcguide, tomshardware, phoronix, lmstudio.ai, docs.ollama.com and the Fedora forums.
