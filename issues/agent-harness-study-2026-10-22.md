# Agent harness study for a local-LLM Lunar Lake console (2026-10-22)

Scope: which agent harness(es) to run on top of vLLM-XPU on the MSI Claw 8 (Arc 140V, 32 GB UMA,
`http://localhost:8080/v1`, Gemma 4 26B-A4B INT4 at ~30 tok/s decode / ~1000 tok/s prefill / 32K
context, Qwen3-Coder-30B-A3B at 18 tok/s; GDN-hybrid models with MTP need prefix caching OFF).
Jobs: (1) agentic coding over SSH from a laptop; (2) two Telegram bots with memory; (3) voice/OCR
endpoints later; (4) offline, open source, low operational burden, routable to DGX Sparks at home.

Method: one researcher per harness filling a fixed 20-field profile from primary sources (repos,
docs, issue trackers), then independent skeptics on the critical claims (32 verdicts; 21 produced
corrections, folded in below). The three "judge" rankings were lost to container restarts; the
scores in §4 are the author's, derived from the verified profiles. Tags: **[confirmed]** skeptic
passed, **[corrected]** skeptic amended the claim (text below is the corrected version),
**[unverified]** could not be checked (Hugging Face / several doc sites are egress-blocked here).

## 1. TL;DR

- **Keep OpenClaw for the Telegram bots, stop using it for coding.** It is a near-perfect structural
  fit for two bots (`channels.telegram.accounts` + `bindings` → separate agents, workspaces,
  models, MEMORY.md) and its old 16K/32K context guard is gone (now ratio-based) — but it is the
  wrong tool for terminal coding, and its churn/security history argues for not expanding its role.
- **Install pi for terminal coding today.** ~1.2K tokens of fixed prompt per turn [confirmed] — an
  order of magnitude below everything else — four tools, MIT, TUI works over SSH/tmux, runtime
  model switching between the Claw and a Spark. It is the only harness whose per-turn cost stays
  trivial when prefix caching is off (GDN + MTP models). Needs four lines of 32K tuning (§3).
- **Try Qwen Code when the model is Qwen3-Coder.** It is the one coding harness with both native
  tool calls *and* an XML text fallback, and its prompt swaps in `<tool_call><function=…>` examples
  that match vLLM's `qwen3_coder` parser exactly. Price: 15–22K tokens of resident prompt — fine
  with prefix caching on, unusable with it off. It also has native Telegram channels with per-channel
  model/cwd/memory, so it could cover job 2 as well.
- **Hermes Agent is the best Telegram assistant on paper and refuses your 32K models.** Hard
  64,000-token floor [confirmed]. It becomes viable only if you serve Gemma 4 at `--max-model-len
  65536` (feasible: ~2.6 GiB of KV at 40 KiB/token on a 16 GiB model), and its hidden auxiliary LLM
  calls need taming on a 30 tok/s box.
- **nanobot is the lightest drop-in OpenClaw replacement** if you do want to leave: first-class
  `vllm` provider block, JSON text-fallback tool parsing with corrective retries, fallback chains
  sized to the smallest window, Telegram with pairing, file memory with scheduled consolidation,
  ~5–8K tokens/turn (est.). Bus factor ~1–2.
- **Do not adopt:** Codex CLI (Responses-only, no `apply_patch` tool for non-OpenAI models, MCP
  tools in a proprietary `namespace` type, 9–13K tokens/turn, no external contributions accepted);
  Claude Code *on the Claw* (first request ~27–33K tokens, 100K compaction floor, Messages API only)
  — it is fine against a Spark-hosted 128K+ model; Cline/Kilo (native-only tools, 4.1.x broke local
  Qwen, Roo is dead); PicoClaw (stale since Aug); Antigravity IDE (cloud brain only).
- **Aider is the emergency harness.** No tool calls at all, so parser failures are impossible;
  dormant since May 2026 but stable; coding-only, manual `y` for every shell command.
- Harness choice is now cheap: every candidate speaks to the same endpoint, so after the frozen
  vLLM image exists, switching is a config file, not a migration.

## 2. Comparison matrix

| Harness | License / runtime | Local endpoint | Tool-call mechanism | Fixed prompt / turn | 32K context handling | Messaging / memory | MCP | Sandbox | Small-model evidence | Verdict (this user) |
|---|---|---|---|---|---|---|---|---|---|---|
| **pi** (earendil-works/pi, ex badlogic/pi-mono) | MIT / Node ≥22 | `~/.pi/agent/models.json` providers, `api: openai-completions` | native only; no text fallback [confirmed] | **~1.2K** [confirmed] | defaults assume 128K/64K; set `contextWindow`, `reserveTokens`, `keepRecentTokens` [confirmed] | none native (3rd-party Telegram ext.); AGENTS.md, no memory | yes (codemode) | none; project trust only | HF hub docs run Gemma 4 26B via llama-server; #10540 Qwen3.5 text-instead-of-tool (unreproduced) | **Coding: yes** |
| **Codex CLI** (openai/codex) | Apache-2.0 / Rust binary | `[model_providers.x] base_url`, `wire_api="responses"` only | native Responses only; unknown models get **no `apply_patch` tool** [confirmed] | ~9–13K (est.) | assumes 272K; set `model_context_window` | none; memories call hard-coded OpenAI models | wrapped in `namespace` type — breaks on llama.cpp/LM Studio, vLLM unknown [corrected] | bubblewrap (good) | #51077 Qwen3.6 tool calls as text; #44529 schema contradiction; no maintainer replies | **No** |
| **Claude Code** | proprietary / binary | Anthropic Messages API only (vLLM serves `/v1/messages`) | native only | **~27–33K** first request, ~2K with `--bare --tools` [corrected] | autocompact window floor 100K (= min(100K, window)); context-400 needs manual `/compact` [corrected] | Telegram/Discord channels native but **Anthropic-auth gated**; CLAUDE.md + auto-memory | full | modes + bubblewrap; auto mode uses your local model as classifier (#78372 loop) | #73165/#88536/#101897 local regressions, stale-bot closures | **No at 32K; yes on a Spark 128K model** |
| **Hermes Agent** (NousResearch) | MIT / Python 3.14 + Node | `provider: custom, base_url, api_mode: chat_completions, context_length` | native only [confirmed] | ~15K (6.8K stable prefix + ~7.3K tools + skills + memory) [corrected] | **refuses < 64K** [confirmed] | **best**: multi-profile gateway, MEMORY/USER.md, FTS5 recall, cron, local whisper/piper | full (client + server) | docker/ssh backends; `approvals.mode: smart` = extra LLM call | DGX Spark + AMD playbooks on vLLM Qwen3.6-35B; #87075 9B refused by floor | **Telegram: yes if served at 64K** |
| **OpenClaw** (current) | MIT / Node 24–26 | bundled `vllm` provider (`openai-completions`) | native only; no text promotion by policy, `tool_choice: required` workaround [corrected] | ~8–15K stock (3.5K @21 tools, +4K skills) [corrected] | ratio-based guard (block <10 %, warn <20 % of native window) — 32K is fine; set `contextWindow` [corrected] | **best structural fit**: multi-account Telegram + bindings; MEMORY.md + embeddings | yes (deferred via tool search) | off by default; docker/podman optional | #49508 Qwen3-Coder XML under responses API; #117787 retry loop (fixed) | **Telegram: keep** |
| **OpenCode v2** (anomalyco) | MIT / Bun binary | `vllm` auto-discovery (port 8000, `owned_by=="vllm"`) or custom provider `settings.baseURL` | native only; MCP behind Code Mode `execute` by default [confirmed] | ~3–5K (est.) | compacts at window−max(15 %,16K) → ~16.8K on 32K; set `compaction.buffer`, `keep.tokens` | **none** — gateway request closed "not planned" Oct 21 [corrected]; no memory | yes | permission rules, no OS sandbox | #53623 Gemma-4-31B confused by Code Mode; #22792 Qwen3-Coder-30B compaction loops at 32K | Coding: good, second to pi |
| **Qwen Code** (QwenLM) | Apache-2.0 / Node ≥22 | `OPENAI_BASE_URL` or `modelProviders.openai[]` with `baseUrl`, `wireApi`, `contextWindowSize` [confirmed] | **native + XML text fallback**; Qwen-coder few-shots match `qwen3_coder` parser | **15–22K** (22.3K measured on 0.19.3 incl. 6K auto-memory) [corrected] | window inferred from model id only (`qwen3.x` → 1M!) — set `contextWindowSize` [corrected] | native Telegram channels (per-channel model, cwd, memory); QWEN.md + auto-memory | yes | bwrap/Landlock | #7148 Gemma 4 fixed via examples; open vLLM 400s (tools w/o params, `tool_choice: required`) | **Coding with Qwen3-Coder: yes** (prefix cache on) |
| **Goose** (aaif-goose, ex Block) | Apache-2.0 / Rust | `OPENAI_HOST` + base path, or custom-provider JSON with `context_limit` | native only on OpenAI path (`strict:false`, no `tool_choice`; `request_params` escape hatch) [corrected] | ~1.3K with developer ext only (est.) | default 128K; discovery reads `meta.n_ctx` (llama.cpp/Ollama), **not** vLLM `max_model_len` → set `context_limit` [corrected] | Telegram gateway: 1 bot per config root, no groups, 5-turn cap; memory ext concatenated unbounded | MCP-native | `GOOSE_MODE` **default auto**; repetition guard is a no-op [confirmed] | #8581 Gemma 4 26B on vLLM worked; #10784 recipe loop | Coding: OK; Telegram: no |
| **Aider** | Apache-2.0 / Python ≤3.12 | `OPENAI_API_BASE` + `openai/<model>` (litellm) | **no tool calls** — plain-text edit formats [confirmed] | ~0.6K (whole) – 2.5K (diff) | unknown model → `whole`, no repo map, 1K history budget: add `.aider.model.settings.yml` + metadata [corrected] | none; git history | none | none; shell cmds always need `y` | leaderboard rows for Qwen3-235B on vLLM; none for your models | **Fallback coding harness** |
| **Cline CLI / Kilo** (Roo dead) | Apache-2.0 / MIT, Node/Bun | Cline: `cline auth -p openai-compatible -b …`; Kilo: `kilo.jsonc` | native only; Cline 4.1.x removed XML toggles (#13008) | Kilo ~15K; Cline ~12K (reports) | Cline estimates chars/3 → compacts at ~55–60 %; Kilo 32K "Compaction exhausted" bug | Cline CLI: native Telegram/Slack/Discord connectors on a local hub | yes | Cline CLI **auto-approve default on**, no allow/deny | #14453 Gemma 4/Qwen2.5 tool JSON as text; #14827 empty args | No |
| **nanobot** (HKUDS) | MIT / Python ≥3.11 | first-class `providers.vllm {apiBase, apiKey:"EMPTY"}` + presets with `contextWindowTokens` | native + JSON text fallback + corrective retry; per-session rate limit | ~5–8K (est.); prompt rebuilt each turn (#2463) | per-preset window; fallback chain fits smallest | native Telegram (pairing/allowFrom); 2 bots = 2 `--home` instances; SOUL/USER/MEMORY + Dream cron | yes | bwrap | #2829 gemma4:e4b no tools on Ollama (open); #1068 Qwen forgets at 60–70 % window | **Best drop-in if leaving OpenClaw** |
| **ZeroClaw** | MIT/Apache / Rust binary | `[providers.models.<family>.<alias>] uri, model, wire_api, context_window` | native, or `native_tools=false` prompt carrier; `local_small` profile (8K budget, 4 iterations, strict parsing) | ~2–4K (est.) | explicit `max_context_tokens` / ratio | Telegram multi-alias → agents; SQLite memory + embeddings (offline? unverified) | yes | landlock/firejail/bwrap | #11296 custom uri ignored in model listing; #9177 cold-load disconnect | Worth a trial; bus factor ~1 |

## 3. Per-harness notes and config

### pi — coding harness of choice
- Fixed overhead ~610 tokens system prompt + ~600 tokens of four tool schemas [confirmed]; skills add
  only name/description; MCP tools hidden behind codemode by default.
- Tool calls come only from streamed `tool_calls` deltas — correctness = vLLM parser correctness.
  Launch vLLM with `--enable-auto-tool-choice --tool-call-parser gemma4` (Gemma 4) or
  `qwen3_coder` (Qwen3-Coder). [confirmed]
- 32K tuning is mandatory: `contextWindow` defaults to 128000 even when the server advertises less
  (#9566 open), compaction defaults (`reserveTokens` 16384, `keepRecentTokens` 20000) assume ≥64K.
  [confirmed]
- Churn: ~2 releases/week with occasional breaking changes; local-model issues sometimes auto-closed.
  Pin a version.

```json
// ~/.pi/agent/models.json
{ "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "vllm-claw":  { "baseUrl": "http://localhost:8080/v1", "api": "openai-completions", "apiKey": "none",
      "models": [ { "id": "gemma-4-26b", "contextWindow": 32768, "maxTokens": 4096,
                    "compaction": { "reserveTokens": 6144, "keepRecentTokens": 8192 } } ] },
    "vllm-spark": { "baseUrl": "http://spark.tailnet:8000/v1", "api": "openai-completions", "apiKey": "none",
      "models": [ { "id": "<big-model>", "contextWindow": 131072 } ] } } }
```
Run with `PI_OFFLINE=1 PI_TELEMETRY=0 pi -c` inside tmux over SSH; switch models with `/model`.

### Qwen Code — when the model is Qwen3-Coder
- Native support for local endpoints via env or `modelProviders` [confirmed]; the system prompt picks
  Qwen-coder XML tool examples for `qwen*-coder` ids, and an XML fallback parser catches calls the
  model emits as text — the most forgiving coding harness for a 3B-active model.
- Resident prompt 15–22K tokens [corrected] — only workable with vLLM prefix caching ON (GQA models:
  Gemma 4, Qwen3-Coder). Do not pair with GDN + MTP configs.
- Set `contextWindowSize` explicitly: window is inferred from the model id, never probed
  (`qwen3.x` → 1,000,000). [corrected]
- Open vLLM wire bugs (Oct 2026): tools without parameters serialized without `parameters`, and
  side-queries sending `tool_choice: "required"` → HTTP 400; fixes are unmerged opt-in PRs. Keep
  `tools.eager` minimal.

```json
// ~/.qwen/settings.json
{ "security": { "auth": { "selectedType": "openai" } },
  "modelProviders": { "openai": [ { "id": "Qwen3-Coder-30B-A3B", "baseUrl": "http://localhost:8080/v1",
      "envKey": "OPENAI_API_KEY", "wireApi": "chat-completions", "contextWindowSize": 32768 } ] },
  "general": { "enableAutoUpdate": false }, "privacy": { "usageStatisticsEnabled": false } }
```
Export `OPENAI_API_KEY=none QWEN_CODE_MODELS_DEV_REFRESH=off`.

### OpenClaw — keep for Telegram
- Two bots: `channels.telegram.accounts.<id>.botToken` + top-level `bindings` routing each account to
  its own agent (workspace, model, tool profile, MEMORY.md). This is the feature the alternatives
  are still catching up to.
- Context guard is ratio-based since Apr 30 2026 (commit 13e917e): 32K models are not at any edge;
  set `contextWindow: 32768` on the provider and `compat.supportsUsageInStreaming` for the meter.
  [corrected]
- Tool calls must arrive as structured `tool_calls` (policy #60601); per-model
  `params.extra_body.tool_choice: "required"` is the documented workaround for stubborn models.
  Keep `tools.profile: messaging` on the bots to shrink the schema block (~3.5K @21 tools).
- Risks: stable releases every 2–4 days, one dominant committer, 5k+ open issues, a serious security
  history; sandboxing off by default (enable `sandbox.mode: non-main`, backend podman).

### Hermes Agent — only with a 64K-served model
- `MINIMUM_CONTEXT_LENGTH = 64_000` enforced at startup and on model switch [confirmed]; PR
  `allow_short_context` unmerged. Serve Gemma 4 with `--max-model-len 65536` (KV ≈ 2.6 GiB) and set
  `model.context_length: 65536`.
- ~15K-token prefix is stable within a session (built once, rebuilt only on compaction) so prefix
  caching absorbs it; but compaction summaries, title generation, "smart" approvals and
  session-search summaries all default to the main model — set `approvals.mode: manual`, give
  `auxiliary.*` a small fast model or accept multi-minute pauses.
- `hermes claw migrate` imports OpenClaw workspaces. Launch vLLM with `--tool-call-parser gemma4`
  (not `hermes`, which is for Hermes-format JSON models). [corrected]

### nanobot — lightest OpenClaw replacement
- Provider block is literally `providers.vllm = {apiBase, apiKey: "EMPTY"}`; presets carry
  `contextWindowTokens`; `fallbackModels` chains shrink context to the smallest window. Text fallback
  parses inline JSON tool calls; dropped calls trigger one corrective retry.
- Watch: prompt rebuilt every turn so memory/skills changes break the prefix (#2463 open); Gemma 4
  E4B via Ollama could not use tools (#2829, vLLM path untested); busy p1 bug queue; 1–2 committers.
- Config and second-instance recipe: see the scan profile (`--home ~/.nanobot-coder gateway --port
  18791`).

### OpenCode v2 — strong coding alternative to pi
- 3–5K fixed overhead, `opencode serve` + `opencode attach` over an SSH tunnel is a nice remote
  workflow. Set `limit.context: 32768`, `compaction.buffer: 4096`, `keep.tokens: 6000`; disable the
  Code Mode `execute` tool (Gemma-4-31B routes native reads through it and burns 10K+ tokens,
  #53623). No gateway, ever (maintainer statement Oct 21). [corrected]
- v2 is a six-week-old rewrite with 20–30 commits/day and v1 local-model issues mass-closed on Oct
  11 — pin a version.

### Goose, Aider, Cline/Kilo, Codex, Claude Code — see matrix
- Goose: small overhead, but `GOOSE_MODE` defaults to auto (set `approve`), the repetition guard is a
  no-op (set `GOOSE_MAX_TURNS=40`), and vLLM's `max_model_len` is not discovered (declare
  `context_limit`). One Telegram bot per config root, no groups. [confirmed/corrected]
- Aider: declare the model in `.aider.model.settings.yml` (`edit_format: diff`, `use_repo_map: true`)
  and `.aider.model.metadata.json` (`max_input_tokens: 32768`) or it silently falls to `whole`
  format with no repo map [corrected]. Immune to tool-parser problems.
- Codex: `wire_api="responses"` only (vLLM supports it, see vLLM's Codex page), but unknown model
  slugs get no native `apply_patch` tool and a contradictory fallback prompt [confirmed]; MCP tools
  are emitted as `namespace` tools with no switch (PRs closed unmerged) [corrected]. 553 open
  `custom-model` issues without maintainer replies.
- Claude Code: `ANTHROPIC_BASE_URL=http://localhost:8080` works because vLLM implements
  `/v1/messages`; default payload ~27–33K tokens; `--bare --tools … --system-prompt-file` gets it
  to ~2K; autocompact floor is min(100K, window) and a context-400 is recovered manually, not fatal
  [corrected]. Sensible only against a Spark-hosted 128K+ model; channels need Anthropic auth.

## 4. Scores (author's, 0–10, from the verified profiles)

| Harness | Small-model robustness | Operational simplicity / longevity | Job fit (coding + Telegram + endpoints + Spark routing) | Mean |
|---|---|---|---|---|
| pi | 7 | 6 | 6 | **6.3** |
| Qwen Code | 6 | 6 | 8 | **6.7** |
| nanobot | 7 | 5 | 8 | **6.7** |
| ZeroClaw | 7 (unverified) | 6 | 7 | 6.7 |
| OpenClaw | 5 | 3 | 8 | 5.3 |
| Hermes | 3 (5 if served at 64K) | 3 | 8 | 4.7–5.3 |
| OpenCode v2 | 6 | 4 | 5 | 5.0 |
| Goose | 5 | 7 | 5 | 5.7 |
| Aider | 8 | 7 (dormant) | 3 | 6.0 |
| Cline CLI / Kilo | 4 | 4 | 6 | 4.7 |
| Claude Code (on Claw) | 2 | 3 | 5 | 3.3 |
| Codex CLI | 2 | 4 | 3 | 3.0 |

Reading: no single harness wins all three jobs; the top scores come from pairing a coding-specific
harness with an assistant-specific one.

## 5. Recommended architecture

| Job | Harness | Model / vLLM flags | Why |
|---|---|---|---|
| Terminal coding over SSH (default) | **pi** | Gemma 4 26B INT4, `--tool-call-parser gemma4`, prefix caching on | lowest overhead, model switching, works with prefix caching off too |
| Terminal coding, Qwen3-Coder sessions | **Qwen Code** | Qwen3-Coder-30B-A3B, `--tool-call-parser qwen3_coder`, prefix caching on | XML fallback + parser-matched prompt; too heavy for GDN+MTP configs |
| Two Telegram bots with memory | **OpenClaw** (now) → bake-off vs **nanobot** | Gemma 4 on the Claw; `fallbacks` to a Spark provider when home | OpenClaw already fits; nanobot is lighter and more tolerant |
| Big-model sessions from the work laptop | **Claude Code** or pi against a Spark provider | 128K+ model on a DGX Spark over Tailscale | Claude Code is excellent where context is not the constraint |
| Emergency / parser-broken models | **Aider** | any; `edit_format: diff` | no tool calls at all |
| Voice / OCR endpoints (later) | MCP servers or OpenAI-compatible tools exposed to pi/OpenClaw/nanobot | OpenVINO services from the multimodal plan | all three speak MCP; nanobot/OpenClaw take a custom STT `api_base` |

One vLLM endpoint serves all of them; the model-per-job mapping is a `--served-model-name` and a
parser flag.

### One-day bake-off (measure, don't guess)
Same five tasks (bug fix in llm-scaler, log-parsing script, Telegram bot feature, a refactor, a
"read the repo and explain" question), same model, thinking off:
1. **Tool-call failure rate**: parse errors, calls emitted as text, empty args — from the harness log
   and vLLM's request log.
2. **TTFT per turn** with prefix caching on and off (`vllm bench` is not needed; wall-clock from the
   harness, plus vLLM `/metrics` prompt-token counts → tokens/turn).
3. **Tasks finished / finished with nudges / failed**, and loop incidents.
4. **Host memory** (`free -g`) with the harness running — Hermes and OpenClaw are the heavy ones.
Expected outcome from the profiles: pi and Qwen Code finish the most tasks on Qwen3-Coder; OpenClaw
and nanobot tie on the Telegram tasks with nanobot cheaper per turn.

## 6. Rejected / not recommended
- **Codex CLI** — Responses-only, no `apply_patch` for third-party models, MCP `namespace` wrapper,
  closed development.
- **Claude Code on the Claw** — 27–33K first request, 100K compaction floor, auth-gated channels.
- **Roo Code** (archived May 2026), **ZooCode** (small anonymous fork), **Cline/Kilo** (native-only
  tools, 4.1.x regressions, Cline CLI auto-approves by default).
- **PicoClaw** — last commit Aug 19, loop/context issues closed "not planned".
- **Crush** (FSL, not OSI), **Continue** (archived), **Letta Code** (local backend experimental,
  UI-only endpoint config), **Mistral Vibe** (docs unreachable), **smolagents** (framework, not a
  harness), **Zed** (editor), **OpenHands** (heavy), **Antigravity IDE** (local model cannot be the brain).

## 7. Open questions (hands-on only)
- pi #10540: does Qwen3.5-class output file contents as text instead of calling `write` on vLLM?
- nanobot: Gemma 4 tool calling via vLLM's `gemma4` parser (only the Ollama failure is on record);
  real prefix-cache hit rate given the per-turn prompt rebuild (#2463).
- ZeroClaw: does the `vllm` family slot honor a custom `uri` (#11296)? Is the memory embedder offline?
- Hermes: status of PR #32770 (`allow_short_context`); measured `hermes prompt-size` for the
  telegram preset.
- Qwen Code: whether PR #6104 brought the 22K "hello" overhead down materially on 0.26.
- Nobody has tested any harness against vLLM-XPU with a GDN-hybrid + MTP model where prefix caching
  is off; per-turn re-prefill cost is the number that decides pi vs everything else there.

## 8. Sources
- pi: https://github.com/earendil-works/pi · docs/models-configuration.md, compaction.md, mcp.md ·
  issues #9566, #10330, #10540, #10575 · npm @earendil-works/pi-coding-agent
- Codex: https://github.com/openai/codex · config.schema.json · PR #10157 · models-manager/src/model_info.rs ·
  issues #26234, #44529, #51077, #33405, #37009 · https://docs.vllm.ai/en/latest/serving/integrations/codex/
- Claude Code: https://github.com/anthropics/claude-code · code.claude.com/docs (model-config, llm-gateway-protocol,
  channels, settings) · issues #73165, #88536, #101897, #78372 · ollama anthropic-compatibility.mdx
- Hermes: https://github.com/NousResearch/hermes-agent · agent/model_metadata.py, agent/system_prompt.py ·
  issues #47967, #87075, #120872, #141641, #141946, #6839 · PR #32770
- OpenClaw: https://github.com/openclaw/openclaw · src/agents/context-window-guard.ts (commit 13e917e) ·
  docs/providers/vllm.md, docs/gateway/local-models.md, docs/reference/token-use.md · issues #14785, #179306,
  #60601, #49508, #117787, #118355, #73687
- OpenCode: https://github.com/anomalyco/opencode · packages/ai/src/protocols/openai-chat.ts · packages/schema/src/mcp.ts ·
  issues #41362, #53623, #22792, #50814, #57458
- Qwen Code: https://github.com/QwenLM/qwen-code · docs/users/configuration/{auth,model-providers}.md ·
  packages/core/src/core/tokenLimits.ts, streamingToolCallParser.ts · issues #6097, #4657, #7148, #13415, #14112
- Goose: https://github.com/aaif-goose/goose · crates/goose/src/prompts/system.md · formats/openai.rs ·
  context_limit.rs · issues #12590, #12150, #12969, #8581, #10784, #10402 · PR #7989, #11213
- Aider: https://github.com/Aider-AI/aider · aider/coders/base_coder.py · aider/models.py · issues #5486, #5111, #3830 ·
  website/_data/polyglot_leaderboard.yml
- Cline/Kilo/Roo: https://github.com/cline/cline (docs/provider-config/openai-compatible.mdx, cli-reference.mdx) ·
  https://github.com/Kilo-Org/kilocode · https://github.com/RooCodeInc/Roo-Code (archived) · issues #13008, #14453, #14827,
  #11076, #12666
- nanobot: https://github.com/HKUDS/nanobot · docs/providers.md, chat-apps.md, memory.md, multiple-instances.md ·
  nanobot/providers/openai_compat_provider.py, agent/runner.py · issues #2463, #2829, #1068, #1590, #5924
- ZeroClaw: https://github.com/zeroclaw-labs/zeroclaw · docs/book (providers/custom.md, channels/telegram.md,
  security/sandboxing.md, agents/history-management.md) · issues #11296, #9177
- PicoClaw: https://github.com/sipeed/picoclaw · Claw-SWE-Bench: arXiv 2606.12344 (via search)
