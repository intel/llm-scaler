# TODO: Bugs affecting Qwen3.5 loading/operation

## 1) `extra_param` key validation regex is too permissive (can admit invalid CLI flags)
- **Where**: `vllm/test/run_scripts/gen_run_script.py` (`SAFE_EXTRA_PARAM_KEY`)
- **Issue**: the character class `"[a-zA-Z0-9-_]"` uses `-` in the middle, which creates a range (`9` to `_`) and unintentionally allows extra punctuation.
- **Why this can break Qwen3.5 runs**: malformed `extra_param` keys can slip through validation, then get emitted into the launch command and cause vLLM argument parsing to fail before model load.
- **Suggested fix**: change to `r"^--[a-zA-Z0-9][a-zA-Z0-9_-]*$"` (move `-` to the end or escape it).

## 2) `extra_param` rendering assumes every flag is `--flag=value`
- **Where**: `vllm/test/run_scripts/gen_run_script.py` (`run_model`)
- **Issue**: all extra parameters are rendered as `flag=value`.
- **Why this can break Qwen3.5 runs**: many vLLM/feature flags are boolean toggles expected as standalone flags (for example, flags used when tuning Qwen reasoning behavior). Rendering them as `--flag=true` or `--flag=value` can be rejected by argparse depending on option type.
- **Suggested fix**: support both forms:
  - bare flags (e.g., `{"--enable-feature": null}` => `--enable-feature`)
  - key/value flags (e.g., `{"--max-model-len": 32768}` => `--max-model-len=32768`)

## 3) Hugging Face cache snapshot selection is nondeterministic/fallible
- **Where**: `vllm/test/run_scripts/script_config.py` (`_resolve_hf_snapshot_candidate`, `_resolve_latest_snapshot`)
- **Issue**:
  - prefers `refs/main` only;
  - if absent, picks snapshot by latest mtime.
- **Why this can break Qwen3.5 runs**: hosts with multiple cached revisions can resolve to an unintended snapshot (or partially downloaded/latest-touched directory), causing config/weight mismatches and startup failure.
- **Suggested fix**:
  - support explicit revision pinning in config;
  - prefer a deterministic snapshot selection rule (e.g., hash in config, or validated `refs/*` fallback order);
  - optionally verify required files (`config.json`, tokenizer files, safetensors index) before accepting snapshot.

## 4) Qwen3.5 support is advertised at repo root but not listed in `vllm/README.md` model matrix
- **Where**: `README.md` vs `vllm/README.md`
- **Issue**: root README lists Qwen3.5 entries; `vllm/README.md` supported-model table currently omits Qwen3.5 rows.
- **Why this can break operation in practice**: users often follow `vllm/README.md` for launch parameters and compatibility checks; this mismatch increases the chance of selecting wrong image/tag/flags for Qwen3.5 and hitting avoidable startup issues.
- **Suggested fix**: synchronize both support matrices and include at least one explicit Qwen3.5 launch example in `vllm/README.md`.
