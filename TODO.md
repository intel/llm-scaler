# TODO: Bugs that can interfere with Qwen3.5 model loading/operation

## 1) `vllm/test/run_scripts` defaults to an older container image tag that predates Qwen3.5 support
- **Where:** `vllm/test/run_scripts/script_config.py` (`default_config["VERSION"] = "0.10.2-b6"`).
- **Why this is a bug for Qwen3.5:** the repository release notes/documentation state Qwen3.5 support was added in the newer `0.14.0-b8.1` image line. Keeping the test harness default on `0.10.2-b6` can make Qwen3.5 model tests fail before inference starts (missing model architecture support in container/runtime stack).
- **Impact:** false negatives in CI/manual validation; users may conclude Qwen3.5 is broken when they are using a pre-support image tag.
- **Suggested fix:** bump the script default version to the current Qwen3.5-capable tag (or require explicit version selection and fail fast if tag < minimum supported version).

## 2) `check_and_get_model_path` does not resolve HuggingFace cache snapshot directories
- **Where:** `vllm/test/run_scripts/script_config.py` (`ScriptConfig._build_model_candidates` + `check_and_get_model_path`).
- **Why this can break Qwen3.5 loading:** when models are downloaded via HF cache layout, model contents are typically under `models--ORG--NAME/snapshots/<hash>/...`. The current candidate resolver stops at top-level folder candidates and never resolves into `snapshots/<hash>`, so path resolution can fail even when Qwen3.5 weights are present on disk.
- **Impact:** model-not-found errors for valid Qwen3.5 local installs that use standard HF cache structure.
- **Suggested fix:** add snapshot-aware candidate resolution (e.g., detect `models--*` folder then pick active snapshot via `refs/main` or newest snapshot directory).

## 3) Root model support table uses Qwen3.5 IDs that are inconsistent with the repo’s own loading examples
- **Where:** root `README.md` support table lists `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3.5-122B-A10B`, while Qwen3.5 loading examples in the vLLM patch/docs commonly use `*-Instruct` IDs.
- **Why this can interfere with operation:** users frequently copy model IDs from the support table directly into `--model`. If the listed ID does not match the actual deployed repo IDs in the environment, startup fails at model fetch/config stage.
- **Impact:** avoidable boot-time failures due to incorrect model ID selection.
- **Suggested fix:** normalize table entries to exact tested repo IDs (or include both base and instruct variants with explicit compatibility notes).

