# TODO: Deep Bug Hunt Findings

Date: 2026-03-31

## 1) `ModelPath` parsing crashes if config uses a YAML list
- **Location:** `vllm/test/run_scripts/script_config.py`
- **Bug:** `build_sub_obj()` assumes `Path.ModelPath` is a semicolon-delimited string and immediately calls `.split(';')`.
- **Why this is a bug:** In YAML, users commonly represent paths as a list (`ModelPath: ["/a", "/b"]`). Passing a list triggers `AttributeError: 'list' object has no attribute 'split'` before validation logic can run.
- **Impact:** Config loading fails at startup for valid YAML styles.
- **Suggested fix:** Accept both `str` and `list[str]`:
  - If value is `str`, split on `;`.
  - If value is `list`, use it directly (after trimming/validation).
  - Otherwise raise a clear `TypeError`.

## 2) `batch` parsing crashes when user supplies a YAML list
- **Location:** `vllm/test/run_scripts/script_config.py`
- **Bug:** `batch=[int(x) for x in model.get("batch").split(',')]` assumes `batch` is always a comma-separated string.
- **Why this is a bug:** YAML users often write `batch: [1, 2, 4]`. That value has no `.split`, causing a runtime failure.
- **Impact:** Benchmark script generation fails for common YAML syntax.
- **Suggested fix:** Support both forms:
  - string CSV (`"1,2,4"`), or
  - list of ints (`[1,2,4]`).
  Normalize to `List[int]` with explicit validation and better error messages.

## 3) Boolean CLI parsing for `--add_config_header` is fragile and easy to misconfigure
- **Location:** `vllm/test/run_scripts/process_data.py`
- **Bug:** `argparse` uses `type=lambda x: x.lower() == 'true'`.
- **Why this is a bug:** Values like `1`, `yes`, `TRUE `, or accidental typos silently become `False` instead of causing an argument error.
- **Impact:** CSV headers/config columns can be silently omitted, producing malformed/ambiguous output and incorrect downstream sorting.
- **Suggested fix:** Use `argparse.BooleanOptionalAction` (e.g., `--add-config-header/--no-add-config-header`) or strict choices (`choices=['true','false']`) with normalization.

## 4) `post_process_data.py` can crash on metric-only CSVs (or sort by unintended columns)
- **Location:** `vllm/test/run_scripts/post_process_data.py`
- **Bug:** The script always sorts by the first five columns (`df.columns[:5]`).
- **Why this is a bug:** For CSV files generated without config headers, the first five columns are metrics, not metadata. On malformed or smaller CSVs this can also produce errors or misleading ordering.
- **Impact:** Incorrect report ordering and potential runtime failures depending on input shape.
- **Suggested fix:**
  - Prefer explicit sort keys when config columns exist (`Date`, `Version`, `Model`, `Tag`, `Batch Size`).
  - Fallback safely to no-sort or a minimal stable key when columns are missing.

## 5) Trust-remote-code flag inconsistency likely breaks one of the generated commands
- **Location:** `vllm/test/run_scripts/gen_run_script.py`
- **Bug:** Generated model-start command uses `--trust-remote-code`, while benchmark command uses `--trust_remote_code`.
- **Why this is a bug:** CLI flags are typically defined with one canonical spelling. One of these forms is likely invalid depending on parser definitions.
- **Impact:** Either server launch or benchmark invocation can fail at runtime with unrecognized argument errors.
- **Suggested fix:** Standardize to the exact flag spelling accepted by the target vLLM command, and add a smoke test that executes generated commands with `--help` validation.

## 6) `extract.py` executes heavy file I/O at import time
- **Location:** `omni/tools/extract.py`
- **Bug:** Script logic (checkpoint loading/saving) runs at module import scope instead of inside `main()`.
- **Why this is a bug:** Any import (tests, REPL, tooling) triggers full extraction side effects unexpectedly.
- **Impact:** Unintended filesystem writes, long import latency, and hard-to-test behavior.
- **Suggested fix:** Move execution into `main()` and guard with `if __name__ == '__main__':`.

## 7) Gradio web UI leaks temporary video files/directories
- **Location:** `vllm/webui/multi-modal-gradio/main.py`
- **Bug:** `VIDEO_TEMP_DIR` is created via `tempfile.mkdtemp(...)` and uploaded videos are copied there, but never cleaned up.
- **Why this is a bug:** Long-lived services accumulate orphaned files and eventually consume disk.
- **Impact:** Progressive disk-space exhaustion in production/testing environments.
- **Suggested fix:**
  - Register `atexit` cleanup for `VIDEO_TEMP_DIR`.
  - Optionally add periodic pruning and max-size retention policy.

## 8) Gradio server defaults to `share=True` in `launch()`
- **Location:** `vllm/webui/multi-modal-gradio/main.py`
- **Bug:** Public sharing is enabled by default.
- **Why this is a bug/risk:** Accidental internet exposure can leak prompts, uploaded files, and model outputs.
- **Impact:** Security/privacy risk in environments where users expect local-only access.
- **Suggested fix:** Default `share=False` and gate public sharing behind an explicit CLI flag.
