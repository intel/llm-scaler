# TODO

## Open efficiency bugs

- [ ] **Container lifecycle is repeated for every model benchmark run** (`vllm/test/run_scripts/gen_run_script.py`).
  - **Why this hurts efficiency:** `gen_run_scripts()` calls `create_container(...)` inside the model loop and then immediately `stop_container(...)` + `rm_container(...)` for each model. This repeatedly pays container startup/teardown costs and can trigger repeated initialization overhead for mounted environments.
  - **Suggested fix:** Create/start one container per config run, benchmark all models in that same container, then stop/remove once at the end.

- [ ] **Prefix caching is forcibly disabled in server launch flags** (`vllm/test/run_scripts/gen_run_script.py`).
  - **Why this hurts efficiency:** `run_model()` adds `--no-enable-prefix-caching`, which prevents prompt-prefix reuse and can reduce token throughput for repeated/shared-prefix workloads.
  - **Suggested fix:** Make prefix caching configurable from YAML and default to enabled for throughput-focused benchmarks.

- [ ] **Eager execution is forced for all runs** (`vllm/test/run_scripts/gen_run_script.py`).
  - **Why this hurts efficiency:** `run_model()` unconditionally passes `--enforce-eager`, which can block graph-level/runtime optimizations and lower steady-state serving throughput.
  - **Suggested fix:** Gate `--enforce-eager` behind a debug/compatibility switch and default to optimized execution mode.

- [ ] **Data processing launches one Python process per file via `find | xargs`** (`vllm/test/run_scripts/gen_run_script.py`).
  - **Why this hurts efficiency:** `process_data()` and `post_process_data()` spawn many short-lived Python processes for large result sets, adding measurable process startup overhead.
  - **Suggested fix:** Add batch-mode support to `process_data.py` / `post_process_data.py` to process multiple files in a single interpreter invocation.

- [ ] **Numeric parsing allocates a full list when only one value is needed** (`vllm/test/run_scripts/process_data.py`).
  - **Why this hurts efficiency:** `get_num()` builds `candidates = []` from all regex matches, even though callers always request index `0`; this adds avoidable allocations for every parsed metric line.
  - **Suggested fix:** Parse lazily and return on the first required match, or use `NUMBER_PATTERN.search(...)` for the default index path.
