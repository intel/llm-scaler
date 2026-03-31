# TODO

## Open efficiency bugs

- [ ] **`vllm/test/run_scripts/gen_run_script.py`: benchmark scripts disable key runtime optimizations by default.**
  - `run_model()` always adds `--disable-sliding-window`, and may add `--no-enable-prefix-caching` depending on config.
  - This can significantly reduce token throughput and increase latency for long-context and repeated-prefix workloads, making generated benchmark runs systematically less efficient than necessary.
  - **Suggested fix:** make both switches opt-in (or auto-tuned by model/workload) instead of default-on in generated commands.

- [ ] **`omni/benchmarks/benchmark_t2v_comfyui.py`: avoid reconnecting WebSocket for every request.**
  - `wait_for_completion()` creates a new `websocket.WebSocket()` connection for each warm-up/trial request.
  - Repeated WS handshakes add avoidable per-request overhead and can skew latency measurements for shorter workflows.
  - **Suggested fix:** keep a persistent connection for the full benchmark run and multiplex prompt tracking by `prompt_id`.

- [ ] **`omni/benchmarks/benchmark_t2v_comfyui.py`: deep-copying full workflow JSON each iteration is unnecessary overhead.**
  - Each loop does `copy.deepcopy(workflow_json_template)` even though only the sampler seed is updated.
  - For large workflows and higher trial counts, this introduces avoidable CPU and memory churn.
  - **Suggested fix:** prebuild immutable workflow sections and clone only the mutable node input (or mutate/reset a single field in-place).

- [ ] **`vllm/test/run_scripts/process_data.py`: duplicate CSV write path doubles I/O per parsed result set.**
  - The script writes all rows to both `result_unrounded.csv` and `result.csv` in separate passes.
  - For large benchmark batches, this doubles file open/write activity and slows post-processing.
  - **Suggested fix:** write a single canonical CSV by default and gate the secondary artifact behind an explicit flag.
