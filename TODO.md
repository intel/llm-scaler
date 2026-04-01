# TODO

## Efficiency bugs to fix

- [ ] **Unbounded model readiness wait can stall the whole benchmark pipeline.**
  In `check_ready`, the generated shell loop polls `/health` forever with no timeout or backoff strategy. If model startup fails, the run never progresses and keeps burning CPU on repeated polling. Add a max wait time and fail-fast path (for example, `timeout` + structured error output). (`vllm/test/run_scripts/gen_run_script.py`)

- [ ] **Large log files are fully loaded into memory before parsing.**
  `gen_evaluation_report.py` uses `f.readlines()` and then parses the full list. On multi-GB logs this increases peak RAM usage and can trigger swapping, which significantly slows report generation. Parse the file as a stream (line-by-line iterator) instead. (`vllm/tools/platform/evaluation/gen_evaluation_report.py`)

- [ ] **Benchmark workflow relaunches models per benchmark call, adding avoidable startup overhead.**
  `benchmark_t2i_model` defaults to `relaunch=True`, and the script entrypoint calls it three times sequentially. Each run pays model launch/terminate cost even when the same service session could be reused for multiple trial groups, reducing throughput of comparative benchmarking. Add a reuse mode at the script level (launch once, benchmark multiple targets, terminate once per target only when required). (`omni/benchmarks/benchmark_t2i_xinference_openai_api.py`)

- [ ] **Result extraction does repeated regex scanning for each metric line.**
  `parse_file_results` checks every line against many `startswith` branches and repeatedly calls regex extraction helpers. For large benchmark outputs, this introduces avoidable parsing overhead. Replace with a dispatch map keyed by prefix and extract numeric tokens once per matching line. (`vllm/test/run_scripts/process_data.py`)
