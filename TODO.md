# TODO

## Open efficiency bugs

- [ ] **Avoid scanning giant checkpoints twice in `omni/tools/extract.py`.**
  The script currently iterates through `state_dict.items()` once to collect UNet keys and then a second time to collect VAE keys. For large checkpoints this doubles Python-level iteration work and adds avoidable wall-clock time. A single-pass split (`if/elif` dispatch per key) would cut this overhead roughly in half. 

- [ ] **Avoid multi-pass parsing of the same benchmark log in `vllm/tools/platform/evaluation/gen_evaluation_report.py`.**
  `main()` loads all lines and then calls four parsers (`parse_p2p_bandwidth`, `parse_gpu_memory_bandwidth`, `parse_gemm_int8`, `parse_ccl_busbw`), each rescanning much or all of the same input. This repeated full-file traversal is unnecessarily expensive for large logs. Consider one streaming parser (or a shared indexed pre-parse) that extracts all sections in one pass.

- [ ] **Remove repeated regex searches inside tight loops in `parse_p2p_bandwidth()`.**
  Within the inner loop, the code checks section boundaries with `any(re.search(pattern, current) for pattern in TARGETS.values())` on each line, causing repeated regex evaluation against all section patterns. Precompiled regex objects and explicit section markers (or a deterministic state machine) would reduce per-line overhead.

- [ ] **Reduce repeated timestamp/path computation and file open churn in `vllm/test/run_scripts/process_data.py`.**
  `process_file()` computes `datetime.now().strftime(...)`, directory paths, and opens output CSV files separately for each raw input file. When many `.out` files are processed, this causes repeated open/close cycles and path recomputation that could be batched by date/output target for faster bulk processing.
