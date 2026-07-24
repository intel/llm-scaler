# Benchmarks

This directory contains diagnostic performance programs for
`omni_xpu_kernel`. They are intentionally separate from `tests/`: pytest
correctness and packaging checks must not collect or execute performance
workloads.

Run the grouped microbenchmarks from the package root:

```bash
python -m benchmarks.run_all
python -m benchmarks.run_all --fp8
python -m benchmarks.run_all --gguf
python -m benchmarks.run_all --norm
python -m benchmarks.run_all --sdp
```

The remaining programs cover larger or workload-specific shapes and are run
individually:

```bash
python benchmarks/bench_int8.py --device xpu --shapes comfyui
python benchmarks/bench_ltx2.py
python benchmarks/bench_qwen_hd128.py
python benchmarks/bench_sdp_hd64_bf16.py
```

Run benchmarks only with a wheel built for the local Torch minor and XPU
target. Results are sensitive to warm-up, driver, power state, competing
processes, tensor shapes, and compiler/runtime versions; preserve these inputs
with any reported measurement.
