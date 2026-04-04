# TODO

## Efficiency bugs to fix

All listed efficiency bugs have been addressed and removed.

---

# Known Serious Bugs

This section tracks serious bugs identified in the llm-scaler codebase during code review.
Each entry includes the file location, a description of the problem, and its potential impact.

---

## Bug 1 — `assert()` Disabled in Release Builds (norm.cpp)

**Severity:** HIGH  
**File:** `omni/omni_xpu_kernel/omni_xpu_kernel/csrc/norm.cpp:35-36`

```cpp
assert(hidden_size % BS == 0);
assert(hidden_size <= 8 * 1024);
```

**Problem:** Standard `assert()` is compiled out when `NDEBUG` is defined (i.e., in all release/optimized builds). These two precondition checks — that `hidden_size` is a multiple of the block size and does not exceed 8192 — are silently skipped in production. If a caller passes an incompatible `hidden_size`, the ESIMD kernel will access memory out of bounds or produce garbage results with no error raised.

**Fix:** Replace both `assert()` calls with `TORCH_CHECK()`, which is always active regardless of build type.

---

## Bug 2 — Race Condition on Static `zp` Tensor (onednn_int4_gemm.cpp)

**Severity:** HIGH  
**File:** `omni/omni_xpu_kernel/omni_xpu_kernel/csrc/onednn_int4_gemm.cpp:246-248, 307-309`

```cpp
static torch::Tensor zp;
if (!zp.defined() || zp.device() != act.device()) {
    zp = torch::tensor({8}, ...);
}
```

**Problem:** The `static torch::Tensor zp` is initialized inside an `if` block that runs after `g_cache_mutex` has already been released. In a multi-threaded inference server, multiple threads can simultaneously pass the `!zp.defined()` check and each create and assign a new tensor, causing a data race on a static variable. This pattern appears identically in both `onednn_int4_gemm` (line 246) and `onednn_int4_gemm_add_to_output` (line 307).

**Fix:** Protect the `zp` initialization with its own mutex, or use `std::call_once` / a per-device atomic flag.

---

## Bug 3 — Unbounded Global Cache Grows Forever (onednn_int4_gemm.cpp)

**Severity:** HIGH  
**File:** `omni/omni_xpu_kernel/omni_xpu_kernel/csrc/onednn_int4_gemm.cpp:46-47`

```cpp
static std::map<CacheKey, CachedPrimitive> g_cache;
static std::map<CacheKey, CachedPrimitive> g_cache_sum;
```

**Problem:** `g_cache` and `g_cache_sum` accumulate one `CachedPrimitive` entry per unique `{dtype, M, K, N, group_size}` tuple and are never evicted. Each `CachedPrimitive` holds a live `dnnl::engine`, `dnnl::stream`, and `dnnl::matmul` primitive. In a serving environment with variable batch sizes or sequence lengths, the number of distinct keys grows unboundedly, steadily consuming GPU and host memory until the process is OOM-killed.

**Fix:** Add a maximum cache size (e.g., 256 entries) with LRU eviction, or at minimum document the assumption that the key space is bounded.

---

## Bug 4 — Integer Overflow in Scales Array Index (svdq_dequant.cpp)

**Severity:** HIGH  
**File:** `omni/omni_xpu_kernel/omni_xpu_kernel/csrc/svdq_dequant.cpp:88`

```cpp
const OT scale_val = scales[grp * N + row];
```

**Problem:** `grp` and `N` are both `int64_t`, but the multiplication `grp * N` is performed in signed 64-bit arithmetic with no overflow check. For sufficiently large weight matrices (e.g., `num_groups = 1,000,000` and `N = 10,000`), the product exceeds `INT64_MAX`, wraps around to a negative value, and indexes into an arbitrary memory location. This causes silent data corruption or a segfault rather than a bounds-check error.

**Fix:** Add a `TORCH_CHECK` before the kernel launch asserting `num_groups * N <= INT64_MAX`, or use `__int128` for the intermediate calculation.

---

## Bug 5 — Test Calls Unexported API Function (test_gguf_correctness.py)

**Severity:** HIGH  
**File:** `omni/omni_xpu_kernel/tests/test_gguf_correctness.py:120`

```python
output = gguf.dequantize_q4_0_comfyui(q4_0_data, torch.float16)
```

**Problem:** `dequantize_q4_0_comfyui` is not exported from `omni_xpu_kernel/gguf/__init__.py`; only `dequantize_q4_0` is. This causes an `AttributeError` at runtime for `test_dequantize_q4_0_comfyui_correctness`. The bug is even acknowledged in a `TODO` comment directly above the call (lines 116–119), yet the broken call remains, meaning the ComfyUI dequantization path has never been tested via the test suite.

**Fix:** Either export `dequantize_q4_0_comfyui` from `gguf/__init__.py`, or update the test to use the correct function name.

---

## Bug 6 — Test File Imports Non-Existent Module Attribute (test_gen_evaluation_report.py)

**Severity:** HIGH  
**File:** `vllm/tools/platform/evaluation/test_gen_evaluation_report.py:9`

```python
parse_p2p_bandwidth = MODULE.parse_p2p_bandwidth
```

**Problem:** The function `parse_p2p_bandwidth` does not exist in `gen_evaluation_report.py`. The module exposes `parse_all_benchmarks` as its top-level parser. This line raises an `AttributeError` at import time, which means **every test in this file fails before any test body runs**. The entire evaluation test suite is silently broken.

**Fix:** Replace `MODULE.parse_p2p_bandwidth` with the correct function name, or add `parse_p2p_bandwidth` as a dedicated function to `gen_evaluation_report.py` if it is intended to be a separate entry point.

---

## Bug 7 — Temp Frame Files Leak on Exception During Video Processing (main.py)

**Severity:** HIGH  
**File:** `vllm/webui/multi-modal-gradio/main.py:195-200`

```python
frame_paths = extract_frames_from_video(str(new_video_path), num_frames=10)
if frame_paths:
    for frame_path in frame_paths:
        base64_data = encode_file_to_base64(frame_path)
        api_user_content.append(...)
        os.unlink(frame_path)  # only reached if no exception above
```

**Problem:** Temporary JPEG frames are extracted to disk and then deleted inside the same `for` loop, one at a time. If `encode_file_to_base64` or any `dict` operation raises an exception for frame N, the loop aborts and frames N+1 through the end are never deleted. Because the WebUI is a long-running process, failed requests continuously accumulate orphaned frame files, eventually exhausting available disk space.

**Fix:** Collect all `frame_paths` before processing and wrap the loop in a `try/finally` block that unconditionally deletes all extracted frames regardless of whether processing succeeded.

---

## Bug 8 — Silent "Successfully uploaded" After Potentially Failed File Copy (main.py)

**Severity:** HIGH  
**File:** `vllm/webui/multi-modal-gradio/main.py:181-182`

```python
shutil.copyfile(filename, new_video_path)
print("Successfully uploaded")
```

**Problem:** `shutil.copyfile` can raise `OSError` (e.g., disk full, permission denied, source file disappeared). There is no try/except around this call. If the copy fails, the exception propagates up, but the "Successfully uploaded" message has already been printed, misleading users and operators. Worse, subsequent code at line 184 (`open(new_video_path, "rb")`) will then raise `FileNotFoundError` because the destination file was never created, producing a confusing secondary error.

**Fix:** Wrap the `shutil.copyfile` call in a try/except, move the success print to after the copy is confirmed to have succeeded, and surface a clear error message to the user on failure.

---

## Bug 9 — Unreachable `None` Guard After `os.getenv` With Default (download_hy3d_pbr_model.py)

**Severity:** MEDIUM  
**File:** `omni/tools/download_hy3d_pbr_model.py:10-16`

```python
base_download_dir = os.getenv(env_var_name, "/llm/models/Hunyuan3D-2.1")

if base_download_dir is None:
    raise ValueError(...)
```

**Problem:** `os.getenv(key, default)` with an explicit default **never returns `None`** — when the environment variable is unset, it returns the default string. The `if base_download_dir is None` branch is therefore dead code and will never execute. The developer's intent was to require the environment variable to be set and raise a clear error if it was missing; instead, the function silently proceeds with the hardcoded fallback path `/llm/models/Hunyuan3D-2.1`, which may not exist on the target machine, causing a cryptic failure later during model loading.

**Fix:** Either remove the default from `os.getenv` (i.e., use `os.getenv(env_var_name)`) so that `None` is actually returned when the variable is absent, or remove the dead `None` check and document the fallback path intentionally.

---

## Bug 10 — `KeyError` on Missing `"data"` Key in WebSocket Message (benchmark_t2v_comfyui.py)

**Severity:** MEDIUM  
**File:** `omni/benchmarks/benchmark_t2v_comfyui.py` (WebSocket message handler)

```python
if message_obj.get("type") == "executing":
    data = message_obj["data"]  # raises KeyError if "data" is absent
    if data.get("prompt_id") == prompt_id and data.get("node") is None:
```

**Problem:** The code safely retrieves `"type"` using `.get()`, but immediately accesses `message_obj["data"]` with a direct key lookup. If a ComfyUI server sends an `"executing"` message without a `"data"` field (e.g., due to a protocol version difference or partial message), a `KeyError` is raised. This exception is caught by the outer bare `except Exception` handler, which returns `False` and causes the entire benchmark run to be reported as a failure — with only a generic error message and no indication that the root cause was a malformed message.

**Fix:** Replace `message_obj["data"]` with `message_obj.get("data", {})` and add a log line when the key is missing so that protocol mismatches are diagnosable.
