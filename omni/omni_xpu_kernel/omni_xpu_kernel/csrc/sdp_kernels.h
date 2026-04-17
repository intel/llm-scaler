#pragma once

// DLL import declarations for ESIMD SDP kernels (built with doubleGRF)

#ifdef _WIN32
  #define ESIMD_KERNEL_API __declspec(dllimport)
#else
  #define ESIMD_KERNEL_API
#endif

extern "C" {

ESIMD_KERNEL_API void sdp_fp16(
    void* Q, void* K, void* V, void* normAlpha, void* out,
    int q_len, int kv_len,
    int headQ, int headKv, void* sycl_queue_ptr);

ESIMD_KERNEL_API void sdp_bf16io(
    void* Q, void* K, void* V, void* normAlpha, void* out,
    int q_len, int kv_len,
    int headQ, int headKv, void* sycl_queue_ptr);

ESIMD_KERNEL_API void sdp_fp16_fast(
    void* Q, void* K, void* V, void* normAlpha, void* out,
    int q_len, int kv_len,
    int headQ, int headKv, void* sycl_queue_ptr);

ESIMD_KERNEL_API void sdp_fp16_hd64(
    void* Q, void* K, void* V, void* normAlpha, void* out,
    int q_len, int kv_len,
    int headQ, int headKv, void* sycl_queue_ptr);

ESIMD_KERNEL_API void sdp_bf16io_hd64(
    void* Q, void* K, void* V, void* normAlpha, void* out,
    int q_len, int kv_len,
    int headQ, int headKv, void* sycl_queue_ptr);

} // extern "C"
