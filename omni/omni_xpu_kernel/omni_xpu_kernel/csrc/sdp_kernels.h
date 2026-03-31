#pragma once

// DLL import declarations for ESIMD SDP kernels (built with doubleGRF)
// The DLL (esimd.unify.lgrf.dll) is loaded at runtime alongside this PYD.

#ifdef _WIN32
  #define ESIMD_KERNEL_API __declspec(dllimport)
#else
  #define ESIMD_KERNEL_API
#endif

extern "C" {

// FP16 optimized Flash Attention (barrier+interleave optimization)
// Q/K/V/out: [L, H, 128] fp16 device pointers
// normAlpha: [H*128] float32
ESIMD_KERNEL_API void sdp_fp16(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr);

// BF16 I/O hybrid Flash Attention (bf16 QK DPAS + fp16 SxV DPAS)
// Q/K/V/out: [L, H, 128] bf16 device pointers
// normAlpha: [H*128] float32
ESIMD_KERNEL_API void sdp_bf16io(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr);

// FP16 Flash Attention fast variant (no compensation clamp, for small V values)
ESIMD_KERNEL_API void sdp_fp16_fast(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr);

// FP16 Flash Attention for head_dim=64 (SD3.5, z-Image, LTX-Video)
// Q/K/V/out: [L, H, 64] fp16 device pointers
// normAlpha: [H*64] float32
ESIMD_KERNEL_API void sdp_fp16_hd64(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr);

// BF16 I/O hybrid Flash Attention for head_dim=64
// Q/K/V/out: [L, H, 64] bf16 device pointers
// normAlpha: [H*64] float32
ESIMD_KERNEL_API void sdp_bf16io_hd64(
    void* Q, void* K, void* V,
    void* normAlpha,
    void* out,
    int q_len, int kv_len,
    int headQ, int headKv,
    void* sycl_queue_ptr);

} // extern "C"
