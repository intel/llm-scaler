# Lunar Lake Xe2 140V Compatibility Report

**Status: NOT COMPATIBLE**

This project targets Intel Arc Pro B60 discrete GPUs and is not compatible with the Lunar Lake Xe2 140V integrated GPU.

## Incompatibility Summary

| Aspect | llm-scaler Requires | Lunar Lake Xe2 140V |
|--------|---------------------|---------------------|
| **GPU Type** | Arc Pro B60 (discrete) | Arc 140V (integrated) |
| **GPU Memory** | 20-50GB dedicated VRAM | ~24GB shared from system RAM |
| **CPU** | Intel Xeon (Workstation/Server) | Core Ultra 7 258V (mobile SoC) |
| **OS** | Ubuntu 25.04 | General Linux |
| **Compute API** | SYCL / oneAPI (DPC++ compiler) | Vulkan |
| **Runtime** | Level-Zero + oneAPI 2025.2+ | xe kernel driver |
| **Custom Kernels** | SYCL ESIMD (`icpx` compiled) | Not supported |
| **PCIe** | Discrete PCIe x16/x8 slot | On-package (no PCIe) |

## Key Blockers

1. **SYCL ESIMD kernels** — The custom GPU kernels (dequantization, normalization, RoPE) are compiled with Intel's `icpx` compiler targeting discrete Arc Pro hardware. These kernels would not run on the Arc 140V iGPU.

2. **Level-Zero runtime** — The project depends on the Level-Zero GPU abstraction layer configured for discrete GPUs. Lunar Lake's integrated GPU uses a different driver path (xe + Vulkan).

3. **Dedicated VRAM requirement** — Models require 20-50GB of dedicated GPU VRAM. The Arc 140V shares system RAM (~24GB usable), with significantly different memory access patterns.

4. **vLLM XPU patches** — The multi-arc vLLM patches assume discrete GPU topology, PCIe P2P communication, and multi-GPU scaling that don't apply to an integrated GPU.

5. **Xeon CPU dependency** — The platform tools and offline installer target Xeon workstation/server platforms, not mobile Core Ultra SoCs.

## Alternative for Lunar Lake

For local LLM inference on Lunar Lake Xe2 140V, use **llama.cpp with Vulkan acceleration** instead. See the [OpenClaw-on-MSI-Claw-8](https://github.com/MegaStood/OpenClaw-on-MSI-Claw-8) project for a complete Lunar Lake-optimized setup.

---

*Report generated: 2026-03-20*
