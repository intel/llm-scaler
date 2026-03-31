// ============================================================================
// SDP Kernel Configuration — compile-time hardware-specific parameters
// ============================================================================
// Template parameterization enables zero-overhead hardware adaptation.
// Add new configs for different GPUs without modifying kernel code.
// ============================================================================
#pragma once

#include <cstdint>

namespace sdp_config {

// Base configuration — all derived configs inherit defaults from here
struct SdpConfigBase {
    // Thread organization
    static constexpr int WG_SIZE      = 16;   // Threads per workgroup (must be power of 2)
    static constexpr int Q_PER_THREAD = 16;   // Q rows processed per thread
    static constexpr int Q_GROUP      = WG_SIZE * Q_PER_THREAD;  // Q rows per workgroup group (256)

    // Tiling
    static constexpr int HEAD_DIM     = 128;  // Head dimension (D)
    static constexpr int KV_TILE      = 64;   // K/V rows processed per loop iteration

    // SLM (Shared Local Memory)
    // V pingpong buffer: 2 * KV_TILE * HEAD_DIM * sizeof(fp16)
    static constexpr int SLM_V_BYTES  = 2 * KV_TILE * HEAD_DIM * 2;  // 32768 bytes (32KB)
    static constexpr int SLM_TOTAL    = SLM_V_BYTES;

    // DPAS configuration
    static constexpr int DPAS_M       = 8;    // DPAS tile M
    static constexpr int DPAS_N       = 8;    // DPAS tile N

    // Prefetch
    static constexpr int PREFETCH_K_BLOCKS = 2;  // Number of K prefetch blocks

    // Derived constants
    static constexpr int V_LOAD_THREADS = 4;  // Threads cooperating on V load (hhv mask = 0x3)
    static constexpr int V_ROWS_PER_THREAD = KV_TILE / V_LOAD_THREADS;  // 16

    // QK iteration structure
    static constexpr int QK_K_BLOCKS  = HEAD_DIM / 16;  // K blocks in QK (8 for HEAD_DIM=128)
    static constexpr int QK_N_TILES   = KV_TILE / 8;    // N tiles per K block (8 for KV_TILE=64)

    // S×V iteration structure
    static constexpr int SV_BLOCKS    = 4;    // S×V blocks (2 halves × 2 V row groups)
    static constexpr int SV_NN        = 2;    // V row sub-blocks per block
    static constexpr int SV_DPAS_PER_NN = 8;  // DPAS per sub-block
};

// ============================================================================
// Hardware-specific configurations
// ============================================================================

// Intel Arc B580 (Battlemage / BMG / Xe2-HPG)
// 160 XVEs, 20 Xe Cores, doubleGRF, 12GB GDDR6
struct ConfigBMG : SdpConfigBase {
    // BMG defaults are same as base — this is what we've tuned for
};

// Intel Arc B770 (Battlemage / BMG-G31)
// More XVEs, potentially different optimal WG_SIZE
struct ConfigB770 : SdpConfigBase {
    // Same architecture, same config for now
    // Future: may benefit from larger WG_SIZE if more EU per subslice
};

// Intel Data Center GPU Max (Ponte Vecchio / PVC / Xe-HPC)
// 128 Xe Cores per tile, HBM2e, larger SLM per subslice
struct ConfigPVC : SdpConfigBase {
    // PVC has more SLM capacity and HBM bandwidth
    // Future: could increase KV_TILE to 128 if register pressure allows
    // static constexpr int KV_TILE = 128;
    // static constexpr int SLM_V_BYTES = 2 * 128 * HEAD_DIM * 2;  // 64KB
};

// Intel Lunar Lake (LNL / Xe2-LPG) — integrated GPU
struct ConfigLNL : SdpConfigBase {
    // Lower EU count, same ISA as BMG
    // May benefit from smaller WG_SIZE for better occupancy
    // static constexpr int WG_SIZE = 8;
    // static constexpr int Q_PER_THREAD = 16;
    // static constexpr int Q_GROUP = 128;
};

// HEAD_DIM=64 variants — for SD3.5, z-Image, LTX-Video
struct ConfigBMG_HD64 : SdpConfigBase {
    static constexpr int HEAD_DIM    = 64;
    static constexpr int SLM_V_BYTES = 2 * KV_TILE * HEAD_DIM * 2;  // 16384 bytes (16KB)
    static constexpr int SLM_TOTAL   = SLM_V_BYTES;
    static constexpr int QK_K_BLOCKS = HEAD_DIM / 16;  // 4
    static constexpr int SV_DPAS_PER_NN = HEAD_DIM / 16;  // 4
};

// ============================================================================
// Active configuration — change this to switch hardware target
// ============================================================================
#if defined(SDP_CONFIG_PVC)
using ActiveConfig = ConfigPVC;
#elif defined(SDP_CONFIG_LNL)
using ActiveConfig = ConfigLNL;
#elif defined(SDP_CONFIG_B770)
using ActiveConfig = ConfigB770;
#else
// Default: BMG (Arc B580)
using ActiveConfig = ConfigBMG;
#endif

}  // namespace sdp_config
