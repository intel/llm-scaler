// ============================================================================
// SDP Kernel Configuration — compile-time hardware-specific parameters
// ============================================================================
// Template parameterization enables zero-overhead hardware adaptation.
// Add new configs for different GPUs without modifying kernel code.
// ============================================================================
#pragma once

#include <cstdint>

namespace sdp_config {

struct BMGTag {};
struct PTLHTag {};

// Primary tuning parameters are template arguments so all resource and launch
// constants are recalculated when a platform changes WG or tile sizes.
template <
    typename ArchTag,
    int HeadDim,
    int WgSize,
    int QPerThread,
    int KvTile,
    int PrefetchKBlocks,
    int VLoadThreads>
struct SdpConfig {
    // Thread organization
    static constexpr int WG_SIZE      = WgSize;
    static constexpr int Q_PER_THREAD = QPerThread;
    static constexpr int Q_GROUP      = WG_SIZE * Q_PER_THREAD;

    // Tiling
    static constexpr int HEAD_DIM     = HeadDim;
    static constexpr int KV_TILE      = KvTile;

    // SLM (Shared Local Memory)
    // V pingpong buffer: 2 * KV_TILE * HEAD_DIM * sizeof(fp16)
    static constexpr int SLM_V_BYTES  = 2 * KV_TILE * HEAD_DIM * 2;  // 32768 bytes (32KB)
    static constexpr int SLM_TOTAL    = SLM_V_BYTES;

    // DPAS configuration
    static constexpr int DPAS_M       = 8;    // DPAS tile M
    static constexpr int DPAS_N       = 8;    // DPAS tile N

    // Prefetch
    static constexpr int PREFETCH_K_BLOCKS = PrefetchKBlocks;

    // Derived constants
    static constexpr int V_LOAD_THREADS = VLoadThreads;
    static constexpr int V_ROWS_PER_THREAD = KV_TILE / V_LOAD_THREADS;

    // QK iteration structure
    static constexpr int QK_K_BLOCKS  = HEAD_DIM / 16;  // K blocks in QK (8 for HEAD_DIM=128)
    static constexpr int QK_N_TILES   = KV_TILE / 8;    // N tiles per K block (8 for KV_TILE=64)

    // S×V iteration structure
    static constexpr int SV_BLOCKS    = 4;    // S×V blocks (2 halves × 2 V row groups)
    static constexpr int SV_NN        = 2;    // V row sub-blocks per block
    static constexpr int SV_DPAS_PER_NN = HEAD_DIM / 16;
};

// ============================================================================
// Hardware-specific configurations. Keep separate types even while their
// values match: the AOT target and the tuning policy must never drift apart.
// ============================================================================

// Intel Arc B580 (Battlemage / BMG / Xe2-HPG)
// 160 XVEs, 20 Xe Cores, doubleGRF, 12GB GDDR6
using ConfigBMG = SdpConfig<BMGTag, 128, 16, 16, 64, 2, 4>;

// Intel Panther Lake H / Arc B390. Correctness and resource use are validated
// with these values. This named policy intentionally starts equal to BMG until
// representative ComfyUI workloads establish a better PTL-H-specific choice.
using ConfigPTLH = SdpConfig<PTLHTag, 128, 16, 16, 64, 2, 4>;

// HEAD_DIM=64 variants — for SD3.5, z-Image, LTX-Video
using ConfigBMG_HD64 = SdpConfig<BMGTag, 64, 16, 16, 64, 2, 4>;
using ConfigPTLH_HD64 = SdpConfig<PTLHTag, 64, 16, 16, 64, 2, 4>;

// ============================================================================
// Active configuration — setup.py defines exactly one architecture macro from
// the validated OMNI_XPU_DEVICE value.
// ============================================================================
#if defined(OMNI_XPU_ARCH_BMG) && defined(OMNI_XPU_ARCH_PTL_H)
#error "Select only one omni_xpu_kernel GPU architecture"
#elif defined(OMNI_XPU_ARCH_BMG)
using ActiveConfig = ConfigBMG;
using ActiveConfigHD64 = ConfigBMG_HD64;
#elif defined(OMNI_XPU_ARCH_PTL_H)
using ActiveConfig = ConfigPTLH;
using ActiveConfigHD64 = ConfigPTLH_HD64;
#else
#error "Define OMNI_XPU_ARCH_BMG or OMNI_XPU_ARCH_PTL_H"
#endif

}  // namespace sdp_config
