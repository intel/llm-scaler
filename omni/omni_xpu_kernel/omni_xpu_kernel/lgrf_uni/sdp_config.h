// ============================================================================
// SDP Kernel Configuration — compile-time hardware-specific parameters
// ============================================================================
// Architecture tags keep AOT policy selection explicit. The current ESIMD
// implementation has a fixed WG/tile mapping; only head dimension is a safe
// independent template parameter until those hard-coded mappings are refactored.
// ============================================================================
#pragma once

#include <cstdint>

namespace sdp_config {

struct BMGTag {};
struct PTLHTag {};

template <typename ArchTag, int HeadDim>
struct SdpConfig {
    static_assert(HeadDim == 64 || HeadDim == 128, "LGRF SDP supports head_dim 64 or 128");

    // These are implementation constraints, not independent policy knobs:
    // kernels map local_id with &0xf, advance Q by 16 rows per work-item, and
    // use fixed 64-row K/V load and DPAS loops.
    static constexpr int WG_SIZE      = 16;
    static constexpr int Q_PER_THREAD = 16;
    static constexpr int Q_GROUP      = WG_SIZE * Q_PER_THREAD;
    static constexpr int HEAD_DIM     = HeadDim;
    static constexpr int KV_TILE      = 64;
};

// ============================================================================
// Hardware-specific configurations. Keep separate types even while their
// values match: the AOT target and the tuning policy must never drift apart.
// ============================================================================

// Intel Arc B580 (Battlemage / BMG / Xe2-HPG)
// 160 XVEs, 20 Xe Cores, doubleGRF, 12GB GDDR6
using ConfigBMG = SdpConfig<BMGTag, 128>;

// Intel Panther Lake H / Arc B390. The fixed implementation mapping was
// performance- and spill-validated with representative ComfyUI workloads.
using ConfigPTLH = SdpConfig<PTLHTag, 128>;

// HEAD_DIM=64 variants — for SD3.5, z-Image, LTX-Video
using ConfigBMG_HD64 = SdpConfig<BMGTag, 64>;
using ConfigPTLH_HD64 = SdpConfig<PTLHTag, 64>;

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
