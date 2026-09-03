#pragma once

// Build-time T1 tuning controls. Defaults and candidate-build detection are
// generated from the maintained policy manifest. These values are shared by
// an architecture wheel; SKU-local defaults belong in Bmg*KernelPolicy.

#if !defined(OMNI_XPU_ARCH_PTL_H) && !defined(OMNI_XPU_ARCH_BMG)
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#if defined(OMNI_XPU_ARCH_PTL_H) && defined(OMNI_XPU_ARCH_BMG)
#error "Define exactly one XPU architecture"
#endif

#include "generated/kernel_tuning_defaults_generated.h"

#if OMNI_RMS_NORM_H120_MODE < 0 || OMNI_RMS_NORM_H120_MODE > 2
#error "OMNI_RMS_NORM_H120_MODE must be 0, 1, or 2"
#endif

#if OMNI_RMS_NORM_H128_BLOCK_SIZE != 16 && \
    OMNI_RMS_NORM_H128_BLOCK_SIZE != 32 && \
    OMNI_RMS_NORM_H128_BLOCK_SIZE != 64 && \
    OMNI_RMS_NORM_H128_BLOCK_SIZE != 128
#error "OMNI_RMS_NORM_H128_BLOCK_SIZE must be 16, 32, 64, or 128"
#endif

#if OMNI_GROUP_NORM_BMG_TILE != 1024 && \
    OMNI_GROUP_NORM_BMG_TILE != 2048 && \
    OMNI_GROUP_NORM_BMG_TILE != 4096 && \
    OMNI_GROUP_NORM_BMG_TILE != 8192 && \
    OMNI_GROUP_NORM_BMG_TILE != 16384 && \
    OMNI_GROUP_NORM_BMG_TILE != 32768 && \
    OMNI_GROUP_NORM_BMG_TILE != 65536
#error "unsupported OMNI_GROUP_NORM_BMG_TILE"
#endif

#if OMNI_GROUP_NORM_BMG_REDUCE_VECTOR != 16 && \
    OMNI_GROUP_NORM_BMG_REDUCE_VECTOR != 32 && \
    OMNI_GROUP_NORM_BMG_REDUCE_VECTOR != 64
#error "unsupported OMNI_GROUP_NORM_BMG_REDUCE_VECTOR"
#endif

#if OMNI_H3_RMS_ROPE_FAST_REDUCE != 0 && \
    OMNI_H3_RMS_ROPE_FAST_REDUCE != 1
#error "OMNI_H3_RMS_ROPE_FAST_REDUCE must be zero or one"
#endif

#if OMNI_H3_RMS_ROPE_SLM_BF16 != 0 && OMNI_H3_RMS_ROPE_SLM_BF16 != 1
#error "OMNI_H3_RMS_ROPE_SLM_BF16 must be zero or one"
#endif

#if OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE < 0
#error "OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE must be zero or positive"
#endif

#if OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE < 0
#error "OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE must be zero or positive"
#endif
