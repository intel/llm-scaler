#pragma once

// Build-time T1 tuning controls. Every default below reproduces the maintained
// PTL-H or BMG source route. Non-default builds are development candidates and
// still require exact-contract correctness before timing.

#if !defined(OMNI_XPU_ARCH_PTL_H) && !defined(OMNI_XPU_ARCH_BMG)
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#if defined(OMNI_XPU_ARCH_PTL_H) && defined(OMNI_XPU_ARCH_BMG)
#error "Define exactly one XPU architecture"
#endif

#ifndef OMNI_FP8_DEQUANT_ELEMENTS_PER_WI
#define OMNI_FP8_DEQUANT_ELEMENTS_PER_WI 256
#endif

#ifndef OMNI_FP8_QUANT_VEC
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_FP8_QUANT_VEC 16
#else
#define OMNI_FP8_QUANT_VEC 8
#endif
#endif

#ifndef OMNI_FP8_STOCHASTIC_ELEMENTS_PER_WORK_ITEM
#if defined(OMNI_XPU_ARCH_BMG)
#define OMNI_FP8_STOCHASTIC_ELEMENTS_PER_WORK_ITEM 6
#else
#define OMNI_FP8_STOCHASTIC_ELEMENTS_PER_WORK_ITEM 8
#endif
#endif

#ifndef OMNI_CONVROT_DEQUANT_WG_SIZE
#define OMNI_CONVROT_DEQUANT_WG_SIZE 1
#endif

#ifndef OMNI_CONVROT_QUANT_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_CONVROT_QUANT_WG_SIZE 1
#else
#define OMNI_CONVROT_QUANT_WG_SIZE 8
#endif
#endif

#ifndef OMNI_INT8_DEQUANT_ELEMENTS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_INT8_DEQUANT_ELEMENTS_PER_WI 64
#else
#define OMNI_INT8_DEQUANT_ELEMENTS_PER_WI 32
#endif
#endif

#ifndef OMNI_SILU_MUL_ELEMENTS_PER_WI
#define OMNI_SILU_MUL_ELEMENTS_PER_WI 1
#endif

#ifndef OMNI_INT8_TENSORWISE_VEC
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_INT8_TENSORWISE_VEC 16
#else
#define OMNI_INT8_TENSORWISE_VEC 8
#endif
#endif

#ifndef OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE
#define OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE 1
#endif

#ifndef OMNI_KITCHEN_ROPE_PAIR_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_KITCHEN_ROPE_PAIR_WG_SIZE 128
#else
#define OMNI_KITCHEN_ROPE_PAIR_WG_SIZE 32
#endif
#endif

#ifndef OMNI_SVDQ_DEQUANT_GROUPS_PER_WI
#define OMNI_SVDQ_DEQUANT_GROUPS_PER_WI 60
#endif

#ifndef OMNI_SVDQ_QUANT_GROUPS_PER_WI
#define OMNI_SVDQ_QUANT_GROUPS_PER_WI 60
#endif

#ifndef OMNI_SVDQ_UNPACK_COLS_PER_WI
#define OMNI_SVDQ_UNPACK_COLS_PER_WI 3840
#endif

#ifndef OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION 64
#else
#define OMNI_SVDQ_UNPACK_BYTES_PER_ITERATION 128
#endif
#endif

#ifndef OMNI_SVDQ_UNPACK_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_SVDQ_UNPACK_WG_SIZE 32
#else
#define OMNI_SVDQ_UNPACK_WG_SIZE 1
#endif
#endif

// Accepted fixed routes that previously had no maintained T1 build surface.
#ifndef OMNI_RMS_NORM_H120_MODE
#define OMNI_RMS_NORM_H120_MODE 2
#endif

#ifndef OMNI_RMS_NORM_H128_BLOCK_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_RMS_NORM_H128_BLOCK_SIZE 32
#else
#define OMNI_RMS_NORM_H128_BLOCK_SIZE 64
#endif
#endif

#ifndef OMNI_GROUP_NORM_BMG_TILE
#define OMNI_GROUP_NORM_BMG_TILE 32768
#endif

#ifndef OMNI_GROUP_NORM_BMG_REDUCE_VECTOR
#define OMNI_GROUP_NORM_BMG_REDUCE_VECTOR 32
#endif

#ifndef OMNI_H3_RMS_ROPE_FAST_REDUCE
#define OMNI_H3_RMS_ROPE_FAST_REDUCE 0
#endif

#ifndef OMNI_H3_RMS_ROPE_SLM_BF16
#define OMNI_H3_RMS_ROPE_SLM_BF16 0
#endif

// Zero preserves each exact route's independently validated value. A positive
// override applies to every accepted large-row route in the selected target.
#ifndef OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE
#define OMNI_ROWQ_VECTOR_WIDTH_OVERRIDE 0
#endif

#ifndef OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE
#define OMNI_ROWQ_SUBGROUPS_PER_ROW_OVERRIDE 0
#endif

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
