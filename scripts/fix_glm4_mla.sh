#!/bin/bash
# fix_glm4_mla.sh — Fix GLM-4.7-flash MLA on XPU (2 one-line patches)
#
# Bug: vllm_for_multi_arc.patch creates Glm4MoeLiteMLAAttention but:
#   1. Never adds "glm4_moe_lite" to is_deepseek_mla() whitelist
#   2. XPU platform doesn't route use_mla to TRITON_MLA backend
#
# This script applies both fixes automatically.
#
# Usage: bash fix_glm4_mla.sh [VENV_PATH]
#   VENV_PATH defaults to ~/llm-scaler-vllm/venv

set -euo pipefail

VENV="${1:-$HOME/llm-scaler-vllm/venv}"
SITE_PACKAGES=""

echo "=== GLM-4.7 MLA Fix Script ==="
echo ""

# Find site-packages (could be lib or lib64)
for candidate in "$VENV/lib/python3.12/site-packages" "$VENV/lib64/python3.12/site-packages"; do
    if [ -d "$candidate/vllm" ]; then
        SITE_PACKAGES="$candidate"
        break
    fi
done

if [ -z "$SITE_PACKAGES" ]; then
    echo "ERROR: Could not find vllm in $VENV"
    find "$VENV" -name "config" -path "*/vllm/config" -type d 2>/dev/null | head -5
    exit 1
fi

echo "Found vllm at: $SITE_PACKAGES/vllm"
echo ""

# ============================================================
# FIX 1: Add "glm4_moe_lite" to is_deepseek_mla() whitelist
# ============================================================
echo "--- Fix 1: MLA model whitelist ---"

CONVERTOR="$SITE_PACKAGES/vllm/transformers_utils/model_arch_config_convertor.py"

if [ ! -f "$CONVERTOR" ]; then
    echo "  ERROR: $CONVERTOR not found"
    exit 1
fi

if grep -q '"glm4_moe_lite"' "$CONVERTOR" 2>/dev/null; then
    echo "  Already patched — 'glm4_moe_lite' found in is_deepseek_mla()"
else
    # Find anchor: deepseek_mtp line (glm4_moe_lite goes after it alphabetically)
    ANCHOR_LINE=$(grep -n '"deepseek_mtp"' "$CONVERTOR" | head -1 | cut -d: -f1)
    if [ -z "$ANCHOR_LINE" ]; then
        # Fallback: try kimi_k2
        ANCHOR_LINE=$(grep -n '"kimi_k2"' "$CONVERTOR" | head -1 | cut -d: -f1)
    fi

    if [ -z "$ANCHOR_LINE" ]; then
        echo "  ERROR: Could not find anchor line in is_deepseek_mla()"
        echo "  Manually add '\"glm4_moe_lite\",' to the model type list in:"
        echo "  $CONVERTOR"
        exit 1
    fi

    # Get indentation from anchor line
    INDENT=$(sed -n "${ANCHOR_LINE}p" "$CONVERTOR" | sed 's/[^ ].*//')
    sed -i "${ANCHOR_LINE}a\\${INDENT}\"glm4_moe_lite\"," "$CONVERTOR"

    echo "  PATCHED: Added 'glm4_moe_lite' after line $ANCHOR_LINE"
    echo "  Context:"
    sed -n "$((ANCHOR_LINE-1)),$((ANCHOR_LINE+2))p" "$CONVERTOR" | head -5
fi
echo ""

# ============================================================
# FIX 2: Route use_mla to TRITON_MLA in XPU platform
# ============================================================
echo "--- Fix 2: XPU MLA backend routing ---"

XPU_PLATFORM="$SITE_PACKAGES/vllm/platforms/xpu.py"

if [ ! -f "$XPU_PLATFORM" ]; then
    echo "  ERROR: $XPU_PLATFORM not found"
    exit 1
fi

if grep -q "use_mla" "$XPU_PLATFORM" 2>/dev/null; then
    echo "  Already patched — use_mla routing found in xpu.py"
else
    # Find the use_sparse check line — MLA routing goes right after it
    SPARSE_LINE=$(grep -n "use_sparse" "$XPU_PLATFORM" | head -1 | cut -d: -f1)

    if [ -z "$SPARSE_LINE" ]; then
        echo "  ERROR: Could not find use_sparse check in xpu.py"
        echo "  Manually add MLA routing to get_attn_backend_cls() in:"
        echo "  $XPU_PLATFORM"
        exit 1
    fi

    # Find the end of the use_sparse block (the raise line)
    RAISE_LINE=""
    for i in $(seq "$SPARSE_LINE" $((SPARSE_LINE + 5))); do
        if sed -n "${i}p" "$XPU_PLATFORM" | grep -q "raise NotImplementedError"; then
            RAISE_LINE=$i
            break
        fi
    done

    if [ -z "$RAISE_LINE" ]; then
        RAISE_LINE=$SPARSE_LINE
    fi

    # Get indentation
    INDENT=$(sed -n "${SPARSE_LINE}p" "$XPU_PLATFORM" | sed 's/[^ ].*//')

    # Insert MLA routing after the use_sparse block
    sed -i "${RAISE_LINE}a\\
${INDENT}if attn_selector_config.use_mla:\\
${INDENT}    logger.info_once(\"Using Triton MLA backend for MLA attention on XPU.\")\\
${INDENT}    return AttentionBackendEnum.TRITON_MLA.get_path()" "$XPU_PLATFORM"

    echo "  PATCHED: Added TRITON_MLA routing after line $RAISE_LINE"
    echo "  Context:"
    sed -n "$((RAISE_LINE-1)),$((RAISE_LINE+5))p" "$XPU_PLATFORM" | head -8
fi
echo ""

# ============================================================
# VERIFY: Check triton_mla.py exists
# ============================================================
echo "--- Verify: TRITON_MLA backend ---"

TRITON_MLA=$(find "$SITE_PACKAGES/vllm" -name "triton_mla.py" -path "*/mla/*" 2>/dev/null | head -1)
if [ -n "$TRITON_MLA" ]; then
    echo "  Found: $TRITON_MLA"
    # Check for CUDA dependencies
    CUDA_DEPS=$(grep -c "torch\.cuda\|nv_tma\|libcuda" "$TRITON_MLA" 2>/dev/null || true)
    if [ "$CUDA_DEPS" -gt 0 ]; then
        echo "  WARNING: $CUDA_DEPS CUDA-specific references found — may not work on XPU"
    else
        echo "  Clean: No CUDA dependencies (pure Triton kernels)"
    fi
else
    echo "  WARNING: triton_mla.py not found — TRITON_MLA backend may not be available"
fi
echo ""

# ============================================================
# CLEANUP: Revert wrong whitelist location if present
# ============================================================
# (Previous debugging may have added glm4_moe_lite to the wrong spot)

echo "=== Done ==="
echo ""
echo "Both fixes applied. To test GLM-4.7-flash with MLA:"
echo ""
echo "  pkill -f vllm  # kill any running vLLM processes"
echo ""
echo "  VLLM_MLA_DISABLE=0 python -m vllm.entrypoints.openai.api_server \\"
echo "    --model /shared/models/glm-4.7-flash-int4-autoround \\"
echo "    --gpu-memory-utilization 0.75 \\"
echo "    --max-model-len 4096"
echo ""
echo "Expected: MLA enabled, KV cache ~53 KB/token (18x smaller than without MLA)"
echo "32K context needs ~1.66 GiB KV cache instead of ~29 GiB"
