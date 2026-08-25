import sys

_PKG = "ComfyUI-OmniXPU"


class OmniXPUStatus:
    CATEGORY = "OmniXPU"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get_status"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def get_status(self):
        lines = ["=== ComfyUI-OmniXPU Status ==="]

        # GPU info
        try:
            import torch
            if torch.xpu.is_available():
                name = torch.xpu.get_device_name(0)
                mem = torch.xpu.get_device_properties(0).total_memory // (1024 * 1024)
                lines.append(f"  GPU: {name} ({mem} MB)")
        except Exception:
            pass

        # omni_xpu_kernel probe
        probe = sys.modules.get(f"{_PKG}.probe")
        if probe:
            try:
                s = probe.summary()
                lines.append(f"  omni_xpu_kernel: {s['version'] or 'not installed'}")
                capabilities = ("sdp", "norm", "rotary", "linear_fp8", "int8")
                caps = [k for k in capabilities if s.get(k)]
                missing = [k for k in capabilities if not s.get(k)]
                if caps:
                    lines.append(f"    available: {', '.join(caps)}")
                if missing:
                    lines.append(f"    missing:   {', '.join(missing)}")
            except Exception:
                pass

        # Runtime providers are activated during prestartup, before this custom
        # node package imports Torch. Keep their ownership and fallback state
        # visible without importing either provider again.
        runtime = sys.modules.get("_comfyui_omnixpu_runtime_bootstrap")
        if runtime and hasattr(runtime, "get_state"):
            try:
                provider_state = runtime.get_state()
                lines.append(
                    "  runtime providers: "
                    f"{provider_state.get('status', 'unknown')} "
                    f"(mode={provider_state.get('mode', 'unknown')})"
                )
                for provider_id, state in sorted(
                    provider_state.get("providers", {}).items()
                ):
                    line = f"    {provider_id}: {state.get('status', 'unknown')}"
                    if state.get("reason"):
                        line += f" ({state['reason']})"
                    lines.append(line)
                for error in provider_state.get("errors", ()):
                    lines.append(f"    rejected: {error}")
            except Exception as exc:
                lines.append(f"  runtime providers: diagnostics failed ({exc})")

        # Kitchen is the authority for generic operator registration and
        # fallback. Report it separately from custom-node adapters.
        try:
            import comfy_kitchen as ck

            kitchen_xpu = ck.list_backends().get("xpu", {})
            available = kitchen_xpu.get("available", False)
            disabled = kitchen_xpu.get("disabled", False)
            state = "available" if available and not disabled else "unavailable"
            if disabled:
                state = "disabled"
            capabilities = kitchen_xpu.get("capabilities", [])
            lines.append(
                f"  comfy_kitchen XPU: {state} "
                f"({len(capabilities)} capabilities)"
            )
            reason = kitchen_xpu.get("unavailable_reason")
            if reason:
                lines.append(f"    reason: {reason}")
        except Exception as exc:
            lines.append(f"  comfy_kitchen XPU: unavailable ({exc})")

        # Patch status
        lines.append("")
        patches = sys.modules.get(f"{_PKG}.patches")
        if patches:
            for entry in patches.get_status():
                name = entry["name"]
                kind = entry.get("kind", "component")
                status = entry["status"]
                reason = entry.get("reason", "")
                mark = {"applied": "+", "skipped": "-", "failed": "!!"}.get(status, "?")
                line = f"  [{mark}] {name} [{kind}]: {status}"
                if reason:
                    line += f" ({reason})"
                lines.append(line)

        # Attention stats
        attn = sys.modules.get(f"{_PKG}.adapters.attention")
        if attn and hasattr(attn, "get_stats"):
            try:
                stats = attn.get_stats()
                if (
                    stats["cute"]
                    or stats["esimd"]
                    or stats["torch_sdpa"]
                    or stats["fallback"]
                ):
                    lines.append("")
                    lines.append(
                        "  Attention calls: "
                        f"cute={stats['cute']} "
                        f"esimd={stats['esimd']} "
                        f"torch={stats['torch_sdpa']} "
                        f"fallback={stats['fallback']}"
                    )
                    for r, c in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
                        lines.append(f"    {r}: {c}")
            except Exception:
                pass

        # Fused INT8 FFN routing stats
        int8_ffn = sys.modules.get(f"{_PKG}.adapters.int8_ffn")
        if int8_ffn and hasattr(int8_ffn, "get_stats"):
            try:
                stats = int8_ffn.get_stats()
                if stats["routed"] or stats["fallback"]:
                    lines.append("")
                    lines.append(
                        "  INT8 FFN calls: "
                        f"fused={stats['routed']} fallback={stats['fallback']}"
                    )
                    for reason, count in sorted(
                        stats["reasons"].items(), key=lambda item: -item[1]
                    ):
                        lines.append(f"    {reason}: {count}")
            except Exception:
                pass

        return ("\n".join(lines),)
