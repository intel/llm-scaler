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
                caps = [k for k in ("sdp", "norm", "rotary", "linear_fp8") if s.get(k)]
                missing = [k for k in ("sdp", "norm", "rotary", "linear_fp8") if not s.get(k)]
                if caps:
                    lines.append(f"    available: {', '.join(caps)}")
                if missing:
                    lines.append(f"    missing:   {', '.join(missing)}")
            except Exception:
                pass

        # Patch status
        lines.append("")
        patches = sys.modules.get(f"{_PKG}.patches")
        if patches:
            for entry in patches.get_status():
                name = entry["name"]
                status = entry["status"]
                reason = entry.get("reason", "")
                mark = {"applied": "+", "skipped": "-", "failed": "!!"}.get(status, "?")
                line = f"  [{mark}] {name}: {status}"
                if reason:
                    line += f" ({reason})"
                lines.append(line)

        # Attention stats
        attn = sys.modules.get(f"{_PKG}.patches.patch_attention")
        if attn and hasattr(attn, "get_stats"):
            try:
                stats = attn.get_stats()
                if stats["esimd"] or stats["fallback"]:
                    lines.append("")
                    lines.append(f"  Attention calls: esimd={stats['esimd']} fallback={stats['fallback']}")
                    for r, c in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
                        lines.append(f"    {r}: {c}")
            except Exception:
                pass

        return ("\n".join(lines),)
