# Changelog

Notable user-facing changes will be recorded here when a public
`omni_xpu_kernel` release is prepared.

## Unreleased

- Support the current Torch 2.13 XPU runtime and matched oneAPI 2026.0
  oneDNN 3.11.2 development contract on Windows.
- Vendor the validated `dnnl.dll` and redistribution notices in Windows
  wheels.
- Add the explicit Windows BMG CUTE FMHA and Sol-Attn sidecar build.
- Discover ABI-suffixed Windows `.pyd` extensions and expose the packaged
  Sol-Attn API.
- Keep Windows CUTE compilation and ComfyUI routing as separate explicit
  opt-ins.
- Use the same BMG CUTE fat-target contract on Linux and Windows, containing
  both G21 and G31 AOT device images for the maintained B60 and B70 SKU set.
- Keep legacy/default Sol-Attn calls on a compile-time no-controls mainloop
  while preserving the separate bias, block-length, route, and tail controls.
