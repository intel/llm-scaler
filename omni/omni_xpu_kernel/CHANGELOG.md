# Changelog

Notable user-facing changes will be recorded here when a public
`omni_xpu_kernel` release is prepared.

## Unreleased

No public release has been published.

## 0.1.0b9.dev1 - 2026-08-05

- Vendor the validated oneDNN 3.9.1 runtime and redistribution notices in
  Windows wheels.
- Prefer the wheel-private `dnnl.dll` at runtime and retain system oneAPI only
  as a source-checkout and legacy-wheel fallback.
- Remove the ineffective Windows `onednn` Python package dependency while
  preserving the Linux dependency and runtime layout.
