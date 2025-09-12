# Table of Contents

## Installation
- [Can I run the platform benchmark under a bare-metal Ubuntu environment?](#can-i-run-the-platform-benchmark-under-a-bare-metal-ubuntu-environment)
- [Can I use Ubuntu 24.04 LTS as the base OS?](#can-i-use-ubuntu-2404-lts-as-the-base-os)
- [Why can't I see the desktop even with Ubuntu 25.04 desktop version installed?](#why-cant-i-see-the-desktop-even-with-ubuntu-2504-desktop-version-installed)
- [Can I update the kernel version or other drivers of Ubuntu to get the latest fixes?](#can-i-update-the-kernel-version-or-other-drivers-of-ubuntu-to-get-the-latest-fixes)
- [Why do I need to run `native_bkc_setup.sh` before using the `vllm/platform` Docker image?](#why-do-i-need-to-run-native_bkc_setupsh-before-using-the-vllmplatform-docker-image)

## Hardware & Firmware
- [No re-sizable BAR configuration in my BIOS. What can I do to enable B60 with a larger BAR2 size?](#no-re-sizable-bar-configuration-in-my-bios-what-can-i-do-to-enable-b60-with-a-larger-bar2-size)
- [Maxsun 2x GPU Card Not Detected Behind PCIe Switch](#maxsun-2x-gpu-card-not-detected-behind-pcie-switch)

## Benchmarking
- [Why do I see unusually high Device-to-Device bandwidth in `ze_peak` benchmark?](#why-do-i-see-unusually-high-device-to-device-bandwidth-in-ze_peak-benchmark)
- [How can I verify if the benchmark data from `platform_basic_evaluation.sh` is valid?](#how-can-i-verify-if-the-benchmark-data-from-platform_basic_evaluationsh-is-valid)

## Tools
- [Why do I always see `2.5GT/s` PCIe link status in `lspci`](#why-do-i-always-see-25gts-pcie-link-status-in-lspci)
- [Why can't I see `xpu-smi` in the `vllm` Docker image?](#why-cant-i-see-xpu-smi-in-the-vllm-docker-image)
- [Why can't I see GPU utilization with `xpu-smi`?](#why-cant-i-see-gpu-utilization-with-xpu-smi)

---

# Installation

## Can I run the platform benchmark under a bare-metal Ubuntu environment?

Yes. Please contact the Intel support team to obtain an offline installer for native setup.
We also plan to make the offline installer publicly available on the Intel RDC website in an upcoming release.

## Can I use Ubuntu 24.04 LTS as the base OS? {#can-i-use-ubuntu-2404-lts-as-the-base-os}

Not yet. Support for Ubuntu 24.04 LTS is planned in future releases (targeting late 2025).

## Why can't I see the desktop even with Ubuntu 25.04 desktop version installed? {#why-cant-i-see-the-desktop-even-with-ubuntu-2504-desktop-version-installed}	

Some versions of Ubuntu may default to text mode (multi-user target) after installation. You can check the current mode:

```bash
sudo systemctl get-default
```

If it returns `multi-user.target`, you can switch to graphical mode:

```bash
sudo systemctl set-default graphical.target
sudo reboot
```

## Can I update the kernel version or other drivers of Ubuntu to get the latest fixes?

During the evaluation phase, we **do not recommend updating the kernel or system packages** to ensure consistency with the validated environment.
Any updates may affect stability or introduce compatibility issues with pre-installed components.

## Why do I need to run `native_bkc_setup.sh` before using the `vllm/platform` Docker image?

To ensure consistent kernel and firmware behavior, `native_bkc_setup.sh` is required to unify Linux kernel version and install B60 GuC/HuC firmware directly on the host system before using the container image.

---

# Hardware & Firmware

## No re-sizable BAR configuration in my BIOS. What can I do to enable B60 with a larger BAR2 size?

Please contact your AIB (Add-In-Board) vendor to request the latest IFWI (firmware image) with max re-sizable BAR pre-configured.
This setup has been validated on Gunnir and Maxsun B60 cards.

## Maxsun 2x GPU Card Not Detected Behind PCIe Switch

Many PCIe switch firmware versions do not support PCIe bifurcation, which prevents detection of dual-GPU cards like Maxsun 2x.

Solution: A firmware update for the PCIe switch is required.
The Broadcom PEX 89104 has been validated. Please contact your PCIe switch vendor for support or an updated firmware.

---

# Benchmarking

## Why do I see unusually high Device-to-Device bandwidth in `ze_peak` benchmark?

Please export the following environment variable before running ze_peak.

```bash
export NEOReadDebugKeys=1
export RenderCompressedBuffersEnabled=0
```

## How can I verify if the benchmark data from `platform_basic_evaluation.sh` is valid?

Sample benchmark results are available in:

```
/opt/intel/multi-arc/results
```

These data points are collected from internal evaluations using an Intel® Xeon® W5-2545X system with dual B60 GPUs.
> **Disclaimer**: This reference is provided for informational purposes only and should not be interpreted as official performance indicators or guarantees. Actual results may vary depending on hardware configuration, software stack, and usage scenarios.

---

# Tools

## Why do I always see `2.5GT/s` PCIe link status in `lspci`
It's an known issue but no real impact on GPU operation. Official explaination: https://www.intel.com/content/www/us/en/support/articles/000094587/graphics.html

## Why can't I see `xpu-smi` in the `vllm` Docker image?

Due to release process limitations, `xpu-smi` is currently not included in the official `vllm` Docker image.
We plan to add it in the next release. In the meantime, you may install it manually using:

[xpu-smi 1.3.1 on GitHub](https://github.com/intel/xpumanager/releases/download/V1.3.1/xpumanager_1.3.1_20250724.061629.60921e5e_u24.04_amd64.deb)

## Why can't I see GPU utilization with `xpu-smi`?

GPU utilization metrics are not yet fully supported by `xpu-smi` in the current release.
This functionality is scheduled to be added in next release.
