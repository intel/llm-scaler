#!/bin/bash
# disable-auto-upgrade.sh
# Permanently disable automatic updates on Ubuntu

set -e

echo "[1/4] Disable unattended-upgrades service..."
sudo systemctl stop unattended-upgrades.service || true
sudo systemctl disable unattended-upgrades.service || true

echo "[2/4] Disable apt-daily timers..."
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer || true
sudo systemctl disable apt-daily.timer apt-daily-upgrade.timer || true

echo "[3/4] Update APT config to disable periodic upgrades..."
CONFIG_FILE="/etc/apt/apt.conf.d/20auto-upgrades"
if [ -f "$CONFIG_FILE" ]; then
    sudo sed -i 's/^\(APT::Periodic::Update-Package-Lists\).*/\1 "0";/' "$CONFIG_FILE"
    sudo sed -i 's/^\(APT::Periodic::Unattended-Upgrade\).*/\1 "0";/' "$CONFIG_FILE"
else
    echo 'APT::Periodic::Update-Package-Lists "0";' | sudo tee "$CONFIG_FILE"
    echo 'APT::Periodic::Unattended-Upgrade "0";' | sudo tee -a "$CONFIG_FILE"
fi

echo "[4/4] Disable Snap auto-refresh..."
sudo systemctl stop snapd.snap-repair.timer || true
sudo systemctl disable snapd.snap-repair.timer || true

echo "✅ Automatic updates have been disabled permanently."
