#!/bin/bash
# disable-snap-apparmor-logs.sh
# Quiet AppArmor DENIED messages from snapd

set -e

CONFIG="/etc/apparmor/parser.conf"

echo "[1/2] Updating AppArmor config to disable audit logs..."
if ! grep -q "^no-audit" "$CONFIG"; then
    echo "no-audit" | sudo tee -a "$CONFIG"
fi

echo "[2/2] Restarting AppArmor..."
sudo systemctl restart apparmor

echo "✅ AppArmor snap-confine DENIED logs have been silenced."
