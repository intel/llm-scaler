WORK_DIR=/tmp/multi-arc
mkdir -p $WORK_DIR

echo -e "\n[INFO] Downloading and installing GPU firmware..."
FIRMWARE_DIR=$WORK_DIR/firmware
mkdir -p "$FIRMWARE_DIR"
cd "$FIRMWARE_DIR"
rm -rf ./*
wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_guc_70.bin
wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_huc.bin
zstd -1 bmg_guc_70.bin -o bmg_guc_70.bin.zst
zstd -1 bmg_huc.bin -o bmg_huc.bin.zst
 
if [ -d /lib/firmware/xe ]; then
  cp *.zst /lib/firmware/xe
else
  echo "[ERROR] /lib/firmware/xe does not exist. Ensure your system supports Xe firmware."
  exit 1
fi

update-initramfs -u
echo -e "Update GPU firmware successfully, please reboot to apply changes!"
