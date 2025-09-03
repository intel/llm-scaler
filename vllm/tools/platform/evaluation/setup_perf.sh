#!/bin/bash
gpu_num=$(sudo xpu-smi discovery | grep card | wc -l)
for((i=0; i<$gpu_num; i++)); do
  echo "Set GPU $i freq to 2400Mhz"
  sudo xpu-smi config -d $i -t 0 --frequencyrange 2400,2400
done
 
echo "Set CPU to performance mode"
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 | sudo tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias
