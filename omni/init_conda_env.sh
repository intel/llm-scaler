conda create -n omni_env python=3.10 -y
conda activate omni_env

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
pip install bigdl-core-xe-all==2.7.0b20250625
apt remove python3-blinker -y

# Install ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git checkout 51696e3fdcdfad657cb15854345fbcbbe70eef8d
git apply ../patches/comfyui_for_multi_arc.patch
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite
cd comfyui-videohelpersuite
pip install -r requirements.txt
cd ..
git clone https://github.com/komikndr/raylight.git
cd raylight
git checkout ff8e90ba1f2c2d23e3ac23746910ddfb523fc8f1
git apply ../../../patches/raylight_for_multi_arc.patch
pip install ray==2.49.2
pip install -r requirements.txt
cd ..
git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use
cd comfyui-easy-use
pip install -r requirements.txt
cd ..
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux
apt install libcairo2-dev pkg-config python3-dev -y
pip install -r requirements.txt
cd ..
git clone https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm
cd comfyui-voxcpm
git checkout 044dd93c0effc9090fb279117de5db4cd90242a0
git apply ../../../patches/comfyui_voxcpm_for_xpu.patch
pip install -r requirements.txt
cd ..
