git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git && \
cd ComfyUI-Hunyuan3d-2-1 && \
git checkout 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
wget https://github.com/intel/llm-scaler/raw/refs/heads/main/omni/patches/comfyui_hunyuan3d_for_xpu.patch && \
git apply ./comfyui_hunyuan3d_for_xpu.patch && \
pip install bigdl-core==2.4.0b1 rembg realesrgan && \
pip install -r requirements.txt && \
cd hy3dpaint/custom_rasterizer && \
python setup.py install && \
cd ../DifferentiableRenderer && \
<<<<<<< HEAD
python setup.py install 
=======
python setup.py install 
>>>>>>> 954033b8087234f342f9d44b3281e9e409de3a48
