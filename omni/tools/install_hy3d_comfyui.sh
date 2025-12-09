git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git && \
cd ComfyUI-Hunyuan3d-2-1 && \
git checkout 9d7ef32509101495a7840b3ae8e718c8d1183305 && \
git apply /tmp/comfyui_hunyuan3d_for_xpu.patch && \
pip install bigdl-core==2.4.0b1 rembg realesrgan && \
pip install -r requirements.txt && \
cd hy3dpaint/custom_rasterizer && \
python setup.py install && \
cd ../DifferentiableRenderer && \
python setup.py install 