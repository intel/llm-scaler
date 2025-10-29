export _LLM_SCALER_OMNI_HY3D_PATH="/llm/models/Hunyuan3D-2.1"
python download_hy3d_pbr_model.py
bash replace_cuda_to_xpu.sh /llm/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1/unet/attn_processor.py
bash fix_degradations.sh 