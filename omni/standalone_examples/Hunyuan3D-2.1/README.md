# Docker setup

Build docker image:

```bash
bash build.sh
```

Run docker image:

```bash
export DOCKER_IMAGE=llm-scaler-visualai:latest
export CONTAINER_NAME=hunyuan3d-2.1
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        --entrypoint=/bin/bash \
        $DOCKER_IMAGE 
```

Run Hunyuan 3D 2.1 demo:
```bash
docker exec -it hunyuan3d-2.1 bash
# At /llm/Hunyuan3D-2.1 path

# Configure proxy to download model files
export http_proxy=<your_http_proxy>
export https_proxy=<your_https_proxy>
export no_proxy=localhost,127.0.0.1

# Run shape + paint demo
python3 demo.py
# (Optional) Run shape only demo
python3 demo_shape.py
# (Optional) Run paint only demo
python3 demo_texture.py
```

Known issues:

- `attn_processor.py` in `hunyuan3d-paintpbr-v2-1` is cuda hard-coded. So you may encounter `Torch not compiled with CUDA enabled` issue:

```
  File "/root/.cache/huggingface/modules/diffusers_modules/local/attn_processor.py", line 750, in __call__
    processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c").to("cuda:0")
  File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 363, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```

It can be fixed by `replace_cuda_to_xpu.sh`:
```bash
bash replace_cuda_to_xpu.sh /root/.cache/huggingface/hub/models--tencent--Hunyuan3D-2.1/snapshots/22ba78d67f37eac53f3b0f019e0856b8f00fb9b5/hunyuan3d-paintpbr-v2-1/unet/attn_processor.py
```
