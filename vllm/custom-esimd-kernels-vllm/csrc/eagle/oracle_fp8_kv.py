import torch, sys
for k in list(sys.modules):
    if k.startswith("custom_esimd_kernels_vllm"): del sys.modules[k]
from custom_esimd_kernels_vllm import eagle_ops

dev = torch.device("xpu:0")
torch.manual_seed(0)
NUM_Q, NUM_KV, HD, PAGE = 8, 1, 256, 64

def run(seq_len, fp8_dtype):
    nblocks = (seq_len + PAGE - 1)//PAGE + 2
    nbt = nblocks + 4
    q = (torch.randn(1, NUM_Q, HD, dtype=torch.float16, device=dev) * 0.2)
    kv8 = (torch.randn(2, nbt, PAGE, NUM_KV, HD, device=dev) * 0.2).to(fp8_dtype)
    bt = torch.arange(nblocks, dtype=torch.int32, device=dev).reshape(1, nblocks)
    sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    # fp8 kernel
    o8 = torch.zeros(1, NUM_Q, HD, dtype=torch.float16, device=dev)
    torch.ops.eagle_ops.page_attn_decode(q, kv8, bt, sl, o8, 1, seq_len, 1.0, 1.0)
    # fp16 oracle: dequant same bytes to fp16, run fp16 kernel
    kv16 = kv8.to(torch.float16)
    o16 = torch.zeros(1, NUM_Q, HD, dtype=torch.float16, device=dev)
    torch.ops.eagle_ops.page_attn_decode(q, kv16, bt, sl, o16, 1, seq_len)
    torch.xpu.synchronize()
    return (o8.cpu().float() - o16.cpu().float()).abs().max().item()

for name, dt in [("e4m3fn", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]:
    print("=== %s (fp8 kernel vs fp16-oracle) ===" % name)
    worst = 0.0
    for sl in [128, 256, 512, 741, 1024, 2048, 4096]:
        md = run(sl, dt)
        worst = max(worst, md)
        print("  seq=%5d  max|d|=%.5f" % (sl, md))
    print("  WORST=%.5f" % worst)
