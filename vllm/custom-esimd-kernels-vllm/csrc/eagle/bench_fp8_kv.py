import torch, sys, time
for k in list(sys.modules):
    if k.startswith('custom_esimd_kernels_vllm'): del sys.modules[k]
from custom_esimd_kernels_vllm import eagle_ops

dev = torch.device('xpu:0')
torch.manual_seed(0)
NUM_Q, NUM_KV, HD, PAGE = 8, 1, 256, 64

def make_inputs(seq_len, fp8=True):
    nblocks = (seq_len + PAGE - 1) // PAGE + 2
    q = torch.randn(1, NUM_Q, HD, dtype=torch.float16, device=dev) * 0.1
    nbt = nblocks + 4
    if fp8:
        kv = (torch.randn(2, nbt, PAGE, NUM_KV, HD, device=dev) * 0.1).to(torch.float8_e4m3fn)
    else:
        kv = (torch.randn(2, nbt, PAGE, NUM_KV, HD, dtype=torch.float16, device=dev) * 0.1)
    block_table = torch.arange(nblocks, dtype=torch.int32, device=dev).reshape(1, nblocks)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    out = torch.zeros(1, NUM_Q, HD, dtype=torch.float16, device=dev)
    return q, kv, block_table, seq_lens, out

def bench(seq_len, fp8, iters=500, rounds=5, warmup=50):
    q, kv, bt, sl, out = make_inputs(seq_len, fp8)
    args = (q, kv, bt, sl, out, 1, seq_len, 1.0, 1.0) if fp8 else (q, kv, bt, sl, out, 1, seq_len)
    for _ in range(warmup):
        torch.ops.eagle_ops.page_attn_decode(*args)
    torch.xpu.synchronize()
    best = 1e18
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.eagle_ops.page_attn_decode(*args)
        torch.xpu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best

print(f'{"seq_len":>8} {"fp16(us)":>10} {"fp8(us)":>10} {"fp8/fp16":>10}')
for sl in [128, 256, 512, 741, 1024, 2048, 4096, 8192]:
    t16 = bench(sl, False)
    t8  = bench(sl, True)
    print(f'{sl:>8} {t16:>10.3f} {t8:>10.3f} {t8/t16:>9.1%}', flush=True)
