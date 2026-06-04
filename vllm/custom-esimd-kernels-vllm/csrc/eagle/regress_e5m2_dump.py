import torch, sys, glob
for k in list(sys.modules):
    if k.startswith("custom_esimd_kernels_vllm"): del sys.modules[k]
from custom_esimd_kernels_vllm import eagle_ops
dev = torch.device("xpu:0")
worst = 0.0
for f in sorted(glob.glob("/tmp/esimd_dump/esimd_dump.*.r0.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    kv = d["kv_cache_raw"].view(torch.float8_e5m2).to(dev)
    q  = d["q"].to(dev); bt = d["block_table"].to(dev).to(torch.int32); sl = d["seq_lens"].to(dev).to(torch.int32)
    out = torch.zeros_like(q)
    torch.ops.eagle_ops.page_attn_decode(q, kv, bt, sl, out, 1, int(d["max_seq_len"]), float(d["k_scale"]), float(d["v_scale"]))
    torch.xpu.synchronize()
    md = (out.cpu().float() - d["ref_out"].float()).abs().max().item()
    worst = max(worst, md)
    print("  L%-3s max|d|=%.6f" % (d["layer"].split(".")[3], md))
print("WORST=%.6f" % worst)
