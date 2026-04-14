import torch
from omni_xpu_kernel import sdp

def ref(q, k, v):
    qb, kb, vb = [x.permute(0,2,1,3).contiguous() for x in (q,k,v)]
    return torch.nn.functional.scaled_dot_product_attention(qb, kb, vb).permute(0,2,1,3).contiguous()

dev = torch.device("xpu")
print("Padded kv_len to kernel:")
all_ok = True
for kv in [1, 5, 10, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128]:
    for dt in [torch.float16, torch.bfloat16]:
        for dim in [64, 128]:
            q = torch.randn(1, 64, 8, dim, device=dev, dtype=dt)
            k = torch.randn(1, kv, 8, dim, device=dev, dtype=dt)
            v = torch.randn(1, kv, 8, dim, device=dev, dtype=dt)
            out = sdp.sdp(q, k, v)
            r = ref(q, k, v)
            has_nan = (out != out).any().item()
            mx = (out - r).abs().max().item() if not has_nan else float("nan")
            dn = "fp16" if dt == torch.float16 else "bf16"
            tol = 0.05 if dt == torch.bfloat16 else 0.01
            flag = "NaN!" if has_nan else ("FAIL" if mx > tol else "ok")
            if flag != "ok":
                print(f"  kv={kv:4d} d={dim:3d} {dn}: max_diff={mx:.6f} {flag}")
                all_ok = False

if all_ok:
    print("  ALL PASSED!")
else:
    print("  Some failures above")
