import torch, numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path("/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/tests")))
from test_moe_prefill_int4 import quantize_int4, dequantize_int4, to_ipex_kmajor, GROUP_SIZE

N, K, GS = 16, 128, 128
torch.manual_seed(0)
W = torch.randn(N, K, dtype=torch.float16) * 0.5
qw_nat, sc_nat = quantize_int4(W, GS)

qw_s = qw_nat.unsqueeze(0); sc_s = sc_nat.unsqueeze(0)
qw_km, sc_km = to_ipex_kmajor(qw_s, sc_s)

SHUFFLE = [0, 4, 1, 5, 2, 6, 3, 7]
qw_m = qw_km[0].numpy().view(np.uint32)
qw_o = qw_nat.numpy().view(np.uint32)

errors = 0
for n in range(N):
    for kp in range(K // 8):
        for b in range(8):
            nat_nib = int((qw_o[n, kp] >> (b * 4)) & 0xF)
            mar_nib = int((qw_m[kp, n] >> (SHUFFLE[b] * 4)) & 0xF)
            if nat_nib != mar_nib:
                errors += 1
                if errors <= 5:
                    print(f"MISMATCH n={n} kp={kp} b={b}: nat={nat_nib} mar={mar_nib}")
print(f"Nibble check: {errors} errors / {N*K}")

# Also check: XeTLA expects signed int4 via implement_zp.
# IPEX marlin stores unsigned 0-15 (dequant: nibble-8).
# implement_zp converts unsigned u4 → signed s4 representation.
# So the flow should be: marlin(unsigned) → unshuffle → repack uint8 → implement_zp → signed
# Let's verify implement_zp on known values
from vllm_xpu_kernels.fused_moe_interface import implement_zp

# Value 8 (unsigned) → 0 (signed), value 0 → -8, value 15 → +7
test = torch.tensor([0x80], dtype=torch.uint8)  # high=8, low=0
result = implement_zp(test)
print(f"implement_zp(0x80)=0x{result[0].item():02x} (expect high=0,low=-8)")

test2 = torch.tensor([0xF7], dtype=torch.uint8)  # high=15, low=7
result2 = implement_zp(test2)
print(f"implement_zp(0xF7)=0x{result2[0].item():02x} (expect high=+7,low=-1)")
