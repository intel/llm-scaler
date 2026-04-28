"""Verify K-major marlin int32 → N-major uint8 conversion."""
import torch
import numpy as np

E, N, K, GS = 2, 16, 64, 32
SHUFFLE = [0, 4, 1, 5, 2, 6, 3, 7]

torch.manual_seed(0)
ref_nibbles = torch.randint(0, 16, (E, N, K), dtype=torch.int32)

# Reference N-major uint8 [E, N, K/2]
ref_uint8 = torch.zeros(E, N, K // 2, dtype=torch.uint8)
for k in range(0, K, 2):
    ref_uint8[:, :, k // 2] = (ref_nibbles[:, :, k] | (ref_nibbles[:, :, k + 1] << 4)).to(torch.uint8)

# Build K-major marlin int32 [E, K/8, N]
marlin_int32 = torch.zeros(E, K // 8, N, dtype=torch.int32)
for kp in range(K // 8):
    for b in range(8):
        k_natural = kp * 8 + b
        slot = SHUFFLE[b]
        marlin_int32[:, kp, :] |= (ref_nibbles[:, :, k_natural] << (slot * 4))

# Convert back: un-shuffle + transpose + repack
def marlin_to_nmajor_uint8(qw_marlin, E, N, K):
    """[E, K/8, N] int32 marlin-shuffled → [E, N, K/2] uint8 natural order."""
    SHUFFLE = [0, 4, 1, 5, 2, 6, 3, 7]
    qw = qw_marlin.to(torch.int64) & 0xFFFFFFFF
    unshuffled = torch.zeros_like(qw)
    for new_pos in range(8):
        old_pos = SHUFFLE[new_pos]
        nibble = (qw >> (old_pos * 4)) & 0xF
        unshuffled |= nibble << (new_pos * 4)
    unshuffled = (unshuffled & 0xFFFFFFFF).to(torch.int32)
    # transpose [E, K/8, N] → [E, N, K/8]
    transposed = unshuffled.permute(0, 2, 1).contiguous()
    # repack int32 (8 nibbles) → uint8 (2 nibbles)
    result = torch.zeros(E, N, K // 2, dtype=torch.uint8, device=qw_marlin.device)
    for kp in range(K // 8):
        for b in range(0, 8, 2):
            k_pair = kp * 4 + b // 2
            lo = (transposed[:, :, kp] >> (b * 4)) & 0xF
            hi = (transposed[:, :, kp] >> ((b + 1) * 4)) & 0xF
            result[:, :, k_pair] = (lo | (hi << 4)).to(torch.uint8)
    return result

result = marlin_to_nmajor_uint8(marlin_int32, E, N, K)
match = (result == ref_uint8).all().item()
print(f"Roundtrip: {'PASS' if match else 'FAIL'}")

# Scale: [E, K/GS, N] fp16 → [E, N, K/GS] fp16 (just transpose)
scale_km = torch.randn(E, K // GS, N, dtype=torch.float16)
scale_nm = scale_km.permute(0, 2, 1).contiguous()
print(f"Scale: {scale_km.shape} → {scale_nm.shape}")
