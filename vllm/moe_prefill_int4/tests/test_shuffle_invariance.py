"""Prove: marlin shuffle within a K-group doesn't affect GEMM result.

The shuffle [0,4,1,5,2,6,3,7] permutes 8 nibbles within each int32 word.
Since all 8 positions share the same per-group scale (GS=128 >> 8),
the dot product along K is invariant to within-group permutation.

This means: we do NOT need to unshuffle marlin weights before GEMM.
Just use the shuffled weights with K-major layout directly.
"""
import torch

torch.manual_seed(42)
M, N, K, GS = 4, 16, 128, 128

# Random fp16 input and per-group scale
x = torch.randn(M, K, dtype=torch.float16)
scale = torch.randn(N, K // GS, dtype=torch.float16)  # [N, 1] for GS=K

# Random int4 nibbles (0-15), dequant formula: (nibble - 8) * scale
nibbles = torch.randint(0, 16, (N, K), dtype=torch.int32)

# Dequant with natural order
w_natural = ((nibbles.float() - 8) * scale[:, :1].repeat(1, K).float())  # broadcast scale

# Shuffle within each 8-position block
SHUFFLE = [0, 4, 1, 5, 2, 6, 3, 7]
nibbles_shuffled = torch.zeros_like(nibbles)
for block_start in range(0, K, 8):
    for i in range(8):
        nibbles_shuffled[:, block_start + i] = nibbles[:, block_start + SHUFFLE[i]]

# Dequant with shuffled order (same scale — within same group)
w_shuffled = ((nibbles_shuffled.float() - 8) * scale[:, :1].repeat(1, K).float())

# GEMM: x @ w^T
out_natural  = x.float() @ w_natural.t()
out_shuffled = x.float() @ w_shuffled.t()

diff = (out_natural - out_shuffled).abs().max().item()
print(f"GEMM diff (natural vs shuffled): {diff:.2e}")
print(f"Result: {'INVARIANT (no unshuffle needed!)' if diff < 1e-3 else 'NOT invariant'}")

# Also test with GS < K to make sure shuffle stays within groups
GS2 = 32
scale2 = torch.randn(N, K // GS2, dtype=torch.float16)
w_natural2 = torch.zeros(N, K)
w_shuffled2 = torch.zeros(N, K)
for g in range(K // GS2):
    s = scale2[:, g:g+1].float()
    w_natural2[:, g*GS2:(g+1)*GS2] = (nibbles[:, g*GS2:(g+1)*GS2].float() - 8) * s
    w_shuffled2[:, g*GS2:(g+1)*GS2] = (nibbles_shuffled[:, g*GS2:(g+1)*GS2].float() - 8) * s

out2_nat = x.float() @ w_natural2.t()
out2_shf = x.float() @ w_shuffled2.t()
diff2 = (out2_nat - out2_shf).abs().max().item()
print(f"GEMM diff (GS={GS2}): {diff2:.2e}")
print(f"Result: {'INVARIANT' if diff2 < 1e-3 else 'NOT invariant'}")
