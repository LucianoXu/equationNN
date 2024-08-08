import torch

from model import *

def test_RoPE_H():
    freqs = precompute_theta_pos_frequencies(6, 3)
    x = torch.randn(1, 3, 2, 6)
    indices = [(0, 0), (0, 2)]
    r2 = apply_rotary_embeddings_H(x, freqs[2], indices)
    r1r1 = apply_rotary_embeddings_H(
        apply_rotary_embeddings_H(x, freqs[1], indices), 
        freqs[1], indices)
    assert torch.allclose(r2, r1r1)

def test_RoPE_B():
    freqs = precompute_theta_pos_frequencies(6, 3)
    x = torch.randn(1, 3, 2, 6)
    indices = [(0, 0), (0, 2)]
    r2 = apply_rotary_embeddings_B(x, freqs[2], indices)
    r1r1 = apply_rotary_embeddings_B(
        apply_rotary_embeddings_B(x, freqs[1], indices), 
        freqs[1], indices)
    assert torch.allclose(r2, r1r1)