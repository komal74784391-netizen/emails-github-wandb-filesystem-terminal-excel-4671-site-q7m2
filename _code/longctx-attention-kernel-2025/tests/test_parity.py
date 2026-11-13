import torch

from src.attention import TiledAttention


def test_parity_small_sequence():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    tiled = TiledAttention(tile_size=8)(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(tiled, ref, atol=1e-5)
