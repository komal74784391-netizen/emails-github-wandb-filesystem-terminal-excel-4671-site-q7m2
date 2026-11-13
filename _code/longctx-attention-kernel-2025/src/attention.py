"""Tiled attention wrapper for long-context inference."""

import torch
import torch.nn.functional as F


class TiledAttention(torch.nn.Module):
    def __init__(self, tile_size: int = 256):
        super().__init__()
        self.tile_size = tile_size

    def forward(self, q, k, v, mask=None):
        scale = q.shape[-1] ** -0.5
        out = torch.empty_like(q)
        for start in range(0, q.shape[-2], self.tile_size):
            end = min(start + self.tile_size, q.shape[-2])
            qt = q[..., start:end, :]
            scores = torch.matmul(qt, k.transpose(-1, -2)) * scale
            if mask is not None:
                scores = scores.masked_fill(mask[..., start:end, :], float("-inf"))
            out[..., start:end, :] = torch.matmul(F.softmax(scores, dim=-1), v)
        return out
