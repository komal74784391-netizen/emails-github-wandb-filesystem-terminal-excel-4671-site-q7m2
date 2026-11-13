"""Benchmark harness for long-context attention."""

import time

import torch

from src.attention import TiledAttention


def bench(seq_len: int, dim: int = 64, iters: int = 5):
    q = torch.randn(1, 8, seq_len, dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attn = TiledAttention().cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        out = attn(q, k, v)
    torch.cuda.synchronize()
    return (time.time() - start) / iters, torch.cuda.max_memory_allocated()


if __name__ == "__main__":
    secs, mem = bench(131072)
    print(f"avg {secs:.4f}s, peak {mem / 1e9:.2f} GB")
