"""Triton kernel for the tiled attention path."""

import torch


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def tiled_attention_forward(q, k, v, tile_size=256):
    if not _has_triton():
        raise RuntimeError("triton is required for the fused kernel path")
    import triton
    import triton.language as tl

    BLOCK = tl.constexpr(tile_size)

    @triton.jit
    def _kernel(Q, K, V, Out, stride_q, stride_k, stride_v, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        q_blk = tl.load(Q + offs[:, None] * stride_q)
        k_mat = tl.load(K + offs[:, None] * stride_k)
        scores = tl.dot(q_blk, tl.trans(k_mat))
        probs = tl.softmax(scores)
        v_mat = tl.load(V + offs[:, None] * stride_v)
        tl.store(Out + offs[:, None] * stride_q, tl.dot(probs, v_mat))

    Out = torch.empty_like(q)
    _kernel[(q.shape[-2] // tile_size,)](q, k, v, Out, q.stride(-2), k.stride(-2), v.stride(-2), BLOCK=tile_size)
    return Out
