import torch

from src.router import RouterConfig, TopKRouter, assignment_counts


def test_topk_assignment_shape():
    router = TopKRouter(8, RouterConfig(num_experts=4, top_k=2))
    hidden = torch.randn(6, 8)
    top_idx, top_probs, probs, capacity = router(hidden)
    assert top_idx.shape == (6, 2)
    assert top_probs.shape == (6, 2)
    assert probs.shape == (6, 4)
    assert capacity >= 1


def test_assignment_counts_sum():
    idx = torch.tensor([[0, 1], [1, 2], [2, 3]])
    counts = assignment_counts(idx, 4)
    assert float(counts.sum()) == 6.0
