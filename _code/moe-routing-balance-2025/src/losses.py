"""Auxiliary losses for mixture-of-experts training."""

import torch
import torch.nn.functional as F


def load_balancing_loss(probs: torch.Tensor, top_idx: torch.Tensor, num_experts: int):
    """Switch-style load balancing: importance times load, scaled by experts^2."""
    importance = probs.mean(dim=0)
    load = torch.zeros(num_experts, device=probs.device)
    flat = top_idx.reshape(-1)
    load.scatter_add_(0, flat, torch.ones_like(flat, dtype=load.dtype))
    load = load / flat.numel()
    return num_experts * torch.sum(importance * load)


def router_z_loss(logits: torch.Tensor):
    """Penalise large router logits to keep routing decisions stable."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def total_aux_loss(logits, probs, top_idx, num_experts, w_balance=0.01, w_z=0.001):
    return w_balance * load_balancing_loss(probs, top_idx, num_experts) + w_z * router_z_loss(logits)
