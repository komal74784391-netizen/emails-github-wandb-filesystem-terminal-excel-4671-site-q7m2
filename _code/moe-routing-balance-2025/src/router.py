"""Top-k mixture-of-experts routing with capacity control."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterConfig:
    num_experts: int = 16
    top_k: int = 2
    capacity_factor: float = 1.25
    jitter: float = 0.01


class TopKRouter(nn.Module):
    """Routes tokens to the top-k experts and tracks assignment statistics."""

    def __init__(self, hidden_size: int, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(hidden_size, cfg.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor):
        logits = self.gate(hidden_states)
        if self.training and self.cfg.jitter > 0.0:
            logits = logits + torch.randn_like(logits) * self.cfg.jitter
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.cfg.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        capacity = int(math.ceil(
            hidden_states.shape[0] * self.cfg.top_k * self.cfg.capacity_factor
            / self.cfg.num_experts
        ))
        return top_idx, top_probs, probs, capacity


def assignment_counts(top_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Count how many token assignments each expert receives."""
    one_hot = F.one_hot(top_idx.reshape(-1), num_classes=num_experts)
    return one_hot.sum(dim=0).float()
