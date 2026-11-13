"""Edge-conditioned graph attention network for molecular regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConditionedAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.edge_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def forward(self, node_feats, edge_feats, adjacency):
        q = self.query(node_feats).view(-1, self.num_heads, node_feats.shape[-1] // self.num_heads)
        k = self.key(node_feats).view(-1, self.num_heads, node_feats.shape[-1] // self.num_heads)
        v = self.value(node_feats).view(-1, self.num_heads, node_feats.shape[-1] // self.num_heads)
        scores = torch.einsum("nhd,nhd->nh", q, k).unsqueeze(-1)
        edge_bias = self.edge_proj(edge_feats)
        scores = scores + edge_bias * adjacency
        weights = F.softmax(scores, dim=0)
        out = torch.einsum("nh,nhd->nhd", weights.squeeze(-1), v).reshape(node_feats.shape)
        return out


class GATMol(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeConditionedAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, node_feats, edge_feats, adjacency):
        for layer in self.layers:
            node_feats = layer(node_feats, edge_feats, adjacency)
        return self.head(node_feats.mean(dim=0)).squeeze(-1)
