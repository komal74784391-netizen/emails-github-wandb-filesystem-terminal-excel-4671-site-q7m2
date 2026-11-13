"""Routing diagnostics for mixture-of-experts models."""

import numpy as np


def routing_entropy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return float(-(probs * np.log(probs + 1e-12)).sum(axis=-1).mean())


def expert_utilisation(assignments, num_experts):
    counts = np.bincount(np.asarray(assignments).ravel(), minlength=num_experts)
    return counts / max(counts.sum(), 1)


def overflow_rate(assignments, capacity, num_experts):
    counts = np.bincount(np.asarray(assignments).ravel(), minlength=num_experts)
    return float(np.clip(counts - capacity, 0, None).sum() / max(counts.sum(), 1))
