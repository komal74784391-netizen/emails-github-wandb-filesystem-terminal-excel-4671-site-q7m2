"""Temperature scaling and ensemble calibration."""

import torch
import torch.nn.functional as F


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        return logits / torch.exp(self.log_temperature)


def expected_calibration_error(probs, targets, bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)
    ece = torch.zeros((), device=probs.device)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += mask.float().mean() * (acc - conf).abs()
    return float(ece)
