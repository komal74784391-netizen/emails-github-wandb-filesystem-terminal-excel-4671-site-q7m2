"""Training loop for molecular property regression."""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.model import GATMol


def evaluate(model, loader):
    losses = []
    model.eval()
    with torch.no_grad():
        for nodes, edges, adj, target in loader:
            pred = model(nodes, edges, adj)
            losses.append(float(F.mse_loss(pred, target)))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = GATMol(cfg["hidden_size"], cfg["num_layers"])
    print("trainable parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
