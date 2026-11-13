"""Training entry point for the sparse MoE translation experiments."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.losses import total_aux_loss
from src.router import RouterConfig, TopKRouter


def build_router(hidden_size: int, cfg: dict) -> TopKRouter:
    return TopKRouter(hidden_size, RouterConfig(**cfg["router"]))


def train_step(model, batch, optimizer, cfg):
    logits, probs, top_idx = model(batch["input_ids"])
    ce = torch.nn.functional.cross_entropy(logits, batch["labels"])
    aux = total_aux_loss(logits, probs, top_idx, cfg["router"]["num_experts"])
    loss = ce + aux
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad"])
    optimizer.step()
    return float(loss.detach()), float(ce.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="runs/moe")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "config.json").write_text(json.dumps(cfg, indent=2))
    print("model configured:", build_router(cfg["model"]["hidden_size"], cfg))


if __name__ == "__main__":
    main()
