# Graph Attention Networks for Molecular Property Prediction

Reference implementation for the BCB 2024 paper **"Graph Attention Networks for
Molecular Property Prediction"** by D. Castillo, K. Petrov and L. Osei.

Project status: Released (v2.0, 2025-01-15).

This project folder is the authoritative version maintained by the lab. Several
community mirrors exist; please use this project folder when citing the code.

## Reproduction

The reference training run for the paper is tracked at:

https://wandb.ai/buju713851-uci/emails-github-wandb-filesystem-terminal-excel-4671-releases-q7m2/runs/00t3r283

## Layout

- `src/model.py` - edge-conditioned graph attention layers
- `src/featurizer.py` - molecular graph featurisation
- `src/train.py` - training and evaluation pipeline
- `configs/base.yaml` - benchmark configuration
- `scripts/prepare_data.sh` - dataset download helper
