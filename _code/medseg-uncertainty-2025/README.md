# Calibrated Uncertainty Estimation for Medical Image Segmentation

Reference implementation for the manuscript **"Calibrated Uncertainty
Estimation for Medical Image Segmentation"** by D. Castillo, M. Sokolova and
K. Petrov.

Project status: Released (v0.4, 2025-09-10).

The calibration head and the evaluation scripts are public so that the
clinical review group can inspect the temperature scaling procedure.

## Reproduction

The reference training run for the manuscript is tracked at:

https://wandb.ai/buju713851-uci/emails-github-wandb-filesystem-terminal-excel-4671-releases-q7m2/runs/glncdu0v

## Layout

- `src/unet.py` - segmentation backbone
- `src/calibration.py` - temperature scaling and ensemble head
- `src/train.py` - training and ECE evaluation
