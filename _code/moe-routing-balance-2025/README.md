# Sparse Mixture-of-Experts Routing with Load-Balancing Regularization

Reference implementation for the EMNLP 2025 paper **"Sparse Mixture-of-Experts
Routing with Load-Balancing Regularization"** by D. Castillo, R. Mensah and
T. Ikeda.

Project status: Released (v1.3, 2025-06-30).

The release contains the router, the auxiliary load-balancing loss and the
training entry point used to produce the numbers in the paper, together with
the configuration files for the multilingual translation experiments.

## Reproduction

The reference training run for the paper is tracked at:

https://wandb.ai/buju713851-uci/emails-github-wandb-filesystem-terminal-excel-4671-releases-q7m2/runs/i656uzqs

## Layout

- `src/router.py` - top-k routed dispatch with capacity factors
- `src/losses.py` - load-balancing and router z-loss terms
- `src/train.py` - training loop and evaluation hooks
- `configs/base.yaml` - base hyper-parameters for the multilingual setup
- `scripts/train.sh` - launcher used for the paper runs
- `tests/test_router.py` - unit tests for dispatch and balancing

## License

MIT.
