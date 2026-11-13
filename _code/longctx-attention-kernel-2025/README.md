# A Memory-Efficient Attention Kernel for Long-Context Transformers

Reference implementation for the USENIX ATC 2025 paper **"A Memory-Efficient
Attention Kernel for Long-Context Transformers"** by D. Castillo, Y. Fujimoto
and A. Boateng.

Project status: Released (v0.9.2).

The kernel and the benchmark harness are public. The benchmark suite reports
throughput and peak memory against the reference PyTorch attention.

## Reproduction

The reference training run for the paper is tracked at:

https://wandb.ai/buju713851-uci/emails-github-wandb-filesystem-terminal-excel-4671-releases-q7m2/runs/um3691ax

## Layout

- `src/attention.py` - tiled attention wrapper
- `src/triton_kernel.py` - kernel implementation
- `benchmarks/bench_longctx.py` - long-context benchmark
- `tests/test_parity.py` - numerical parity tests
