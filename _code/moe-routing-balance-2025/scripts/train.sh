#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/base.yaml --output-dir runs/moe-base
