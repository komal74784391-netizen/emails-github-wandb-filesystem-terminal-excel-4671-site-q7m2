"""Training and calibration evaluation loop."""

import argparse

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    print("segmentation config:", cfg)


if __name__ == "__main__":
    main()
