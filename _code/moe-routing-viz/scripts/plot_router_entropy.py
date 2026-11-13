"""Plot routing entropy curves from an exported statistics file."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats")
    args = parser.parse_args()
    data = json.load(open(args.stats))
    xs = np.arange(len(data["entropy"]))
    plt.plot(xs, data["entropy"])
    plt.xlabel("step")
    plt.ylabel("router entropy")
    plt.tight_layout()
    plt.savefig("router_entropy.png")


if __name__ == "__main__":
    main()
