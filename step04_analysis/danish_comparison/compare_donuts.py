#!/usr/bin/env python3
"""Display a specified simulated/real CCS image pair with the same grayscale rules."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from step04_analysis.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=Path(__file__).with_name("donut_example.npz"))
    parser.add_argument("--sim-key", default="simulation_original_200")
    parser.add_argument("--real-key", default="real_original_200")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/danish_comparison/donuts")
    args = parser.parse_args()
    with np.load(args.arrays, allow_pickle=False) as data:
        images = [np.asarray(data[key], float) for key in (args.sim_key, args.real_key)]
    if any(im.ndim != 2 or not np.isfinite(im).all() for im in images):
        raise ValueError("Select finite 2-D CCS images")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), layout="constrained")
    lo, hi = np.quantile(np.concatenate([im.ravel() for im in images]), [.003, .997])
    for col, (image, name) in enumerate(zip(images, ("imSim", "Real"))):
        axes[0, col].imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        if image.std() == 0:
            raise ValueError("Constant image")
        scaled = (image - image.mean()) / image.std()
        axes[1, col].imshow(scaled, origin="lower", cmap="gray", vmin=-1, vmax=4)
        axes[0, col].set_title(name)
        for row in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Shared linear scale")
    axes[1, 0].set_ylabel("Per-image z-score")
    fig.savefig(args.output_dir / "sim_real_grayscale.png", dpi=180)
    fig.savefig(args.output_dir / "sim_real_grayscale.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
