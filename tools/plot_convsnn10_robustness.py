#!/usr/bin/env python3
"""Plot measured evaluation rows only; no smoothing or interpolated samples."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/convsnn10_robustness"))
    args = parser.parse_args()
    with (args.results / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = [float(r["sigma"]) * 100 for r in rows]
    y = [float(r["accuracy_mean"]) * 100 for r in rows]
    sd = [float(r["accuracy_std"]) * 100 for r in rows]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        ax.errorbar(x, y, yerr=sd, fmt="o-", capsize=4, color="#2166ac", linewidth=1.6,
                    label="Frozen hardware-aware ConvSNN Top10")
        ax.set(xlabel="Conductance noise SD / (Gmax − Gmin) (%)",
               ylabel="Frozen-test accuracy (%)", ylim=(0, 100), xticks=x)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Static independent Gaussian conductance variation")
        fig.get_layout_engine().set(rect=(0, .07, 1, .93))
        fig.text(.5, .02, "4,894 images · 5 noise seeds per nonzero level · mean ± sample SD\n"
                 "One trained model; clipped G+/G−; simulation, not chip measurements", ha="center", fontsize=9)
        fig.savefig(args.results / "accuracy_vs_conductance_noise.png", dpi=220)
        fig.savefig(args.results / "accuracy_vs_conductance_noise.svg")
        svg = args.results / "accuracy_vs_conductance_noise.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)


if __name__ == "__main__":
    main()
