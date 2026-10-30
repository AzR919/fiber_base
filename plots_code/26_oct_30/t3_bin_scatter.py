"""Oct 30 presentation: bin-level differential scatter of the mixture test (T3), next to the window-level one.

The window-level scatter (vis_utils.plot_differential_scatter, copied to plots/26_oct_30/t3/t3_mix_differential_
scatter.png) places every chr21 tile at its 5 kb mean difference, but colours it by its strongest 1 kb bin, so a
strong but narrow difference is diluted towards the origin. This plot uses the flagged 1 kb bin itself: one dot per
differential tile (the bins sign accuracy is scored on), x = true difference GM12878 - HepG2 over the bin's scored
positions, y = predicted difference (each cell type's prediction from its own fibers), coloured by the cell type
with the higher track. Dashed: y = x. Titles: Pearson r over the bins and sign accuracy (matches the eval summary).
x is the mean of the per-bp asinh differences, while the bins were flagged on asinh(mean FC) (build_loci.py), so
narrow peaks just over the 0.5 threshold can land inside +-0.5 (29% of CTCF bins, 2% of H3K27ac).

    python plots_code/26_oct_30/t3_bin_scatter.py [--run_dir ...] [--out ...]
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from vis_utils import PAIR_COLORS, FS_LABEL, FS_TICK, FS_TITLE

PAIR = "eval24_pair_GM12878_HepG2_200U"
CELLS = ["GM12878", "HepG2_200U"]


def plot_bin_scatter(z, out):
    assays = [str(a) for a in z["assays"]]
    K = len(assays)
    ncol = min(4, K)
    nrow = int(math.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    for k, a in enumerate(assays):
        ax = axes[k // ncol][k % ncol]
        d = z["tile_d"][:, k]
        has = (d != 0) & (z["tile_n_bin"][:, k] > 0)
        x, y, d = z["tile_dt_bin"][has, k], z["tile_dp_bin"][has, k], d[has]
        for sign, c, name in ((1, PAIR_COLORS[0], f"higher in {CELLS[0]}"), (-1, PAIR_COLORS[1], f"higher in {CELLS[1]}")):
            m = np.sign(d) == sign
            ax.scatter(x[m], y[m], s=6, color=c, alpha=0.6, lw=0, label=name)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="0.5", ls="--", lw=0.8, label="y = x")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        sign_acc = np.mean(np.sign(y) == np.sign(d))
        ax.set_title(f"{a}  |  r = {r:.2f}, sign {sign_acc:.2f}  (n = {len(x):,})", fontsize=FS_LABEL - 1)
        ax.set_xlabel(f"target {CELLS[0]} - {CELLS[1]}", fontsize=FS_TICK)
        ax.set_ylabel("predicted difference", fontsize=FS_TICK)
        if k == 0:
            ax.legend(fontsize=FS_TICK - 3, frameon=False, markerscale=2)
        print(f"{a:9s} n {len(x):5d}  r {r:.3f}  sign {sign_acc:.3f}")
    for k in range(K, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Bin-level differences at differential 1 kb bins: {CELLS[0]} vs {CELLS[1]} (each from its own fibers)",
                 fontsize=FS_TITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/26-10-21_T02-06-54_T3_mix_data27_mix_GM12878_HepG2_200U_model15_"
                                         "uct_mid_8A_train02_eval")
    ap.add_argument("--out", default="plots/26_oct_30/t3/t3_mix_differential_scatter_bin.png")
    args = ap.parse_args()
    z = np.load(os.path.join(args.run_dir, f"{PAIR}_paired.npz"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_bin_scatter(z, args.out)


if __name__ == "__main__":
    main()
