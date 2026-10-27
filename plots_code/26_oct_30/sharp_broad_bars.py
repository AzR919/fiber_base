"""Oct 30 presentation: the dev model vs the cross-cell-average baseline, sharp vs broad marks.

Reads per-assay Pearson, MSE and cell-specific r (partial correlation given the baseline, all scored chr21 bases)
from the dev5 long-run eval summary, averages them within sharp (ATAC, H3K4me3, H3K27ac, CTCF) and broad (H3K4me1,
H3K27me3, H3K36me3, H3K9me3) assays per cell type, and draws one figure per metric with one panel per group.
H1 / H9 have no ATAC, so their sharp mean is over 3 assays. The cell-type r figure has no baseline bars: the
baseline scores exactly 0 by construction.

    python plots_code/26_oct_30/sharp_broad_bars.py [--summary ...] [--out_dir ...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = {
    "Sharp marks": ["atac", "h3k4me3", "h3k27ac", "ctcf"],
    "Broad marks": ["h3k4me1", "h3k27me3", "h3k36me3", "h3k9me3"],
}
GROUP_NOTE = {
    "Sharp marks": "ATAC, H3K4me3, H3K27ac, CTCF",
    "Broad marks": "H3K4me1, H3K27me3, H3K36me3, H3K9me3",
}
DEV = [("GM12878", "GM12878"), ("K562", "K562_200U"), ("HepG2", "HepG2_200U"), ("Panc1", "Panc1_200U"),
       ("H1", "H1_200U")]
HELD_OUT = [("H9", "H9_200U", "near"), ("HCT116", "HCT116_200U", "medium"), ("MCF-7", "MCF_7_200U", "far")]

# metric: (file suffix, y label, model getter, baseline getter or None, label format)
METRICS = {
    "pearson": ("Pearson r (mean over assays)",
                lambda r, a: r["per_assay_pearson"][a],
                lambda r, a: r["baseline"]["per_assay"][a]["pearson"], "{:.2f}"),
    "mse": ("MSE (mean over assays; lower is better)",
            lambda r, a: r["per_assay_loss"][a],
            lambda r, a: r["baseline"]["per_assay"][a]["loss"], "{:.3f}"),
    "cell_r": ("Cell-type r (mean over assays)",
               lambda r, a: r["partial"]["strata"]["all"]["per_assay"][a]["pearson"],
               None, "{:+.2f}"),
}

MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#b7b5ad"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def group_mean(res, get, assays):
    xs = []
    for a in assays:
        try:
            v = get(res, a)
        except (KeyError, TypeError):
            v = None
        if v is not None:
            xs.append(v)
    return sum(xs) / len(xs)


def plot_metric(results, metric, out_path):
    ylabel, get_model, get_base, fmt = METRICS[metric]
    keys = [k for _, k in DEV] + [k for _, k, _ in HELD_OUT]
    labels = [n for n, _ in DEV] + [f"{n}\n({tag})" for n, _, tag in HELD_OUT] + ["Mean"]
    # x positions: dev block, gap, held-out block, gap, mean
    step = 1.25
    x = step * np.array(list(range(5)) + [5.6 + i for i in range(3)] + [9.2])

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
    paired = get_base is not None
    w = 0.46 if paired else 0.7
    top = 0
    for ax, (name, assays) in zip(axes, GROUPS.items()):
        model = [group_mean(results[k], get_model, assays) for k in keys]
        model.append(np.mean(model))
        if paired:
            base = [group_mean(results[k], get_base, assays) for k in keys]
            base.append(np.mean(base))
            ax.bar(x - w / 2 - 0.02, model, w, color=MODEL_COLOR, label="Model", zorder=3)
            ax.bar(x + w / 2 + 0.02, base, w, color=BASELINE_COLOR, label="Baseline (cross-cell average)",
                   zorder=3)
            hi = max(model[-1], base[-1])
            mean_label = f"{fmt.format(model[-1])} vs {fmt.format(base[-1])}"
            top = max(top, max(model), max(base))
        else:
            ax.bar(x, model, w, color=MODEL_COLOR, zorder=3)
            hi = model[-1]
            mean_label = fmt.format(model[-1])
            top = max(top, max(model))
        ax.text(x[-1], hi, mean_label, ha="center", va="bottom", fontsize=11, color=TEXT)

        ax.set_title(name + "\n", fontsize=15, fontweight="bold", color=TEXT, loc="left")
        ax.text(0, 1.03, GROUP_NOTE[name], transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
                color=TEXT_2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # block labels under the tick labels
        for lo, hi_x, txt in ((x[0], x[4], "Dev"), (x[5], x[7], "Held out")):
            ax.annotate("", xy=(lo - 0.5, -0.2), xytext=(hi_x + 0.5, -0.2), xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=1))
            ax.text((lo + hi_x) / 2, -0.23, txt, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=12, color=TEXT_2)
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel(ylabel)

    if paired:
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="upper center", ncol=2, frameon=False, fontsize=12,
                   bbox_to_anchor=(0.5, 1.0))
    else:
        fig.text(0.5, 0.985, "Baseline scores 0 by definition: bars are signal beyond the cross-cell average",
                 ha="center", va="top", fontsize=12, color=TEXT_2)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.8, bottom=0.25, wspace=0.07)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/26-10-18_T15-31-09_DEV5_long_data20_dev5_200U_model15_uct_mid_8A"
                                         "_train03_eval/eval_summary.json")
    ap.add_argument("--out_dir", default="reports/figures/oct30")
    args = ap.parse_args()

    results = json.load(open(args.summary))["results"]
    os.makedirs(args.out_dir, exist_ok=True)
    for metric in METRICS:
        plot_metric(results, metric, os.path.join(args.out_dir, f"sharp_broad_{metric}.png"))


if __name__ == "__main__":
    main()
