"""Oct 30 presentation: the dev model vs the cross-cell-average baseline per assay, dev vs held-out cell types.

Bar charts of Tables 2-4 in reports/Oct_30.md, from the dev5 long-run eval summary (all scored chr21 bases):
  per_assay_mse.png      MSE, model vs baseline; one panel for the 5 dev cell types, one for the 3 held-out ones
  per_assay_pearson.png  Pearson r, same layout
  per_assay_cell_r.png   cell-type r (partial correlation given the baseline), dev vs held-out in one panel, with
                         each cell type as a dot (the baseline scores exactly 0 by construction, so no baseline bars)
Bars are means over the cell types in the group; "All" is the mean over cell types of their mean over assays (as in
Table 1). The number above each pair is how many cell types the model beats the baseline in. H1 / H9 have no ATAC.

    python plots_code/26_oct_30/per_assay_bars.py [--summary ...] [--out_dir ...]
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSAYS = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]
LABELS = ["ATAC", "H3K4me3", "H3K27ac", "CTCF", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3", "All"]
NARROW, BROAD = (0, 3), (4, 7)                     # index ranges of the narrow- and broad-peak assays
DEV = ["GM12878", "K562_200U", "HepG2_200U", "Panc1_200U", "H1_200U"]
HELD_OUT = ["H9_200U", "HCT116_200U", "MCF_7_200U"]

# metric: (y label, model getter, baseline getter, label format, model better if)
METRICS = {
    "mse": ("MSE (lower is better)",
            lambda r, a: r["per_assay_loss"][a],
            lambda r, a: r["baseline"]["per_assay"][a]["loss"], "{:.3f}", lambda m, b: m < b),
    "pearson": ("Pearson r (higher is better)",
                lambda r, a: r["per_assay_pearson"][a],
                lambda r, a: r["baseline"]["per_assay"][a]["pearson"], "{:.2f}", lambda m, b: m > b),
}
CELL_R = lambda r, a: r["partial"]["strata"]["all"]["per_assay"][a]["pearson"]

MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#b7b5ad"
DEV_COLOR = "#2a78d6"
HELD_COLOR = "#eb6834"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def value(res, get, a):
    try:
        v = get(res, a)
    except (KeyError, TypeError):
        return None
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def per_cell(results, cells, get):
    """{assay: [value per cell type, None where missing]} plus "all": each cell type's mean over its assays."""
    out = {a: [value(results[c], get, a) for c in cells] for a in ASSAYS}
    out["all"] = [np.mean([v for v in (value(results[c], get, a) for a in ASSAYS) if v is not None]) for c in cells]
    return out


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def assay_axis(ax, x):
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    for (lo, hi), txt in ((NARROW, "Narrow peaks"), (BROAD, "Broad peaks")):
        ax.annotate("", xy=(x[lo] - 0.5, -0.2), xytext=(x[hi] + 0.5, -0.2), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))
        ax.text((x[lo] + x[hi]) / 2, -0.22, txt, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color=TEXT_2)


def positions():
    return np.array([i for i in range(8)] + [8.7], dtype=float)   # gap before "All"


def plot_paired(results, metric, out_path):
    ylabel, get_m, get_b, fmt, better = METRICS[metric]
    x = positions()
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6), sharey=True)
    top = 0
    for ax, cells, name in ((axes[0], DEV, "Dev (5 cell types)"), (axes[1], HELD_OUT, "Held out (3 cell types)")):
        m, b = per_cell(results, cells, get_m), per_cell(results, cells, get_b)
        keys = ASSAYS + ["all"]
        mm, bb, wins = [], [], []
        for k in keys:
            pairs = [(p, q) for p, q in zip(m[k], b[k]) if p is not None and q is not None]
            mm.append(np.mean([p for p, _ in pairs]))
            bb.append(np.mean([q for _, q in pairs]))
            wins.append((sum(better(p, q) for p, q in pairs), len(pairs)))
        wins[-1] = (sum(w_[0] for w_ in wins[:-1]), sum(w_[1] for w_ in wins[:-1]))   # "All": pairs over assays
        ax.bar(x - w / 2 - 0.02, mm, w, color=MODEL_COLOR, label="Model", zorder=3)
        ax.bar(x + w / 2 + 0.02, bb, w, color=BASELINE_COLOR, label="Baseline (cross-cell average)", zorder=3)
        for xi, p, q, (k, n) in zip(x, mm, bb, wins):
            ax.annotate(f"{k}/{n}", (xi, max(p, q)), xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9.5, color=TEXT_2)
        ax.annotate(f"{fmt.format(mm[-1])} vs {fmt.format(bb[-1])}", (x[-1], max(mm[-1], bb[-1])), xytext=(0, 16),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=TEXT)
        top = max(top, max(mm), max(bb))
        ax.set_title(name, fontsize=15, fontweight="bold", color=TEXT, loc="left")
        assay_axis(ax, x)
        style(ax)
    axes[0].set_ylim(0, top * 1.2)
    axes[0].set_ylabel(ylabel)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.925, "Numbers above the bars: cell types where the model beats the baseline", ha="center",
             va="top", fontsize=11, color=TEXT_2)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.82, bottom=0.22, wspace=0.06)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_cell_r(results, out_path):
    x = positions()
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.6))
    rng = np.random.default_rng(0)
    for cells, off, color, name in ((DEV, -1, DEV_COLOR, "Dev (5 cell types)"),
                                    (HELD_OUT, 1, HELD_COLOR, "Held out (3 cell types)")):
        v = per_cell(results, cells, CELL_R)
        keys = ASSAYS + ["all"]
        means = [np.mean([p for p in v[k] if p is not None]) for k in keys]
        xs = x + off * (w / 2 + 0.02)
        ax.bar(xs, means, w, color=color, alpha=0.85, label=name, zorder=3)
        for xi, k in zip(xs, keys):
            pts = [p for p in v[k] if p is not None]
            ax.scatter(xi + rng.uniform(-0.08, 0.08, len(pts)), pts, s=16, color=TEXT, alpha=0.75, zorder=4,
                       linewidths=0)
        ax.annotate(f"{means[-1]:+.2f}", (xs[-1], max(means[-1], max(v["all"]))), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=TEXT)
    ax.axhline(0, color=TEXT_2, linewidth=1, zorder=2)
    ax.set_ylabel("Cell-type r (higher is better)")
    ax.set_title("Cell-type r per assay", fontsize=15, fontweight="bold", color=TEXT, loc="left")
    assay_axis(ax, x)
    style(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.text(0.5, 0.925, "Bars: mean over cell types; dots: each cell type. The baseline scores 0 by definition.",
             ha="center", va="top", fontsize=11, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.22)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/26-10-18_T15-31-09_DEV5_long_data20_dev5_200U_model15_uct_mid_8A"
                                         "_train03_eval/eval_summary.json")
    ap.add_argument("--out_dir", default="plots/26_oct_30/rest")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    results = json.load(open(args.summary))["results"]
    os.makedirs(args.out_dir, exist_ok=True)
    for metric in METRICS:
        plot_paired(results, metric, os.path.join(args.out_dir, f"per_assay_{metric}.png"))
    plot_cell_r(results, os.path.join(args.out_dir, "per_assay_cell_r.png"))


if __name__ == "__main__":
    main()
