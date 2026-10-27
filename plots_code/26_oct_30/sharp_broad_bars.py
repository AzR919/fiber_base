"""Oct 30 presentation: Pearson r of the dev model vs the cross-cell-average baseline, sharp vs broad marks.

Reads the per-assay "all" rows (model and baseline Pearson) from the dev5 long-run eval log, averages them within
sharp (ATAC, H3K4me3, H3K27ac, CTCF) and broad (H3K4me1, H3K27me3, H3K36me3, H3K9me3) assays per cell type, and
draws one panel per group. H1 / H9 have no ATAC, so their sharp mean is over 3 assays.

    python plots_code/26_oct_30/sharp_broad_bars.py [--log ...] [--out ...]
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k27me3", "h3k4me1", "h3k36me3", "h3k9me3", "ctcf"]
GROUPS = {
    "Sharp marks": ["atac", "h3k4me3", "h3k27ac", "ctcf"],
    "Broad marks": ["h3k4me1", "h3k27me3", "h3k36me3", "h3k9me3"],
}
GROUP_NOTE = {
    "Sharp marks": "ATAC, H3K4me3, H3K27ac, CTCF",
    "Broad marks": "H3K4me1, H3K27me3, H3K36me3, H3K9me3",
}
DEV = ["GM12878", "K562", "HepG2", "Panc1", "H1"]
HELD_OUT = [("H9", "near"), ("HCT116", "medium"), ("MCF-7", "far")]

MODEL_COLOR = "#2a78d6"
BASELINE_COLOR = "#b7b5ad"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def read_log(path):
    """{cell: {"model"|"base": {assay: r or None}}} from the 'all' rows of the eval log."""
    out, cell, kind = {}, None, None
    for line in open(path):
        m = re.search(r"Cell type: (\S+)", line)
        if m:
            cell = m.group(1).replace("_200U", "").replace("MCF_7", "MCF-7")
        if "Pearson r per base pair" in line:
            kind = "model"
        elif "BASELINE (cross" in line:
            kind = "base"
        elif "cell-specific r:" in line:
            kind = None
        parts = line.split()
        if kind and parts and parts[0] == "all":
            out.setdefault(cell, {})[kind] = {
                a: (None if v == "-" else float(v)) for a, v in zip(ASSAYS, parts[2:2 + len(ASSAYS)])}
            kind = None
    return out


def group_mean(vals, assays):
    xs = [vals[a] for a in assays if vals[a] is not None]
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/2026-10-18_04_dev5_long_eval_baseline.out")
    ap.add_argument("--out", default="reports/figures/oct30/sharp_broad_pearson.png")
    args = ap.parse_args()

    res = read_log(args.log)
    cells = DEV + [c for c, _ in HELD_OUT]
    labels = DEV + [f"{c}\n({tag})" for c, tag in HELD_OUT] + ["Mean"]
    # x positions: dev block, gap, held-out block, gap, mean
    step = 1.25
    x = step * np.array(list(range(5)) + [5.6 + i for i in range(3)] + [9.2])

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), sharey=True)
    w = 0.46
    for ax, (name, assays) in zip(axes, GROUPS.items()):
        model = [group_mean(res[c]["model"], assays) for c in cells]
        base = [group_mean(res[c]["base"], assays) for c in cells]
        model.append(np.mean(model))
        base.append(np.mean(base))

        ax.bar(x - w / 2 - 0.02, model, w, color=MODEL_COLOR, label="Model", zorder=3)
        ax.bar(x + w / 2 + 0.02, base, w, color=BASELINE_COLOR, label="Baseline (cross-cell average)", zorder=3)
        # direct label on the mean only: model vs baseline, centred over the pair
        ax.text(x[-1], max(model[-1], base[-1]) + 0.015, f"{model[-1]:.2f} vs {base[-1]:.2f}", ha="center",
                va="bottom", fontsize=11, color=TEXT)

        ax.set_title(name, fontsize=15, fontweight="bold", color=TEXT, loc="left")
        ax.text(0, 1.03, GROUP_NOTE[name], transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
                color=TEXT_2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_ylim(0, 0.7)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # block labels under the tick labels
        for lo, hi, txt in ((x[0], x[4], "Dev"), (x[5], x[7], "Held out")):
            ax.annotate("", xy=(lo - 0.5, -0.2), xytext=(hi + 0.5, -0.2), xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=1))
            ax.text((lo + hi) / 2, -0.23, txt, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=12, color=TEXT_2)
    axes[0].set_ylabel("Pearson r (mean over assays)")
    axes[0].set_title("Sharp marks\n", fontsize=15, fontweight="bold", color=TEXT, loc="left")
    axes[1].set_title("Broad marks\n", fontsize=15, fontweight="bold", color=TEXT, loc="left")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.8, bottom=0.25, wspace=0.07)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
