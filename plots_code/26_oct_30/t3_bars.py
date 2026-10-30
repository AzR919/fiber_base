"""Oct 30 presentation: bar charts of the mixture test (T3), mixture model vs upper bound, sharp vs broad marks.

Reads both models' eval outputs (eval_summary.json and the paired .npz of the GM12878 / HepG2 pair) and draws:
  t3_summary_bars.png    differential r (per 5 kb window), sign accuracy and fiber-origin AUC, averaged over the
                         sharp assays, the broad assays and all 8; H0 ("bulk on every molecule") as a dashed line
  t3_per_assay_bars.png  the same three readouts per assay
  t3_library_bars.png    mixture model: fiber-origin |AUC - 0.5| for GM12878 vs HepG2 and for the library control
                         (HepG2 100U vs 200U), per assay
Window-level differential r is recomputed from the .npz the way plot_differential_scatter does (windows scored in
both cell types). Sharp / broad as in reports/Oct_30.md.

    python plots_code/26_oct_30/t3_bars.py [--mixture ...] [--upper ...] [--out_dir ...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIR = "eval24_pair_GM12878_HepG2_200U"
CONTROL = "eval25_pair_HepG2_100U_200U"
SHARP = ["atac", "h3k4me3", "h3k27ac", "ctcf"]
BROAD = ["h3k4me1", "h3k27me3", "h3k36me3", "h3k9me3"]
ASSAY_LABELS = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3",
                "h3k4me1": "H3K4me1", "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
READOUTS = [("diff_r", "Differential r (per 5 kb window)", 0.0),
            ("sign", "Sign accuracy", 0.5),
            ("auc", "Fiber-origin AUC", 0.5)]

MIX_COLOR = "#2a78d6"
UPPER_COLOR = "#eb6834"
CONTROL_COLOR = "#b7b5ad"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def load(run_dir):
    """{readout: {assay: value}} for one model, plus the library control's |AUC - 0.5|."""
    res = json.load(open(os.path.join(run_dir, "eval_summary.json")))["results"]
    loci = res[f"{PAIR}/paired"]["loci"]
    z = np.load(os.path.join(run_dir, f"{PAIR}_paired.npz"))
    diff_r = {}
    for k, a in enumerate(z["assays"]):
        ok = z["tile_n_win"][:, k] > 0
        x, y = z["tile_dt_win"][ok, k], z["tile_dp_win"][ok, k]
        diff_r[str(a)] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    return {"diff_r": diff_r,
            "sign": {a: v["sign_accuracy"] for a, v in loci.items()},
            "auc": {a: v["fiber_auc"] for a, v in loci.items()},
            "auc_dev": {a: v["fiber_auc_abs_dev"] for a, v in loci.items()},
            "lib_dev": {a: v["fiber_auc_abs_dev"] for a, v in res[f"{CONTROL}/paired"]["loci"].items()}}


def mean(d, assays):
    return float(np.mean([d[a] for a in assays]))


def style(ax, h0=None):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if h0 is not None:
        ax.axhline(h0, color=TEXT, ls="--", lw=1.4, zorder=4)


def paired_bars(ax, x, a, b, w, labels):
    ax.bar(x - w / 2 - 0.02, a, w, color=MIX_COLOR, label=labels[0], zorder=3)
    ax.bar(x + w / 2 + 0.02, b, w, color=UPPER_COLOR, label=labels[1], zorder=3)


def summary(mix, upper, out):
    groups = [("Sharp marks", SHARP), ("Broad marks", BROAD), ("All 8", SHARP + BROAD)]
    x = np.array([0.0, 1.0, 2.25])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, h0) in zip(axes, READOUTS):
        m = [mean(mix[key], g) for _, g in groups]
        u = [mean(upper[key], g) for _, g in groups]
        paired_bars(ax, x, m, u, 0.36, ("Mixture model", "Upper bound (own tracks)"))
        for xi, v in zip(x - 0.2, m):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=TEXT)
        for xi, v in zip(x + 0.2, u):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=TEXT_2)
        style(ax, h0)
        ax.set_ylim(0, 1.0 if h0 else 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in groups], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, loc="left")
        ax.text(0, 1.0, f"H0 = {h0:g}", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=TEXT_2)
        ax.set_title(title + "\n", fontsize=14, fontweight="bold", color=TEXT, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=TEXT, ls="--", lw=1.4))
    labels.append("H0: bulk on every molecule")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.1, wspace=0.22)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def assay_axis(ax):
    """x positions of the 8 assays (sharp, gap, broad) with group labels under the ticks."""
    x = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6], dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([ASSAY_LABELS[a] for a in SHARP + BROAD], fontsize=11)
    for lo, hi, txt in ((x[0], x[3], "Sharp"), (x[4], x[7], "Broad")):
        ax.annotate("", xy=(lo - 0.4, -0.13), xytext=(hi + 0.4, -0.13), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))
        ax.text((lo + hi) / 2, -0.16, txt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12,
                color=TEXT_2)
    return x


def per_assay(mix, upper, out):
    assays = SHARP + BROAD
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    for ax, (key, title, h0) in zip(axes, READOUTS):
        x = assay_axis(ax)
        paired_bars(ax, x, [mix[key][a] for a in assays], [upper[key][a] for a in assays], 0.36,
                    ("Mixture model", "Upper bound (own tracks)"))
        style(ax, h0)
        ax.set_ylim(0, 1.0 if h0 else 0.85)
        ax.set_title(f"{title}   (H0 = {h0:g})", fontsize=14, fontweight="bold", color=TEXT, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=TEXT, ls="--", lw=1.4))
    labels.append("H0: bulk on every molecule")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.06, hspace=0.55)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def library(mix, out):
    assays = SHARP + BROAD
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = assay_axis(ax)
    w = 0.36
    ax.bar(x - w / 2 - 0.02, [mix["auc_dev"][a] for a in assays], w, color=MIX_COLOR,
           label="GM12878 vs HepG2 (different cell types)", zorder=3)
    ax.bar(x + w / 2 + 0.02, [mix["lib_dev"][a] for a in assays], w, color=CONTROL_COLOR,
           label="HepG2 100U vs 200U (same cell type, two libraries)", zorder=3)
    style(ax)
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("|AUC − 0.5|  (fiber origin)")
    ax.set_title("Mixture model: fibers separated by cell type, not by library\n", fontsize=14, fontweight="bold",
                 color=TEXT, loc="left")
    ax.text(0, 1.0, f"Mean over assays: {mean(mix['auc_dev'], assays):.3f} vs {mean(mix['lib_dev'], assays):.3f} "
                    "(sampling noise alone ≈ 0.05)", transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            color=TEXT_2)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.2)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", default="results/26-10-21_T02-06-54_T3_mix_data27_mix_GM12878_HepG2_200U_model15_"
                                         "uct_mid_8A_train02_eval")
    ap.add_argument("--upper", default="results/26-10-21_T02-04-44_T3_upper_data28_GM12878_HepG2_200U_model15_"
                                       "uct_mid_8A_train02_eval")
    ap.add_argument("--out_dir", default="plots/26_oct_30/t3")
    args = ap.parse_args()

    mix, upper = load(args.mixture), load(args.upper)
    for key, _, _ in READOUTS:
        print(key, "mixture", {a: round(v, 3) for a, v in mix[key].items()})
        print(key, "upper  ", {a: round(v, 3) for a, v in upper[key].items()})
    os.makedirs(args.out_dir, exist_ok=True)
    summary(mix, upper, os.path.join(args.out_dir, "t3_summary_bars.png"))
    per_assay(mix, upper, os.path.join(args.out_dir, "t3_per_assay_bars.png"))
    library(mix, os.path.join(args.out_dir, "t3_library_bars.png"))


if __name__ == "__main__":
    main()
