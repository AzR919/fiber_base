"""ENCODE replicate ceiling (TODO_extended.md I11) for DEV4, its input ablations and the cross-cell baseline.

replicate_ceiling.py scored replicate A against replicate B of each pooled ENCODE target, on the eval positions of
eval.py (chr21 tiles, fiber coverage >= 5, asinh): Pearson r, cell-type r (partial r given the cross-cell
baseline) and MSE. Its summary also holds the ceiling against the pooled track, sqrt(Spearman-Brown(rep r, k)) with
k = replicates pooled, and the MSE floor rep MSE / (2k). Here each model's bulk numbers (ablations.py: bulk_{r,mse,
ctr}.tsv) are put against those: fraction of the r ceiling, of the cell-type r ceiling, and MSE / floor.

    python plots_code/26_nov_13/ceiling.py [--summaries <replicate_summary.json> ...]
--summaries defaults to every results/*_replicates_data24_*/replicate_summary.json (newest wins per cell type).
Needs plots/26_nov_13/ablations/bulk_{r,mse,ctr}.tsv. Writes plots/26_nov_13/ceiling/: ceiling_cells.tsv (one row
per cell type x assay), ceiling_groups.tsv (means per cell-type group, assay and model), ceiling_overview.png,
ceiling_models.png, log_ceiling.txt.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import MODELS, NAMES  # noqa: E402

OUT = "plots/26_nov_13/ceiling"
BULK = "plots/26_nov_13/ablations"
# Assays, cell-type groups and model colours as in ablations.py (which runs on import, so they are repeated here)
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k4me1", "ctcf", "h3k36me3", "h3k27me3", "h3k9me3"]
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
GROUPS = {"dev": ["GM12878", "K562_200U", "HepG2_200U", "H1_200U"],
          "held-out": ["H9_200U", "HCT116_200U", "MCF_7_200U"],
          "Panc1": ["Panc1_200U"]}
COL = {"DEV4": "#4C9BE8", "noCpG": "#D9534F", "noFIRE": "#F0AD4E", "noAcc": "#5CB85C", "m6aCpG": "#8E6CC0",
       "baseline": "#9a9a9a"}
SERIES = list(MODELS) + ["baseline"]
NAME = {**NAMES, "baseline": "cross-cell baseline"}
GROUP_LABEL = {"dev": "Dev (GM12878, K562, HepG2, H1)", "held-out": "Held out (H9, HCT116, MCF-7)",
               "Panc1": "Panc1 (held out)"}
IMBALANCE = 0.15  # |r(A, pooled) - r(B, pooled)| above this: the pooled track is mostly one replicate


def load_summaries(paths):
    """{cell: summary entry} from full-tile evals (mini test configs skipped); the newest file wins per cell type."""
    out = {}
    for p in sorted(paths, key=os.path.getmtime):
        out.update({c: v for c, v in json.load(open(p))["results"].items() if "mini" not in v["eval_config"]})
    return out


def load_bulk():
    """Long table model, cell, assay, r, mse, ctr from ablations.py's outputs."""
    cols = {"r": "bulk_r.tsv", "mse": "bulk_mse.tsv", "ctr": "bulk_ctr.tsv"}
    tabs = [pd.read_csv(os.path.join(BULK, f), sep="\t").set_index(["model", "cell", "assay"])[[c]]
            for c, f in cols.items()]
    return pd.concat(tabs, axis=1).reset_index()


def cell_table(summ, bulk):
    group_of = {c: g for g, cells in GROUPS.items() for c in cells}
    rows = []
    for cell, v in summ.items():
        for a in ASSAYS:
            e = v["per_assay"].get(a)
            if e is None:
                continue
            row = {"cell": cell, "group": group_of[cell], "assay": a, "k_pooled": e["k_pooled"],
                   "rep_r": e["rep_r"], "ceiling_r": e["ceiling_r"], "rep_ct_r": e["rep_ct_r"],
                   "ceiling_ct_r": e["ceiling_ct_r"], "rep_mse": e["rep_mse"], "mse_floor": e["mse_floor"],
                   "a_vs_pooled_r": e["a_vs_pooled_r"], "b_vs_pooled_r": e["b_vs_pooled_r"]}
            row["imbalanced"] = abs(e["a_vs_pooled_r"] - e["b_vs_pooled_r"]) > IMBALANCE
            b = bulk[(bulk.cell == cell) & (bulk.assay == a)].set_index("model")
            for m in SERIES:
                r, mse, ctr = (b.loc[m, c] if m in b.index else np.nan for c in ("r", "mse", "ctr"))
                row[f"{m}_r"], row[f"{m}_frac_r"] = r, r / e["ceiling_r"]
                row[f"{m}_mse"], row[f"{m}_mse_x_floor"] = mse, mse / e["mse_floor"]
                if m != "baseline":
                    row[f"{m}_ctr"], row[f"{m}_frac_ctr"] = ctr, ctr / e["ceiling_ct_r"]
            rows.append(row)
    return pd.DataFrame(rows)


def group_table(cells):
    """Means over the cell types of each group (and of all), per assay and over all assays."""
    keep = [c for c in cells.columns if c not in ("cell", "group", "assay", "imbalanced")]
    out = []
    for g in list(GROUPS) + ["all"]:
        sub = cells if g == "all" else cells[cells.group == g]
        for a in ASSAYS + ["all"]:
            s = sub if a == "all" else sub[sub.assay == a]
            if len(s):
                out.append({"group": g, "assay": a, "n": len(s), **s[keep].mean().to_dict()})
    return pd.DataFrame(out)


def plot_overview(groups):
    """Per assay and cell-type group: replicate r, ceiling, DEV4 and baseline r (all Pearson r, one axis)."""
    series = [("rep_r", "replicate A vs B", "#d5d5d5"), ("ceiling_r", "ceiling (vs pooled)", "#555555"),
              ("DEV4_r", NAME["DEV4"], COL["DEV4"]), ("baseline_r", NAME["baseline"], COL["baseline"])]
    fig, ax = plt.subplots(1, len(GROUPS), figsize=(6.2 * len(GROUPS), 4.2), sharey=True)
    w = 0.8 / len(series)
    for i, g in enumerate(GROUPS):
        t = groups[(groups.group == g) & (groups.assay != "all")].set_index("assay").reindex(ASSAYS)
        x = np.arange(len(ASSAYS))
        for j, (c, lab, col) in enumerate(series):
            ax[i].bar(x + (j - len(series) / 2 + 0.5) * w, t[c], w * 0.9, color=col, label=lab)
        ax[i].set_xticks(x, [LAB[a] for a in ASSAYS], rotation=45, ha="right")
        ax[i].set_title(GROUP_LABEL[g], fontsize=10)
        ax[i].set_ylim(0, 1)
        ax[i].grid(axis="y", color="#e6e6e6", lw=0.6)
        ax[i].set_axisbelow(True)
        for s in ("top", "right"):
            ax[i].spines[s].set_visible(False)
    ax[0].set_ylabel("Pearson r on chr21")
    ax[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Replicate agreement, the ceiling it implies, and DEV4 / baseline against the pooled track",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ceiling_overview.png"), dpi=130)
    plt.close(fig)


def plot_models(groups):
    """Fraction of the r ceiling (top) and of the cell-type r ceiling (bottom) per model, assay and group."""
    rows = [("frac_r", "Pearson r / ceiling", SERIES), ("frac_ctr", "cell-type r / ceiling", list(MODELS))]
    fig, ax = plt.subplots(2, len(GROUPS), figsize=(6.2 * len(GROUPS), 8.4), sharey="row")
    x = np.arange(len(ASSAYS) + 1)
    for r, (key, ylab, series) in enumerate(rows):
        w = 0.84 / len(series)
        for i, g in enumerate(GROUPS):
            t = groups[groups.group == g].set_index("assay").reindex(ASSAYS + ["all"])
            for j, m in enumerate(series):
                ax[r, i].bar(x + (j - len(series) / 2 + 0.5) * w, t[f"{m}_{key}"], w * 0.9, color=COL[m],
                             label=NAME[m])
            ax[r, i].axvline(len(ASSAYS) - 0.5, color="#999999", lw=0.8, ls=":")
            ax[r, i].axhline(0, color="#333333", lw=0.8)
            ax[r, i].set_xticks(x, [LAB[a] for a in ASSAYS] + ["mean"], rotation=45, ha="right")
            ax[r, i].set_ylim(-0.15, 1)
            ax[r, i].grid(axis="y", color="#e6e6e6", lw=0.6)
            ax[r, i].set_axisbelow(True)
            for s in ("top", "right"):
                ax[r, i].spines[s].set_visible(False)
            if r == 0:
                ax[r, i].set_title(GROUP_LABEL[g], fontsize=10)
        ax[r, 0].set_ylabel(ylab)
    ax[0, 0].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.suptitle("How close each model gets to the replicate ceiling (1 = as good as the ENCODE replicates allow)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ceiling_models.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+",
                    default=glob.glob("results/*_replicates_data24_*/replicate_summary.json"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = cell_table(load_summaries(args.summaries), load_bulk())
    groups = group_table(cells)
    cells.to_csv(os.path.join(OUT, "ceiling_cells.tsv"), sep="\t", index=False, float_format="%.4f")
    groups.to_csv(os.path.join(OUT, "ceiling_groups.tsv"), sep="\t", index=False, float_format="%.4f")
    plot_overview(groups)
    plot_models(groups)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    lines = [f"{len(cells)} cell type x assay pairs from {cells.cell.nunique()} cell types"]
    for g in list(GROUPS) + ["all"]:
        t = groups[groups.group == g].set_index("assay")
        lines += [f"\n=== {g}: ceiling and DEV4",
                  t[["n", "rep_r", "ceiling_r", "DEV4_r", "baseline_r", "DEV4_frac_r", "baseline_frac_r",
                     "rep_ct_r", "ceiling_ct_r", "DEV4_ctr", "DEV4_frac_ctr", "mse_floor", "DEV4_mse",
                     "baseline_mse"]].round(3).to_string(),
                  f"--- {g}: fraction of the r ceiling",
                  t[[f"{m}_frac_r" for m in SERIES]].rename(columns=lambda c: c[:-7]).round(2).to_string(),
                  f"--- {g}: fraction of the cell-type r ceiling",
                  t[[f"{m}_frac_ctr" for m in MODELS]].rename(columns=lambda c: c[:-9]).round(2).to_string(),
                  f"--- {g}: MSE / floor",
                  t[[f"{m}_mse_x_floor" for m in SERIES]].rename(columns=lambda c: c[:-12]).round(2).to_string()]
    imb = cells[cells.imbalanced]
    lines += [f"\nimbalanced replicates (|A - B vs pooled| > {IMBALANCE}): {len(imb)} of {len(cells)}",
              imb[["cell", "assay", "k_pooled", "a_vs_pooled_r", "b_vs_pooled_r"]].round(2).to_string(index=False),
              f"\nbaseline r > DEV4 r: {(cells.baseline_r > cells.DEV4_r).sum()} of {len(cells)}",
              f"DEV4 r > replicate A vs B r: {(cells.DEV4_r > cells.rep_r).sum()} of {len(cells)}"]
    text = "\n".join(lines)
    open(os.path.join(OUT, "log_ceiling.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
