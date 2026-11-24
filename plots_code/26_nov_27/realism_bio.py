"""Mixture realism test, first run (TODO_extended.md section 2b, 23 Nov; GM12878 + HepG2, seed 919, DEV4NF's
channels): the two biology tests on the four models. Reads the per-model outputs that ctcf_footprints.py (hole sites
excluded) and xi_promoters.py wrote into plots/26_nov_27/realism/; nothing is recomputed per fiber.

Models (abl_common keys):
  DEV4NF  noFIRE  trained on the 4 dev cell types' own tracks (reference, has seen both cells)
  R0      mixR0   GM12878 + HepG2 mixtures, share redrawn per window (data27)
  R1a     mixR1a  share fixed at 0.5, stack balanced to it, same target (data36)
  Pair    pairNF  GM12878 and HepG2 own tracks (data28, the ceiling for mixture supervision)

CTCF (Nov_20.md section 1): H1 footprint - open pred CTCF per bound site > 0; H2 the same minus the ATAC
difference > 0; H3 open - nucleosome > 0; H5 bound > unbound at matched motif score; fibers of GM12878 and HepG2.
X inactivation (Nov_20.md section 2): GM12878, silenced genes with a CGI promoter, Xa - Xi in the expected direction
for ATAC, H3K4me3, H3K27ac (+) and H3K9me3, H3K27me3 (-); escape + PAR and chr21 as nulls; XIST reversed.

    python plots_code/26_nov_27/realism_bio.py
Writes plots/26_nov_27/realism/: bio_ctcf.tsv, bio_ctcf_h5.tsv, bio_xi.tsv, bio_xist.tsv, bio_ctcf.png, bio_xi.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "26_nov_13"))
from sm_stats import wilcoxon_p, mannwhitney_greater_p

D = "plots/26_nov_27/realism"
MODELS = {"DEV4NF": "noFIRE", "R0": "mixR0", "R1a": "mixR1a", "Pair": "pairNF"}
COL = dict(zip(MODELS, ["#606060", "#2a78d6", "#eb6834", "#1baf7a"]))
MARKS = {"atac": ("ATAC", 1), "h3k4me3": ("H3K4me3", 1), "h3k27ac": ("H3K27ac", 1),
         "h3k9me3": ("H3K9me3", -1), "h3k27me3": ("H3K27me3", -1)}
SAMPLES = ["GM12878", "HepG2_200U"]


def xi_groups(d):
    sil = d["class"] == "subject"
    return {"silenced_cgi": d[sil & (d.cgi == 1)], "escape_par": d[d["class"].isin(["escape", "par"])],
            "chr21": d[d["class"] == "autosomal"], "xist": d[d["class"] == "xist"]}


def ctcf():
    rows, h5 = [], []
    for m, k in MODELS.items():
        s = pd.read_csv(os.path.join(D, f"ctcf_summary_{k}.tsv"), sep="\t")
        s.insert(0, "model", m)
        rows.append(s[s["sample"].isin(SAMPLES)])
        b = pd.read_csv(os.path.join(D, f"ctcf_bound_vs_unbound_by_score_{k}.tsv"), sep="\t")
        for sample, g in b[b["sample"].isin(SAMPLES)].groupby("sample"):
            h5.append({"model": m, "sample": sample, "bins": len(g),
                       "bins_bound_higher": int((g.bound_mean > g.unbound_mean).sum()),
                       "n_bound": int(g.n_bound.sum()), "n_unbound": int(g.n_unbound.sum()),
                       "mean_bound_minus_unbound": float((g.bound_mean - g.unbound_mean).mean())})
    df, h5 = pd.concat(rows, ignore_index=True), pd.DataFrame(h5)
    df.to_csv(os.path.join(D, "bio_ctcf.tsv"), sep="\t", index=False, float_format="%.4g")
    h5.to_csv(os.path.join(D, "bio_ctcf_h5.tsv"), sep="\t", index=False, float_format="%.4g")
    cols = ["n_sites", "footprint_frac_of_accessible", "pred_ctcf_site_mean", "d_fp_open_ctcf_median",
            "d_fp_open_ctcf_frac_pos", "d_fp_open_ctcf_p", "d_fp_open_ctcf_n", "d_fp_open_atac_median",
            "d_ctcf_minus_atac_median", "d_ctcf_minus_atac_p", "d_open_nuc_ctcf_median", "d_open_nuc_ctcf_p"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        for st in ["bound", "unbound", "bound_other_line"]:
            print(f"\n== CTCF, {st} sites")
            print(df[df.site_set == st].set_index(["sample", "model"])[cols].sort_index(level=0, sort_remaining=False)
                  .to_string(float_format=lambda v: f"{v:.3g}"))
        print("\n== CTCF H5: score bins where bound > unbound")
        print(h5.to_string(index=False, float_format=lambda v: f"{v:.3g}"), flush=True)
    return df


def xi():
    rows, xist = [], []
    for m, k in MODELS.items():
        g = xi_groups(pd.read_csv(os.path.join(D, f"xi_per_gene_{k}.tsv"), sep="\t"))
        for gname in ["silenced_cgi", "escape_par", "chr21"]:
            for a, (lab, e) in MARKS.items():
                v = g[gname][f"pred_{a}"].dropna()
                rows.append({"model": m, "group": gname, "assay": lab, "n_genes": len(v), "median_d": v.median(),
                             "frac_expected": (np.sign(v) == e).mean(), "wilcoxon_p": wilcoxon_p(v),
                             "mwu_p_sil_absd_gt_escape": mannwhitney_greater_p(
                                 g["silenced_cgi"][f"pred_{a}"].abs(), g["escape_par"][f"pred_{a}"].abs())
                             if gname == "silenced_cgi" else np.nan})
        x = g["xist"].iloc[0]
        xist.append({"model": m, **{MARKS[a][0]: x[f"pred_{a}"] for a in MARKS}})
    res, xist = pd.DataFrame(rows), pd.DataFrame(xist)
    res.to_csv(os.path.join(D, "bio_xi.tsv"), sep="\t", index=False, float_format="%.4g")
    xist.to_csv(os.path.join(D, "bio_xist.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for gname in ["silenced_cgi", "escape_par", "chr21"]:
            r = res[res.group == gname]
            for v, f in [("frac_expected", "%.2f"), ("median_d", "%.3f"), ("wilcoxon_p", "%.1e")]:
                print(f"\n== X inactivation, {gname}: {v} (n genes {r.n_genes.iloc[0]})")
                print(r.pivot_table(index="model", columns="assay", values=v, sort=False)
                      .reindex(columns=[l for l, _ in MARKS.values()]).to_string(float_format=f))
        print("\n== H2: |d| silenced CGI > escape + PAR, Mann-Whitney p")
        print(res[res.group == "silenced_cgi"].pivot_table(index="model", columns="assay",
              values="mwu_p_sil_absd_gt_escape", sort=False).to_string(float_format="%.1e"))
        print("\n== XIST, Xa - Xi (expected reversed: ATAC/H3K4me3/H3K27ac < 0)")
        print(xist.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    return res


def plot_ctcf(df):
    """Bound sites per sample: per-site footprint - open medians for pred CTCF and pred ATAC, open - nucleosome."""
    b = df[df.site_set == "bound"]
    stats = [("d_fp_open_ctcf_median", "H1: footprint − open, pred CTCF"),
             ("d_fp_open_atac_median", "footprint − open, pred ATAC"),
             ("d_ctcf_minus_atac_median", "H2: CTCF effect − ATAC effect"),
             ("d_open_nuc_ctcf_median", "H3: open − nucleosome, pred CTCF")]
    fig, ax = plt.subplots(1, len(stats), figsize=(16, 3.8))
    x = np.arange(len(SAMPLES))
    w = 0.8 / len(MODELS)
    for k, (c, title) in enumerate(stats):
        a = ax[k]
        for i, m in enumerate(MODELS):
            v = [b[(b.model == m) & (b["sample"] == s)][c].iloc[0] for s in SAMPLES]
            a.bar(x + (i - (len(MODELS) - 1) / 2) * w, v, w * 0.92, color=COL[m], label=m)
        a.axhline(0, color="#999999", lw=0.8)
        a.set_xticks(x)
        a.set_xticklabels([s.split("_")[0] for s in SAMPLES])
        a.set_title(title, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel("median over bound sites")
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=COL[m], markersize=9, label=m) for m in MODELS],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(MODELS), frameon=False)
    fig.suptitle("CTCF footprints at bound chr21 sites (hole sites excluded), GM12878 + HepG2 models, seed 919",
                 y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(D, "bio_ctcf.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {D}/bio_ctcf.png", flush=True)


def plot_xi(res):
    """Fraction of genes in the expected direction, per mark, silenced CGI vs the two nulls."""
    groups = [("silenced_cgi", "silenced, CGI promoter"), ("escape_par", "escape + PAR (null)"),
              ("chr21", "chr21 (null)")]
    fig, ax = plt.subplots(1, len(groups), figsize=(16, 3.8), sharey=True)
    labs = [l for l, _ in MARKS.values()]
    x = np.arange(len(labs))
    w = 0.8 / len(MODELS)
    for k, (gname, title) in enumerate(groups):
        a = ax[k]
        r = res[res.group == gname]
        for i, m in enumerate(MODELS):
            v = [r[(r.model == m) & (r.assay == l)].frac_expected.iloc[0] for l in labs]
            a.bar(x + (i - (len(MODELS) - 1) / 2) * w, v, w * 0.92, color=COL[m])
        a.axhline(0.5, color="#999999", lw=0.8, ls="--")
        a.set_ylim(0, 1)
        a.set_xticks(x)
        a.set_xticklabels([f"{l} ({'+' if e > 0 else '−'})" for l, e in MARKS.values()], fontsize=8)
        a.set_title(f"{title}, n = {r.n_genes.iloc[0]}", fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    ax[0].set_ylabel("fraction of genes with the\nexpected sign of Xa − Xi")
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=COL[m], markersize=9, label=m) for m in MODELS],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(MODELS), frameon=False)
    fig.suptitle("X inactivation in GM12878 promoters, GM12878 + HepG2 models, seed 919", y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(D, "bio_xi.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {D}/bio_xi.png", flush=True)


def main():
    plot_ctcf(ctcf())
    plot_xi(xi())


if __name__ == "__main__":
    main()
