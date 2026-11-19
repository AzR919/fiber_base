"""X inactivation (Nov_20 report, section 2), restricted to the gene group the test is about: silenced X genes
(subject to X inactivation) with a CpG-island promoter. Reads the per-gene Xa - Xi tables that
plots_code/26_nov_13/xi_promoters.py wrote for every model and training seed; nothing is recomputed per fiber.

Groups (TSS +- 500 bp, genes with >= 5 fibers on each haplotype):
  silenced_cgi  chrX genes not on the escape / PAR list, CpG-island promoter (UCSC CpG-island track)
  silenced_noncgi  the same without a CpG island (input check only: X inactivation methylates CGIs)
  escape_par    escape list (provisional) + PAR genes: expressed from both copies, null
  chr21         chr21 genes, HP1 / HP2 an arbitrary pairing, null
  xist          XIST, expressed only from the Xi, expected reversed

DEV4NF = model16, DEV4 without the FIRE channel (abl_common key 'noFIRE').

    python plots_code/26_nov_20/xi_cgi.py
Writes plots/26_nov_20/xi/: xi_cgi_summary.tsv, xi_cgi_inputs.tsv, xi_cgi_xist.tsv, xi_cgi_h2.tsv,
xi_DEV4NF.png, xi_cgi_fractions.png, log_xi_cgi.txt (stdout).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "26_nov_13"))
from sm_stats import wilcoxon_p, mannwhitney_greater_p

IN = "plots/26_nov_13/molecules"
OUT = "plots/26_nov_20/xi"
# report name -> abl_common key; order = the report's table order
MODELS = {"DEV4NF": "noFIRE", "DEV4": "DEV4", "m6A + CpG": "m6aCpG", "no accessibility": "noAcc", "no CpG": "noCpG"}
SEEDS = [919, 2]
MARKS = {"atac": ("ATAC", 1), "h3k4me3": ("H3K4me3", 1), "h3k27ac": ("H3K27ac", 1),
         "h3k9me3": ("H3K9me3", -1), "h3k27me3": ("H3K27me3", -1)}   # expected sign of Xa - Xi
INPUTS = {"cpg_meth_t": "CpG methylation", "msp_t": "MSP fraction"}
SEED_COL = {919: "#2a78d6", 2: "#eb6834"}
MODEL_COL = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))


def load(key, seed):
    f = os.path.join(IN, f"xi_per_gene_{key}{'' if seed == 919 else f'_seed{seed}'}.tsv")
    return pd.read_csv(f, sep="\t")


def groups(d):
    sil = d["class"] == "subject"
    return {"silenced_cgi": d[sil & (d.cgi == 1)], "silenced_noncgi": d[sil & (d.cgi == 0)],
            "escape_par": d[d["class"].isin(["escape", "par"])], "chr21": d[d["class"] == "autosomal"],
            "xist": d[d["class"] == "xist"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {(m, s): load(k, s) for m, k in MODELS.items() for s in SEEDS}

    # inputs are measured, identical across models and seeds
    g = groups(data[("DEV4NF", 919)])
    rows = [{"group": n, "n_genes": len(v), "n_cgi": int(v.cgi.sum()),
             **{f"median_{c}": v[c].median() for c in INPUTS}} for n, v in g.items()]
    inp = pd.DataFrame(rows)
    inp.to_csv(os.path.join(OUT, "xi_cgi_inputs.tsv"), sep="\t", index=False, float_format="%.4g")
    print("Inputs, median Xa - Xi:\n", inp.to_string(index=False), "\n", flush=True)

    rows, h2, xist = [], [], []
    for (m, s), d in data.items():
        g = groups(d)
        for gname in ["silenced_cgi", "escape_par", "chr21"]:
            for a, (lab, e) in MARKS.items():
                v = g[gname][f"pred_{a}"].dropna()
                rows.append({"model": m, "seed": s, "group": gname, "assay": lab, "n_genes": len(v),
                             "median_d": v.median(), "frac_expected": (np.sign(v) == e).mean(),
                             "wilcoxon_p": wilcoxon_p(v)})
        for a, (lab, _) in MARKS.items():
            c = f"pred_{a}"
            h2.append({"model": m, "seed": s, "assay": lab,
                       "median_absd_silenced_cgi": g["silenced_cgi"][c].abs().median(),
                       "median_absd_escape_par": g["escape_par"][c].abs().median(),
                       "mwu_p_silenced_cgi_greater": mannwhitney_greater_p(g["silenced_cgi"][c].abs(),
                                                                           g["escape_par"][c].abs())})
        x = g["xist"].iloc[0]
        xist.append({"model": m, "seed": s, **{MARKS[a][0]: x[f"pred_{a}"] for a in MARKS},
                     **{INPUTS[c]: x[c] for c in INPUTS}})
    res, h2, xist = pd.DataFrame(rows), pd.DataFrame(h2), pd.DataFrame(xist)
    res.to_csv(os.path.join(OUT, "xi_cgi_summary.tsv"), sep="\t", index=False, float_format="%.4g")
    h2.to_csv(os.path.join(OUT, "xi_cgi_h2.tsv"), sep="\t", index=False, float_format="%.4g")
    xist.to_csv(os.path.join(OUT, "xi_cgi_xist.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        for gname in ["silenced_cgi", "escape_par", "chr21"]:
            r = res[res.group == gname]
            print(f"== {gname}: fraction expected sign (seed 919 / 2)")
            print(r.pivot_table(index="model", columns=["assay", "seed"], values="frac_expected", sort=False)
                  .round(2).to_string(), "\n")
            print(f"== {gname}: median d")
            print(r.pivot_table(index="model", columns=["assay", "seed"], values="median_d", sort=False)
                  .round(3).to_string(), "\n")
            print(f"== {gname}: Wilcoxon p")
            print(r.pivot_table(index="model", columns=["assay", "seed"], values="wilcoxon_p", sort=False)
                  .to_string(float_format="%.1e"), "\n")
        print("== H2: |d| silenced CGI > escape + PAR\n", h2.round(4).to_string(index=False), "\n")
        print("== XIST (Xa - Xi)\n", xist.round(3).to_string(index=False), "\n", flush=True)

    plot_dev4nf(data)
    plot_fractions(res)


def plot_dev4nf(data):
    """DEV4NF, both seeds: d per gene by group for the five marks, plus the two inputs; XIST as a star."""
    gnames = [("silenced_cgi", "silenced,\nCGI promoter"), ("escape_par", "escape\n+ PAR"), ("chr21", "chr21")]
    cols = [(f"pred_{a}", f"{lab}  (expected {'+' if e > 0 else '−'})") for a, (lab, e) in MARKS.items()]
    cols += [(c, f"{lab} (input)") for c, lab in INPUTS.items()]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    rng = np.random.default_rng(0)
    for k, (c, title) in enumerate(cols):
        a = ax.flat[k]
        seeds = SEEDS if c.startswith("pred_") else [919]
        w = 0.36 if len(seeds) == 2 else 0.6
        for j, (gname, _) in enumerate(gnames):
            for i, s in enumerate(seeds):
                v = groups(data[("DEV4NF", s)])[gname][c].dropna()
                pos = j + 1 + (i - (len(seeds) - 1) / 2) * 0.4
                col = SEED_COL[s] if len(seeds) == 2 else "#606060"
                a.boxplot([v], positions=[pos], widths=w, showfliers=False, patch_artist=True,
                          boxprops=dict(facecolor="none", edgecolor=col, lw=1.2),
                          medianprops=dict(color=col, lw=2), whiskerprops=dict(color=col),
                          capprops=dict(color=col))
                a.scatter(pos + rng.uniform(-w / 3, w / 3, len(v)), v, s=4, alpha=0.25, color=col, lw=0)
                if j == 0:
                    x = groups(data[("DEV4NF", s)])["xist"][c].iloc[0]
                    a.scatter([len(gnames) + 0.75 + (i - (len(seeds) - 1) / 2) * 0.3], [x], marker="*", s=140,
                              color=col, edgecolor="white", lw=0.8, zorder=3)
        a.axhline(0, color="#999999", lw=0.8, zorder=0)
        a.set_xticks(list(range(1, len(gnames) + 1)) + [len(gnames) + 0.75])
        a.set_xticklabels([lab for _, lab in gnames] + ["XIST"], fontsize=8)
        a.set_xlim(0.4, len(gnames) + 1.2)
        a.set_title(title, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    ax.flat[-1].axis("off")
    ax.flat[-1].legend(handles=[Patch(facecolor="none", edgecolor=SEED_COL[919], label="seed 919"),
                                Patch(facecolor="none", edgecolor=SEED_COL[2], label="seed 2"),
                                Patch(facecolor="none", edgecolor="#606060", label="input (same for both)"),
                                Line2D([], [], marker="*", ls="", color="#606060", markersize=11, label="XIST")],
                       loc="center", frameon=False, fontsize=10)
    for a in ax[:, 0]:
        a.set_ylabel("Xa − Xi per gene")
    fig.suptitle("GM12878 promoters, DEV4NF per-fiber predictions: Xa (HP1) − Xi (HP2) per gene", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xi_DEV4NF.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}/xi_DEV4NF.png", flush=True)


def plot_fractions(res):
    """Fraction of silenced-CGI genes in the expected direction, every model and seed, one panel per mark."""
    r = res[res.group == "silenced_cgi"]
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for k, (a, (lab, e)) in enumerate(MARKS.items()):
        p = ax[k]
        for i, m in enumerate(MODELS):
            v = [r[(r.model == m) & (r.seed == s) & (r.assay == lab)].frac_expected.iloc[0] for s in SEEDS]
            x = [i - 0.17, i + 0.17]   # seed 919 left, seed 2 right, so equal values don't hide each other
            p.plot(x, v, color=MODEL_COL[m], lw=1.5, zorder=1)
            p.scatter([x[0]], [v[0]], marker="o", s=60, color=MODEL_COL[m], edgecolor="white", lw=1.5, zorder=2)
            p.scatter([x[1]], [v[1]], marker="s", s=60, color=MODEL_COL[m], edgecolor="white", lw=1.5, zorder=2)
        p.axhline(0.5, color="#999999", lw=0.8, ls="--", zorder=0)
        p.set_ylim(0, 1)
        p.set_xticks(range(len(MODELS)))
        p.set_xticklabels(list(MODELS), rotation=35, ha="right", fontsize=8)
        p.set_title(f"{lab} ({'Xa higher' if e > 0 else 'Xa lower'})", fontsize=10)
        p.spines[["top", "right"]].set_visible(False)
        p.grid(axis="y", color="#e6e6e6", lw=0.6)
        p.set_axisbelow(True)
    ax[0].set_ylabel("fraction of silenced CGI genes\nin the expected direction")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#606060", label="seed 919 (left)"),
                        Line2D([], [], marker="s", ls="", color="#606060", label="seed 2 (right)"),
                        Line2D([], [], ls="--", color="#999999", label="chance")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("X inactivation at silenced CGI promoters (425 genes), every model and training seed", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "xi_cgi_fractions.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}/xi_cgi_fractions.png", flush=True)


if __name__ == "__main__":
    main()
