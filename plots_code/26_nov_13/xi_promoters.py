"""Single-molecule test T5b (reports/TODO.md 2a/2b): X inactivation in GM12878, read per haplotype. The model was
trained on bulk tracks (both X copies mixed) and never saw haplotypes, so an Xa/Xi split in its per-fiber
predictions has to come from the molecules themselves.

Input: molecule_predict.py output for GM12878 on data_prep/build_targets.py's tss_chrX_chr21.bed (protein-coding
TSS +- 500 bp; class par / escape / xist / subject on chrX, autosomal on chr21; cgi = CpG-island promoter).
GM12878 is trio-phased (one phase block per chromosome), so HP1 / HP2 are whole-chromosome haplotypes. Xi is the
haplotype whose XIST promoter is CpG-unmethylated in the fibers (measured input, not a model output); the smoke test
gave HP2 (XIST promoter CpG methylation 0.07 on HP2 vs 0.24 on HP1). This script re-derives it.

Per gene: d = mean over Xa fibers - mean over Xi fibers (genes with >= MIN_FIBERS fibers on each haplotype), for
every predicted assay and for the inputs (CpG methylation, MSP fraction). On chr21, 'Xa' / 'Xi' are just HP1 / HP2
mapped the same way (arbitrary on an autosome, so the null).

Hypotheses and expected directions (fixed in advance; Xi promoters are silent, CpG-island methylated, depleted of
active marks and enriched for H3K27me3 / H3K9me3; escape and PAR genes are expressed from both copies):
  H1 subject genes: d > 0 for pred ATAC, H3K4me3, H3K27ac; d < 0 for pred H3K27me3, H3K9me3 (Wilcoxon over genes;
     also the fraction of genes with the expected sign). Strongest at CGI promoters.
  H2 escape + PAR genes: |d| smaller than for subject genes (Mann-Whitney on d, per assay).
  H3 null: chr21 genes d ~ 0 (median near 0, no excess of either sign).
  H4 XIST: opposite to H1 (active on Xi).
  Data check (inputs, not the model): subject CGI promoters have higher CpG methylation and lower MSP fraction on Xi.
Caveat: the model sees the fibers' CpG methylation and accessibility, which already differ between Xa and Xi. The
test shows its predictions follow X-inactivation biology on unseen chromosomes; the input ablations (model19 no
CpG, model20 CpG only, model16 no FIRE) show which input carries it.

    python plots_code/26_nov_13/xi_promoters.py [--model DEV4|noFIRE|m6aCpG|noAcc|noCpG] [--seed 919|2] [--dir DIR]
                                                [--out DIR]
Writes plots/26_nov_13/molecules/xi_*_<model>.tsv and xi_promoters_<model>.png (or into --out).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_stats import wilcoxon_p, mannwhitney_greater_p, spearman
from abl_common import MODELS, TRAIN_SEEDS, molecule_dir, tag as out_tag

OUT = "plots/26_nov_13/molecules"
MIN_FIBERS = 5
SAMPLE = "GM12878"
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k27me3", "h3k4me1", "h3k36me3", "h3k9me3", "ctcf"]
EXPECT = {"atac": 1, "h3k4me3": 1, "h3k27ac": 1, "h3k27me3": -1, "h3k9me3": -1}   # sign of Xa - Xi on subject genes
INPUTS = {"cpg_meth_t": "CpG meth (input)", "msp_t": "MSP fraction (input)"}
GROUPS = ["subject", "subject_cgi", "escape+par", "autosomal"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DEV4", choices=list(MODELS), help="abl_common.MODELS key (output suffix)")
    p.add_argument("--seed", type=int, default=919, choices=TRAIN_SEEDS, help="training seed (output suffix _seed<n> if not 919)")
    p.add_argument("--dir", default=None, help="molecule_predict output dir (default: the model's newest *_tss_chrX_chr21)")
    p.add_argument("--out", default=OUT, help="output folder")
    return p.parse_args()


def main():
    args = parse_args()
    d = args.dir or molecule_dir(args.model, "tss_chrX_chr21", args.seed)
    tag = out_tag(args.model, args.seed)
    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(d, f"fibers_{SAMPLE}.tsv.gz"), sep="\t")
    df = df[df.hp.isin([1, 2, 1.0, 2.0])].copy()
    df["hp"] = df.hp.astype(int)
    print(f"{d}: {len(df)} phased fiber x TSS rows, {df.target.nunique()} TSSs", flush=True)

    # which haplotype is Xi: the one with the unmethylated XIST promoter
    x = df[df["class"] == "xist"].groupby("hp").cpg_meth_t.agg(["mean", "size"])
    print(f"XIST promoter CpG methylation by haplotype:\n{x}", flush=True)
    xi = int(x["mean"].idxmin())
    xa = 3 - xi
    print(f"Xi = HP{xi}, Xa = HP{xa}", flush=True)

    cols = [f"pred_{a}" for a in ASSAYS if f"pred_{a}" in df] + list(INPUTS)
    g = df.groupby(["target", "gene", "class", "cgi", "hp"])[cols].mean()
    n = df.groupby(["target", "hp"]).size().unstack(fill_value=0)
    ok = n.index[(n.get(1, 0) >= MIN_FIBERS) & (n.get(2, 0) >= MIN_FIBERS)]
    g = g[g.index.get_level_values("target").isin(ok)]
    dlt = g.xs(xa, level="hp") - g.xs(xi, level="hp")
    dlt = dlt.reset_index()
    dlt["n_xa"] = dlt.target.map(n[xa])
    dlt["n_xi"] = dlt.target.map(n[xi])
    dlt.to_csv(os.path.join(OUT, f"xi_per_gene_{tag}.tsv"), sep="\t", index=False, float_format="%.4g")

    groups = {"subject": dlt["class"] == "subject",
              "subject_cgi": (dlt["class"] == "subject") & (dlt.cgi == 1),
              "escape+par": dlt["class"].isin(["escape", "par"]),
              "autosomal": dlt["class"] == "autosomal"}
    rows = []
    for gname, mask in groups.items():
        sub = dlt[mask]
        for c in cols:
            v = sub[c].dropna()
            a = c.replace("pred_", "")
            e = EXPECT.get(a)
            rows.append({"group": gname, "column": c, "n_genes": len(v), "median_d": v.median(),
                         "mean_d": v.mean(), "frac_pos": (v > 0).mean(),
                         "frac_expected_sign": ((np.sign(v) == e).mean() if e else np.nan),
                         "wilcoxon_p": wilcoxon_p(v)})
    res = pd.DataFrame(rows)
    # H2: |d| subject vs escape+par
    for c in cols:
        s, e = dlt[groups["subject"]][c].abs().dropna(), dlt[groups["escape+par"]][c].abs().dropna()
        if len(s) >= 5 and len(e) >= 5:
            p = mannwhitney_greater_p(s, e)
            res.loc[(res.group == "escape+par") & (res.column == c), "mwu_p_subject_absd_greater"] = p
    res.to_csv(os.path.join(OUT, f"xi_summary_{tag}.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(res, flush=True)
    xr = dlt[dlt["class"] == "xist"]
    print(f"\nXIST (Xa - Xi):\n{xr.T}", flush=True)

    # figure: one panel per column, d by group (box + strip), expected sign marked
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    for k, c in enumerate(cols[:10]):
        a = ax.flat[k]
        data = [dlt[groups[gname]][c].dropna() for gname in GROUPS]
        a.boxplot(data, labels=[f"{gname}\n(n={len(v)})" for gname, v in zip(GROUPS, data)], showfliers=False)
        for j, v in enumerate(data):
            a.scatter(np.full(len(v), j + 1) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v,
                      s=3, alpha=0.3, color="k")
        a.axhline(0, color="r", lw=0.8)
        name = INPUTS.get(c, f"pred {c.replace('pred_', '')}")
        e = EXPECT.get(c.replace("pred_", ""))
        a.set_title(name + (f"  (expected {'+' if e > 0 else '-'} on subject)" if e else ""), fontsize=10)
        a.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"GM12878 promoters: Xa (HP{xa}) - Xi (HP{xi}) per gene, {tag} per-fiber predictions", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"xi_promoters_{tag}.png"), dpi=130, bbox_inches="tight")
    print(f"\nwrote {OUT}/xi_promoters_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
