"""X inactivation with block phasing (reports/TODO.md section 3 #2; TODO_extended.md T5b). The sealed XI cells
(HG008_3E4, GM25455, HEK293T) are phased in HiPhase blocks of a few promoters, so HP1 / HP2 mean nothing across
blocks and the Xi haplotype has to be set per block. This script defines that rule and validates it on GM12878, whose
trio phasing is chromosome-wide (one block, PS = 1), by cutting it into the sealed cells' real block layouts.

Rule (fixed before looking at any block result):
  signal     per informative promoter j with >= MIN_FIBERS fibers with a CpG call (>= MIN_CPG covered CpGs over the
             TSS +- 500 bp) on each haplotype: s_j = mean fiber CpG methylation on HP2 - on HP1. Informative on chrX:
             silenced CGI promoters (class subject, cgi = 1; methylated on the Xi) and XIST (unmethylated on the Xi,
             so s_j is negated). On chr21 (the null): autosomal CGI promoters, the same way.
  per gene   for a tested promoter g in block b: score = mean of s_j over the informative promoters j != g in b (the
             tested gene left out, so its own methylation never sets its Xi). Xi = HP2 if score >= MARGIN, HP1 if
             score <= -MARGIN, else no call (the gene is dropped).
  then       as xi_promoters.py: per gene d = mean over Xa fibers - mean over Xi fibers (>= MIN_FIBERS fibers on each
             haplotype), for the predicted marks and the inputs; readouts as realism_seeds.py: per group (silenced CGI,
             escape + PAR, chr21, XIST) the fraction of genes with the expected sign, per mark and the mean of 5 marks.
Validation on GM12878 (every model and seed of the XI realism test), per block map:
  - pseudo-blocks: each GM12878 target gets the PS block the sealed cell has at that target (targets without a block
    there are unphased and dropped); each block's HP labels are swapped at random (N_FLIPS draws), so the rule can't
    use GM12878's chromosome-wide phase; results must not depend on the draw;
  - accuracy: decided Xi vs the chromosome-wide truth (Xi = the haplotype with the unmethylated XIST promoter);
  - coverage: silenced CGI genes with a call / those the chromosome-wide test scores;
  - agreement: frac_expected and per-mark readouts vs the chromosome-wide test on the same model and seed;
  - chr21 null twice: by the rule (Xi from autosomal allele-specific methylation, mostly no call) and with a random
    orientation per block;
  - escape control without an Xi call (added 27 Nov, after the first run: escape + PAR genes cluster in blocks
    without silenced neighbours and mostly get no call): per gene |pred(HP1) - pred(HP2)|, silenced CGI > escape +
    PAR (one-sided Mann-Whitney per mark), on every phased gene; expected as chromosome-wide.
Expected (fixed in advance): accuracy >= 0.95; coverage >= 0.5; silenced CGI frac_expected within 0.05 of the
chromosome-wide value in every run; nulls near 0.5.

    python plots_code/26_dec_04/xi_blocks.py
Writes plots/26_dec_04/xi_blocks/: xi_blocks_runs.tsv (one row per run x map x flip draw), xi_blocks_summary.tsv,
xi_blocks.png; prints the tables. Needs block_map_<cell>.tsv from xi_block_maps.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_13"))
from abl_common import molecule_dir  # noqa: E402
from sm_stats import mannwhitney_greater_p  # noqa: E402

OUT = "plots/26_dec_04/xi_blocks"
SAMPLE = "GM12878"
MIN_FIBERS, MIN_CPG, MARGIN, N_FLIPS = 5, 3, 0.2, 5
MAPS = ["HG008_3E4", "GM25455_LCL", "HEK293T"]
S3 = [919, 2, 3]
RUNS = [("DEV4NF", "noFIRE", [919, 2]), ("R1a A", "mixR1a", S3), ("Pair A", "pairNF", S3),
        ("R1a B", "mixR1aB", S3), ("Pair B", "pairNFB", S3)]
MARKS = {"atac": 1, "h3k4me3": 1, "h3k27ac": 1, "h3k9me3": -1, "h3k27me3": -1}   # as realism_bio.MARKS
GROUPS = ["silenced_cgi", "escape_par", "chr21", "xist"]


# ---------------------------------------------------------------- per (target, haplotype) table
def load(key, seed):
    d = molecule_dir(key, "tss_chrX_chr21", seed)
    df = pd.read_csv(os.path.join(d, f"fibers_{SAMPLE}.tsv.gz"), sep="\t")
    df = df[df.hp.isin([1, 2])].copy()
    df["hp"] = df.hp.astype(int)
    df["meth"] = np.where(df.cpg_n_t >= MIN_CPG, df.cpg_meth_t, np.nan)
    cols = [f"pred_{a}" for a in MARKS] + ["cpg_meth_t", "msp_t"]
    tab = df.groupby(["target", "hp"]).agg(n=("read", "size"), n_meth=("meth", "count"), meth=("meth", "mean"),
                                           **{c: (c, "mean") for c in cols}).reset_index()
    meta = df.groupby("target").agg(chrom=("chrom", "first"), gene=("gene", "first"), cls=("class", "first"),
                                    cgi=("cgi", "first"))
    return tab, meta, d


def wide(tab, col):
    return tab.pivot(index="target", columns="hp", values=col).reindex(columns=[1, 2])


def group_of(meta):
    g = pd.Series("", index=meta.index)
    g[(meta.cls == "subject") & (meta.cgi == 1)] = "silenced_cgi"
    g[meta.cls.isin(["escape", "par"])] = "escape_par"
    g[meta.cls == "autosomal"] = "chr21"
    g[meta.cls == "xist"] = "xist"
    return g


# ---------------------------------------------------------------- Xi calls
def xi_reference(tab, meta):
    """Chromosome-wide (xi_promoters.py): Xi = the haplotype with the lower XIST promoter methylation, for every
    target (chr21: the same HP mapping, arbitrary, the null)."""
    x = tab[tab.target.isin(meta.index[meta.cls == "xist"])].groupby("hp").cpg_meth_t.mean()
    xi = int(x.idxmin())
    return pd.Series(xi, index=meta.index, dtype=float), xi


def xi_blocks(tab, meta, block, flip):
    """Per target: Xi in the original HP labels by the block rule, or NaN. block: target -> block id (NaN = none);
    flip: block id -> bool (that block's HP labels are swapped before the rule sees them)."""
    n = wide(tab, "n_meth").fillna(0)
    m = wide(tab, "meth")
    blk = block.reindex(meta.index)
    fl = blk.map(lambda b: bool(flip.get(b, False)) if pd.notna(b) else False)
    s = (m[2] - m[1]).reindex(meta.index)
    s[fl] = -s[fl]                                                       # what the rule sees after the swap
    informative = ((meta.cls == "subject") & (meta.cgi == 1)) | (meta.cls == "xist") | \
                  ((meta.cls == "autosomal") & (meta.cgi == 1))
    ok = informative & (n[1].reindex(meta.index) >= MIN_FIBERS) & (n[2].reindex(meta.index) >= MIN_FIBERS) & blk.notna()
    s = s.where(ok)
    s[meta.cls == "xist"] = -s[meta.cls == "xist"]
    tot = s.groupby(blk).agg(["sum", "count"])
    out = pd.Series(np.nan, index=meta.index)
    score = pd.Series(np.nan, index=meta.index)
    for t in meta.index:
        b = blk[t]
        if pd.isna(b) or b not in tot.index:
            continue
        sm, ct = tot.loc[b, "sum"], tot.loc[b, "count"]
        if pd.notna(s[t]):
            sm, ct = sm - s[t], ct - 1
        if ct < 1:
            continue
        sc = sm / ct
        score[t] = sc
        if abs(sc) >= MARGIN:
            xi_seen = 2 if sc > 0 else 1
            out[t] = 3 - xi_seen if fl[t] else xi_seen
    return out, score


def random_orientation(meta, block, rng):
    blk = block.reindex(meta.index)
    ori = {b: int(rng.integers(1, 3)) for b in blk.dropna().unique()}
    return blk.map(lambda b: ori.get(b, np.nan) if pd.notna(b) else np.nan).astype(float)


# ---------------------------------------------------------------- readouts
def deltas(tab, xi):
    """Per target with an Xi call and >= MIN_FIBERS fibers on each haplotype: Xa - Xi for marks and inputs."""
    n = wide(tab, "n").fillna(0)
    rows = {}
    for c in [f"pred_{a}" for a in MARKS] + ["cpg_meth_t", "msp_t"]:
        w = wide(tab, c)
        xi_ = xi.reindex(w.index)
        a = np.where(xi_ == 1, w[2], w[1])
        i = np.where(xi_ == 1, w[1], w[2])
        rows[c] = pd.Series(a - i, index=w.index)
    d = pd.DataFrame(rows)
    keep = xi.reindex(d.index).notna() & (n[1].reindex(d.index) >= MIN_FIBERS) & (n[2].reindex(d.index) >= MIN_FIBERS)
    return d[keep]


def readouts(d, groups):
    r = {}
    for gname in GROUPS:
        sub = d[groups.reindex(d.index) == gname]
        r[f"{gname}_n"] = len(sub)
        fr = []
        for a, e in MARKS.items():
            v = sub[f"pred_{a}"].dropna()
            f = (np.sign(v) == e).mean() if len(v) else np.nan
            fr.append(f)
            if gname == "silenced_cgi":
                r[f"{gname}_{a}"] = f
        r[f"{gname}_frac_expected"] = np.nanmean(fr) if len(sub) else np.nan
        if gname == "silenced_cgi":
            r[f"{gname}_cpg_d_median"] = sub.cpg_meth_t.median()
    return r


def absd_readouts(tab, meta, groups, block=None):
    """Orientation-free escape control (H2 of xi_promoters.py): per gene |pred(HP1) - pred(HP2)| needs no Xi call,
    only both haplotypes (>= MIN_FIBERS fibers each) and, with block phasing, a block. Silenced CGI genes should
    differ more between their copies than escape + PAR genes (expressed from both). Per mark: median |d| per group
    and the one-sided Mann-Whitney p for silenced > escape + PAR."""
    n = wide(tab, "n").fillna(0)
    keep = (n[1] >= MIN_FIBERS) & (n[2] >= MIN_FIBERS)
    if block is not None:
        keep &= block.reindex(n.index).notna()
    r = {}
    passed = 0
    for a in MARKS:
        w = wide(tab, f"pred_{a}")[keep]
        ad = (w[1] - w[2]).abs()
        g = groups.reindex(ad.index)
        sil, esc = ad[g == "silenced_cgi"].dropna(), ad[g == "escape_par"].dropna()
        p = mannwhitney_greater_p(sil, esc)
        r[f"absd_{a}_silenced"], r[f"absd_{a}_escape"], r[f"absd_{a}_p"] = sil.median(), esc.median(), p
        passed += int(p < 0.05)
    r["absd_n_silenced"], r["absd_n_escape"] = int((groups.reindex(n.index[keep]) == "silenced_cgi").sum()), \
        int((groups.reindex(n.index[keep]) == "escape_par").sum())
    r["absd_marks_pass"] = passed
    return r


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    maps = {}
    for c in MAPS:
        f = f"{OUT}/block_map_{c}.tsv"
        bm = pd.read_csv(f, sep="\t")
        maps[c] = bm.set_index("target").ps
    rows = []
    for label, key, seeds in RUNS:
        for seed in seeds:
            tab, meta, d = load(key, seed)
            groups = group_of(meta)
            ref_xi, xi_hp = xi_reference(tab, meta)
            ref_d = deltas(tab, ref_xi)
            ref = readouts(ref_d, groups)
            base = {"model": label, "key": key, "seed": seed}
            rows.append({**base, "map": "chromosome-wide", "draw": 0, **ref, **absd_readouts(tab, meta, groups),
                         "accuracy_all": 1.0,
                         "accuracy_silenced_cgi": 1.0, "coverage_silenced_cgi": 1.0})
            print(f"{label} seed {seed}: {d}; chromosome-wide Xi = HP{xi_hp}, silenced CGI genes "
                  f"{ref['silenced_cgi_n']}, frac_expected {ref['silenced_cgi_frac_expected']:.3f}", flush=True)
            for mname, block in maps.items():
                for draw in range(N_FLIPS):
                    rng = np.random.default_rng(919 + draw)
                    flip = {b: bool(rng.integers(0, 2)) for b in block.dropna().unique()}
                    xi, score = xi_blocks(tab, meta, block, flip)
                    bd = deltas(tab, xi)
                    r = readouts(bd, groups)
                    x = meta.index[(meta.chrom == "chrX")]
                    dec = xi.reindex(x).dropna()
                    sil = groups.reindex(dec.index) == "silenced_cgi"
                    acc_all = float((dec == ref_xi.reindex(dec.index)).mean()) if len(dec) else np.nan
                    acc_sil = float((dec[sil] == ref_xi.reindex(dec[sil].index)).mean()) if sil.any() else np.nan
                    rnd = random_orientation(meta, block, rng)
                    rd = readouts(deltas(tab, rnd), groups)
                    rows.append({**base, "map": mname, "draw": draw, **r,
                                 **absd_readouts(tab, meta, groups, block), "accuracy_all": acc_all,
                                 "accuracy_silenced_cgi": acc_sil,
                                 "coverage_silenced_cgi": r["silenced_cgi_n"] / max(1, ref["silenced_cgi_n"]),
                                 "n_called_chrX": len(dec),
                                 "chr21_random_n": rd["chr21_n"], "chr21_random_frac_expected": rd["chr21_frac_expected"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/xi_blocks_runs.tsv", sep="\t", index=False, float_format="%.4g")

    # flip invariance: the rule must give the same calls whatever the draw
    blk = df[df["map"] != "chromosome-wide"]
    var = blk.groupby(["model", "seed", "map"])[["silenced_cgi_frac_expected", "silenced_cgi_n", "accuracy_all"]].agg(
        lambda v: v.max() - v.min()).max()
    print(f"\nlargest change between flip draws (must be 0): {var.to_dict()}", flush=True)

    one = blk[blk.draw == 0].copy()
    ref = df[df["map"] == "chromosome-wide"].set_index(["model", "seed"])
    one["ref_frac_expected"] = [ref.loc[(m, s), "silenced_cgi_frac_expected"] for m, s in zip(one.model, one.seed)]
    one["delta_frac_expected"] = one.silenced_cgi_frac_expected - one.ref_frac_expected
    cols = ["model", "seed", "map", "silenced_cgi_n", "coverage_silenced_cgi", "accuracy_silenced_cgi", "accuracy_all",
            "silenced_cgi_frac_expected", "ref_frac_expected", "delta_frac_expected", "escape_par_frac_expected",
            "chr21_n", "chr21_frac_expected", "chr21_random_n", "chr21_random_frac_expected", "xist_frac_expected"]
    one[cols].to_csv(f"{OUT}/xi_blocks_summary.tsv", sep="\t", index=False, float_format="%.4g")
    print("\n=== per run and block map (draw 0) ===")
    print(one[cols].round(3).to_string(index=False))
    print("\n=== chromosome-wide reference ===")
    print(ref.reset_index()[["model", "seed", "silenced_cgi_n", "silenced_cgi_frac_expected", "escape_par_frac_expected",
                             "chr21_frac_expected", "xist_frac_expected"]].round(3).to_string(index=False))
    print("\n=== per map, over all runs ===")
    agg = one.groupby("map").agg(coverage=("coverage_silenced_cgi", "mean"), acc_sil=("accuracy_silenced_cgi", "min"),
                                 acc_all=("accuracy_all", "min"), max_abs_delta=("delta_frac_expected",
                                                                                lambda v: v.abs().max()),
                                 chr21_n=("chr21_n", "mean"), chr21_random=("chr21_random_frac_expected", "mean"))
    print(agg.round(3).to_string())
    print("\n=== escape control, orientation-free: median |HP1 - HP2| silenced CGI vs escape + PAR, MWU p ===")
    ac = ["absd_n_silenced", "absd_n_escape", "absd_marks_pass"] + \
        [f"absd_{a}_{k}" for a in MARKS for k in ("silenced", "escape", "p")]
    both = pd.concat([ref.reset_index().assign(map="chromosome-wide"), one], ignore_index=True)
    print(both[["model", "seed", "map"] + ac].round(3).to_string(index=False))
    both[["model", "seed", "map"] + ac].to_csv(f"{OUT}/xi_blocks_escape.tsv", sep="\t", index=False,
                                               float_format="%.4g")
    per_mark = [f"silenced_cgi_{a}" for a in MARKS]
    print("\n=== silenced CGI, fraction with the expected sign per mark: block (mean over maps) vs chromosome-wide ===")
    pm = one.groupby(["model", "seed"])[per_mark].mean().round(3)
    pr = ref[per_mark].round(3)
    print(pd.concat({"block": pm, "chromosome-wide": pr}, axis=1).to_string())

    # figure: block vs chromosome-wide frac_expected; coverage and accuracy per map
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    col = {"HG008_3E4": "#d62728", "GM25455_LCL": "#1f77b4", "HEK293T": "#2ca02c"}
    for mname, g in one.groupby("map"):
        ax[0].scatter(g.ref_frac_expected, g.silenced_cgi_frac_expected, color=col[mname], label=mname, s=20)
    lo, hi = 0.4, 1.0
    ax[0].plot([lo, hi], [lo, hi], "k-", lw=0.6)
    ax[0].fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05], color="k", alpha=0.08)
    ax[0].set_xlabel("chromosome-wide phasing")
    ax[0].set_ylabel("block phasing")
    ax[0].set_title("Silenced CGI promoters: fraction with the expected sign\n(mean of 5 marks; each point a model x seed)")
    ax[0].legend(fontsize=8)
    x = np.arange(len(MAPS))
    ax[1].bar(x - 0.2, [agg.loc[m, "coverage"] for m in MAPS], 0.4, label="coverage (genes called / chromosome-wide)")
    ax[1].bar(x + 0.2, [agg.loc[m, "acc_sil"] for m in MAPS], 0.4, label="accuracy of the Xi call (min over runs)")
    ax[1].axhline(0.95, color="k", ls="--", lw=0.6)
    ax[1].axhline(0.5, color="k", ls=":", lw=0.6)
    ax[1].set_xticks(x, MAPS)
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("Block layouts of the sealed XI cells, applied to GM12878")
    fig.tight_layout()
    fig.savefig(f"{OUT}/xi_blocks.png", dpi=150)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
