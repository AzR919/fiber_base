"""Single-molecule test T5e (reports/TODO.md 2e): does a fiber's predicted CTCF follow whether CTCF is bound on that
molecule? The model was trained on bulk ChIP only, so per-molecule binding was never a label.

Input: molecule_predict.py output on data_prep/build_targets.py's ctcf_sites_chr21.bed (CTCF motif sites inside
ENCODE CTCF peaks = bound, and score-matched motif sites outside every dev line's peaks = unbound; chr21 = test
chromosome). One row per (site, fiber) with pred_<assay> and the fiber's input summaries over the motif (t) and the
100 bp on either side (l, r).

Per-fiber state at the motif (fixed before looking at results; Stergachis et al. 2020: ~30% of accessible fibers
at a CTCF site carry the footprint):
  nucleosome   nuc_t >= 0.5
  footprint    msp_t >= 0.5 (inside an MSP) and the motif's m6A rate <= 0.5 x the mean flank rate (protected)
  open         msp_t >= 0.5 and motif m6A rate > 0.5 x flank rate (accessible, not protected)
  other        anything else (read ends, no A/T calls, no flank calls)
Motifs with fewer than MIN_AT A/T positions, or flanks with no m6A calls, are 'other'.

Hypotheses and expected directions (fixed in advance):
  H1 (main) at bound sites, footprint fibers get higher pred_ctcf than open fibers at the same site: per-site
     difference mean(footprint) - mean(open) > 0, Wilcoxon signed-rank over sites (sites with >= MIN_PER_CLASS
     fibers in both states).
  H2 (specificity) the same per-site difference is larger for pred_ctcf than for pred_atac (both states are
     accessible, so accessibility alone should not separate them).
  H3 nucleosome-covered fibers get the lowest pred_ctcf (nucleosome < open < footprint, per-site means).
  H4 (data check, no model) footprints are more frequent among accessible fibers at bound than at unbound sites.
  H5 (null) at unbound sites, mean pred_ctcf is lower than at bound sites with the same motif score.
Caveat: the model sees m6A (so it sees the footprint) and the reference sequence (so it sees the motif in both
sets). The test is whether bulk supervision taught it that a protected motif on an accessible molecule means CTCF.

    python plots_code/26_nov_13/ctcf_footprints.py [--model DEV4|noFIRE|m6aCpG|noAcc|noCpG] [--seed 919|2] [--dir DIR]
Writes plots/26_nov_13/molecules/ctcf_*_<model>.tsv and ctcf_footprints_<model>.png.
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
from abl_common import MODELS, SEEDS, molecule_dir, tag as out_tag

OUT = "plots/26_nov_13/molecules"
MIN_AT, MIN_PER_CLASS = 4, 3
STATES = ["nucleosome", "open", "footprint"]
COLORS = {"nucleosome": "#8c8c8c", "open": "#4C9BE8", "footprint": "#D9534F", "other": "#dddddd"}
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k27me3", "h3k4me1", "h3k36me3", "h3k9me3", "ctcf"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DEV4", choices=list(MODELS), help="abl_common.MODELS key (output suffix)")
    p.add_argument("--seed", type=int, default=919, choices=SEEDS, help="training seed (output suffix _seed<n> if not 919)")
    p.add_argument("--dir", default=None, help="molecule_predict output dir (default: the model's newest *_ctcf_sites_chr21)")
    return p.parse_args()


def fiber_state(df):
    flank = df[["m6a_rate_l", "m6a_rate_r"]].mean(1)
    acc = df.msp_t >= 0.5
    ok = (df.m6a_n_t >= MIN_AT) & (flank > 0)
    prot = df.m6a_rate_t <= 0.5 * flank
    state = np.full(len(df), "other", dtype=object)
    state[(acc & ok & prot).values] = "footprint"
    state[(acc & ok & ~prot).values] = "open"
    state[(df.nuc_t >= 0.5).values] = "nucleosome"
    return state


def paired(site_means, a, b, col):
    """Per-site mean(col | state a) - mean(col | state b) over sites with both."""
    w = site_means[col].unstack("state")
    w = w.dropna(subset=[a, b])
    return w[a] - w[b]


def wilcoxon(x):
    x = x.dropna()
    if len(x) < 5:
        return np.nan
    return wilcoxon_p(x)


def main():
    args = parse_args()
    d = args.dir or molecule_dir(args.model, "ctcf_sites_chr21", args.seed)
    tag = out_tag(args.model, args.seed)
    os.makedirs(OUT, exist_ok=True)
    print(f"reading {d}", flush=True)
    files = sorted(glob.glob(os.path.join(d, "fibers_*.tsv.gz")))
    summary, perfib = [], {}
    for f in files:
        sample = os.path.basename(f)[len("fibers_"):-len(".tsv.gz")]
        cell = sample.split("_")[0]
        df = pd.read_csv(f, sep="\t")
        df["state"] = fiber_state(df)
        # a site is bound *in this cell line* when this line has a peak there; bound elsewhere only = 'other_line'
        df["site_set"] = np.where(df["set"] == "unbound", "unbound",
                                  np.where(df[cell] > 0, "bound", "bound_other_line"))
        perfib[sample] = df
        counts = df.groupby(["site_set", "state"]).size().unstack(fill_value=0)
        n_sites = df.groupby("site_set").target.nunique()
        print(f"\n{sample}: {len(df)} fiber x site rows, sites {n_sites.to_dict()}\n{counts}", flush=True)

        pcols = [f"pred_{a}" for a in ASSAYS if f"pred_{a}" in df]
        sm = df[df.state.isin(STATES)].groupby(["site_set", "target", "state"])[pcols].agg(["mean", "size"])
        means = sm.xs("mean", axis=1, level=1)
        sizes = sm.xs("size", axis=1, level=1).iloc[:, 0]
        means = means[sizes >= MIN_PER_CLASS]

        for site_set in ["bound", "unbound", "bound_other_line"]:
            if site_set not in means.index.get_level_values(0):
                continue
            m = means.xs(site_set, level="site_set")
            sub = df[df.site_set == site_set]
            acc = sub[sub.state.isin(["open", "footprint"])]
            row = {"sample": sample, "site_set": site_set, "n_sites": sub.target.nunique(),
                   "n_fibers": len(sub),
                   **{f"frac_{s}": float((sub.state == s).mean()) for s in STATES + ["other"]},
                   "footprint_frac_of_accessible": float((acc.state == "footprint").mean()) if len(acc) else np.nan,
                   "pred_ctcf_site_mean": float(sub.groupby("target").pred_ctcf.mean().mean())}
            for a in ["ctcf", "atac", "h3k4me3"]:
                dlt = paired(m, "footprint", "open", f"pred_{a}")
                row.update({f"d_fp_open_{a}_median": float(dlt.median()) if len(dlt) else np.nan,
                            f"d_fp_open_{a}_frac_pos": float((dlt > 0).mean()) if len(dlt) else np.nan,
                            f"d_fp_open_{a}_p": wilcoxon(dlt), f"d_fp_open_{a}_n": int(len(dlt))})
            dl = paired(m, "open", "nucleosome", "pred_ctcf")
            row.update({"d_open_nuc_ctcf_median": float(dl.median()) if len(dl) else np.nan,
                        "d_open_nuc_ctcf_p": wilcoxon(dl), "d_open_nuc_ctcf_n": int(len(dl))})
            # H2: per-site specificity, ctcf effect minus atac effect (same sites)
            dc, da = paired(m, "footprint", "open", "pred_ctcf"), paired(m, "footprint", "open", "pred_atac")
            spec = (dc - da).dropna()
            row.update({"d_ctcf_minus_atac_median": float(spec.median()) if len(spec) else np.nan,
                        "d_ctcf_minus_atac_p": wilcoxon(spec)})
            summary.append(row)

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT, f"ctcf_summary_{tag}.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n", summary.T, flush=True)

    # H5: bound vs unbound pred_ctcf at matched motif score (score bins of 1 bit)
    h5 = []
    for sample, df in perfib.items():
        site = df[df.site_set != "bound_other_line"].groupby(["target", "site_set", "score"]).pred_ctcf.mean()
        site = site.reset_index()
        site["bin"] = np.floor(site.score).astype(int)
        for b, g in site.groupby("bin"):
            bd, ub = g[g.site_set == "bound"].pred_ctcf, g[g.site_set == "unbound"].pred_ctcf
            if len(bd) and len(ub):
                h5.append({"sample": sample, "score_bin": b, "n_bound": len(bd), "n_unbound": len(ub),
                           "bound_mean": bd.mean(), "unbound_mean": ub.mean()})
    h5 = pd.DataFrame(h5)
    h5.to_csv(os.path.join(OUT, f"ctcf_bound_vs_unbound_by_score_{tag}.tsv"), sep="\t", index=False, float_format="%.4g")
    print("\n", h5, flush=True)

    # figure: per sample, (a) state fractions by site set, (b) pred_ctcf by state at bound sites,
    # (c) per-site footprint - open difference for ctcf vs atac, (d) site mean pred_ctcf vs this line's peak signal
    samples = list(perfib)
    fig, ax = plt.subplots(len(samples), 4, figsize=(18, 4.2 * len(samples)), squeeze=False)
    for i, sample in enumerate(samples):
        df = perfib[sample]
        cell = sample.split("_")[0]
        sets = [s for s in ["bound", "bound_other_line", "unbound"] if s in set(df.site_set)]
        bottom = np.zeros(len(sets))
        for s in STATES + ["other"]:
            v = np.array([(df[df.site_set == x].state == s).mean() for x in sets])
            ax[i, 0].bar(sets, v, bottom=bottom, color=COLORS[s], label=s)
            bottom += v
        ax[i, 0].set_ylabel(f"{sample}\nfraction of fibers")
        ax[i, 0].set_title("fiber state at the motif")
        ax[i, 0].legend(fontsize=8)

        b = df[(df.site_set == "bound") & df.state.isin(STATES)]
        ax[i, 1].boxplot([b[b.state == s].pred_ctcf for s in STATES], labels=STATES, showfliers=False)
        ax[i, 1].set_title("bound sites: per-fiber pred CTCF")

        sm = b.groupby(["target", "state"])[["pred_ctcf", "pred_atac"]].agg(["mean", "size"])
        means = sm.xs("mean", axis=1, level=1)[sm.xs("size", axis=1, level=1).iloc[:, 0] >= MIN_PER_CLASS]
        dc, da = paired(means, "footprint", "open", "pred_ctcf"), paired(means, "footprint", "open", "pred_atac")
        lim = np.nanmax(np.abs(np.r_[dc.values, da.values, [0.1]]))
        bins = np.linspace(-lim, lim, 31)
        ax[i, 2].hist(dc, bins=bins, alpha=0.6, color="#D9534F", label=f"CTCF (median {dc.median():+.3f})")
        ax[i, 2].hist(da, bins=bins, alpha=0.6, color="#4C9BE8", label=f"ATAC (median {da.median():+.3f})")
        ax[i, 2].axvline(0, color="k", lw=0.8)
        ax[i, 2].set_title(f"bound sites: footprint - open, per site (n={len(dc)})")
        ax[i, 2].legend(fontsize=8)

        site = df.groupby(["target", "site_set"]).agg(pred=("pred_ctcf", "mean"), sig=(cell, "first")).reset_index()
        for s, c in [("unbound", "#8c8c8c"), ("bound_other_line", "#F0AD4E"), ("bound", "#D9534F")]:
            g = site[site.site_set == s]
            ax[i, 3].scatter(np.log1p(g.sig), g.pred, s=8, alpha=0.6, color=c, label=s)
        g = site[site.site_set == "bound"]
        r = spearman(g.sig, g.pred) if len(g) > 3 else np.nan
        ax[i, 3].set_xlabel(f"log1p {cell} CTCF peak signalValue")
        ax[i, 3].set_ylabel("site mean pred CTCF")
        ax[i, 3].set_title(f"site level (bound: Spearman {r:.2f})")
        ax[i, 3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"ctcf_footprints_{tag}.png"), dpi=130)
    print(f"\nwrote {OUT}/ctcf_footprints_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
