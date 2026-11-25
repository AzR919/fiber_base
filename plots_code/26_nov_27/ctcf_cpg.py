"""CTCF x CpG methylation on single molecules (T5f; reports/TODO.md section 3, defined 25 Nov). CTCF binding is
blocked when its motif's CpGs are methylated (e.g. the H19/IGF2 ICR). Do single fibers show it, and did the model,
trained on bulk ChIP only, learn that methylation lowers CTCF on a molecule? In the data the link goes both ways
(methylation blocks binding; bound CTCF keeps its site unmethylated), so A and B are associations; C (in-silico flips)
tests the direction inside the model.

Definitions (fixed before looking):
  sites      the CTCF footprint test's chr21 sets, hole sites excluded: bound (a peak in this cell), bound_other_line
             (a peak in another dev line only), unbound (motif hits away from every peak, score-matched)
  fiber methylation   methylated CpGs / covered CpGs over motif +- 100 bp (cpg_*_t, _l, _r), >= 3 CpGs covered;
             methylated > 0.5, unmethylated < 0.5
  fiber state         as ctcf_footprints.py (footprint, open, nucleosome); accessible = footprint or open
  per-site tests      difference between the two groups at each site with >= 3 fibers in each, Wilcoxon over sites
A  data (no model; one table per cell, the fibers are the same for every model)
   A1 at bound sites, among accessible fibers, footprints are rarer on methylated fibers (per-site difference of
      footprint fractions; Mantel-Haenszel odds ratio over sites, expected < 1)
   A2 bound sites are less methylated than unbound ones at the same motif score; across bound sites, more
      methylation goes with fewer footprints (Spearman)
   A3 how many accessible fibers are methylated (power)
B  model on real fibers, bound and bound_other_line sites, accessible fibers
   B1 pred CTCF methylated - unmethylated < 0 (per site)
   B2 the same within footprint fibers only and within open fibers only (does the model use CpG beyond the footprint?)
   B3 control: B1 / B2 for DEV4 and its no-CpG ablation (model19; GM12878, K562, 2 seeds): what is left without CpG
      comes through m6A alone
C  in-silico flips (molecule_predict.py --perturb), bound sites, accessible fibers: methylate the covered CpGs over
   motif +- 100 bp on unmethylated fibers (meth), unmethylate them on methylated fibers (unmeth), and the same at a
   region 1.5 kb away (_distal). Readout: per-fiber change of pred CTCF over the motif, median per site, Wilcoxon over
   sites; ATAC and H3K4me3 as specificity; unbound sites for meth. Expected if the model learned the biology:
   meth < 0, unmeth > 0, both larger than the distal flips.

    python plots_code/26_nov_27/ctcf_cpg.py
Writes plots/26_nov_27/ctcf_cpg/: cpg_A_counts.tsv, cpg_A1.tsv, cpg_A2.tsv, cpg_B.tsv, cpg_C.tsv, cpg_A.png,
cpg_B.png, cpg_C.png; prints the tables (log_ctcf_cpg.txt via tee).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_13"))
from abl_common import run_name, molecule_dir  # noqa: E402
from sm_stats import wilcoxon_p, mannwhitney_greater_p, spearman  # noqa: E402
from ctcf_footprints import fiber_state  # noqa: E402

OUT = "plots/26_nov_27/ctcf_cpg"
HOLES = "plots/26_nov_20/ctcf_noholes/holes.txt"
MIN_CPG, MIN_PER_GROUP = 3, 3
CELLS = {"A": ["GM12878", "HepG2_200U"], "B": ["K562_200U", "H1_200U"]}
# label -> (abl_common key, pair) ; seeds
RUNS = [("DEV4NF", "noFIRE", "AB", [919, 2]), ("R1a", "mixR1a", "A", [919, 2, 3]), ("R1a", "mixR1aB", "B", [919, 2, 3]),
        ("Pair", "pairNF", "A", [919, 2, 3]), ("Pair", "pairNFB", "B", [919, 2, 3])]
CONTROLS = [("DEV4", "DEV4", [919, 2]), ("noCpG", "noCpG", [919, 2])]
COL = {"DEV4NF": "#606060", "R1a": "#eb6834", "Pair": "#1baf7a", "DEV4": "#8c6bb1", "noCpG": "#c9a227"}
MODES = ["meth", "unmeth", "meth_distal", "unmeth_distal"]
HOLE_SET = set(open(HOLES).read().split())


def perturb_dir(key, seed):
    """Newest CTCF molecule dir of a run made with --perturb."""
    dirs = sorted(glob.glob(f"results/*_molecules_{run_name(key, seed)}_ctcf_sites_chr21"), reverse=True)
    for d in dirs:
        try:
            if json.load(open(os.path.join(d, "run.json")))["args"].get("perturb"):
                return d
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(f"no --perturb CTCF run for {key} seed {seed}")


def load(d, sample):
    df = pd.read_csv(os.path.join(d, f"fibers_{sample}.tsv.gz"), sep="\t")
    df = df[~df.target.isin(HOLE_SET)].copy()
    df["state"] = fiber_state(df)
    cell = sample.split("_")[0]
    df["site_set"] = np.where(df["set"] == "unbound", "unbound", np.where(df[cell] > 0, "bound", "bound_other_line"))
    n = df[["cpg_n_t", "cpg_n_l", "cpg_n_r"]].sum(1)
    m = sum(df[f"cpg_meth_{x}"].fillna(0) * df[f"cpg_n_{x}"] for x in "tlr")
    df["meth"] = np.where(n >= MIN_CPG, m / n.where(n > 0), np.nan)
    df["mclass"] = np.select([df.meth > 0.5, df.meth < 0.5], ["meth", "unmeth"], default="")
    df.loc[df.meth.isna(), "mclass"] = ""
    df["acc"] = df.state.isin(["footprint", "open"])
    return df


def per_site_diff(df, value, sets=("bound",), min_n=MIN_PER_GROUP):
    """Per site: mean(value | methylated) - mean(value | unmethylated) over sites with >= min_n fibers in each."""
    s = df[df.site_set.isin(sets) & (df.mclass != "")]
    g = s.groupby(["target", "mclass"])[value].agg(["mean", "size"]).unstack("mclass")
    if ("size", "meth") not in g or ("size", "unmeth") not in g:
        return pd.Series(dtype=float)
    ok = (g[("size", "meth")] >= min_n) & (g[("size", "unmeth")] >= min_n)
    return (g[("mean", "meth")] - g[("mean", "unmeth")])[ok]


def summarize(d):
    d = d.dropna()
    return {"n_sites": len(d), "median": d.median() if len(d) else np.nan,
            "frac_neg": (d < 0).mean() if len(d) else np.nan, "p": wilcoxon_p(d) if len(d) >= 5 else np.nan}


# ---------------------------------------------------------------- A: data
def part_a(tables):
    counts, a1, a2 = [], [], []
    for sample, df in tables.items():
        for st in ["bound", "bound_other_line", "unbound"]:
            s = df[df.site_set == st]
            acc = s[s.acc & (s.mclass != "")]
            g = acc.groupby(["target", "mclass"]).size().unstack(fill_value=0)
            n_both = int(((g.get("meth", 0) >= MIN_PER_GROUP) & (g.get("unmeth", 0) >= MIN_PER_GROUP)).sum()) \
                if len(g) else 0
            counts.append({"sample": sample, "site_set": st, "sites": s.target.nunique(), "fibers": len(s),
                           "with_meth_call": int((s.mclass != "").sum()),
                           "frac_methylated": float((s.mclass == "meth").sum() / max(1, (s.mclass != "").sum())),
                           "accessible": int(acc.shape[0]),
                           "accessible_frac_methylated": float((acc.mclass == "meth").mean()) if len(acc) else np.nan,
                           "sites_with_both_groups": n_both})
        acc = df[df.acc].copy()
        acc["fp"] = (acc.state == "footprint").astype(float)
        for sets, lab in [(("bound",), "bound"), (("bound_other_line",), "bound_other_line"),
                          (("bound", "bound_other_line"), "bound + other")]:
            d = per_site_diff(acc, "fp", sets)
            # Mantel-Haenszel odds ratio of footprint (vs open) for methylated vs unmethylated, strata = sites
            s = acc[acc.site_set.isin(sets) & (acc.mclass != "")]
            num = den = 0.0
            for _, g in s.groupby("target"):
                a = ((g.mclass == "meth") & (g.fp == 1)).sum(); b = ((g.mclass == "meth") & (g.fp == 0)).sum()
                c = ((g.mclass == "unmeth") & (g.fp == 1)).sum(); e = ((g.mclass == "unmeth") & (g.fp == 0)).sum()
                n = a + b + c + e
                if (a + b) and (c + e):
                    num += a * e / n; den += b * c / n
            fp_m = s[s.mclass == "meth"].fp.mean(); fp_u = s[s.mclass == "unmeth"].fp.mean()
            a1.append({"sample": sample, "sites": lab, **summarize(d), "mh_odds_ratio": num / den if den else np.nan,
                       "pooled_fp_frac_meth": fp_m, "pooled_fp_frac_unmeth": fp_u,
                       "n_acc_meth": int((s.mclass == "meth").sum()), "n_acc_unmeth": int((s.mclass == "unmeth").sum())})
        # A2: site methylation, bound vs unbound by motif score; methylation vs footprint frequency over bound sites
        site = df[df.mclass != ""].groupby(["target", "site_set", "score"]).meth.mean().reset_index()
        site["bin"] = np.floor(site.score)
        bu = site[site.site_set.isin(["bound", "unbound"])]
        piv = bu.groupby(["bin", "site_set"]).meth.mean().unstack()
        both = piv.dropna()
        fpf = acc[acc.site_set == "bound"].groupby("target").fp.agg(["mean", "size"])
        fpf = fpf[fpf["size"] >= 5]["mean"]
        mb = site[site.site_set == "bound"].set_index("target").meth.reindex(fpf.index)
        a2.append({"sample": sample, "bound_site_meth_median": site[site.site_set == "bound"].meth.median(),
                   "unbound_site_meth_median": site[site.site_set == "unbound"].meth.median(),
                   "mwu_p_unbound_gt_bound": mannwhitney_greater_p(site[site.site_set == "unbound"].meth,
                                                                   site[site.site_set == "bound"].meth),
                   "score_bins": len(both), "bins_unbound_more_meth": int((both.unbound > both.bound).sum()),
                   "spearman_meth_vs_fp_bound": spearman(mb.values, fpf.values) if len(fpf) > 5 else np.nan,
                   "n_bound_sites_fp": len(fpf)})
    return pd.DataFrame(counts), pd.DataFrame(a1), pd.DataFrame(a2)


# ---------------------------------------------------------------- B: model on real fibers
def part_b(df, label, seed, sample):
    rows = []
    acc = df[df.acc]
    for sets, lab in [(("bound",), "bound"), (("bound", "bound_other_line"), "bound + other")]:
        for subset, sub in [("accessible", acc), ("footprint", acc[acc.state == "footprint"]),
                            ("open", acc[acc.state == "open"])]:
            for assay in ["ctcf", "atac", "h3k4me3"]:
                d = per_site_diff(sub, f"pred_{assay}", sets)
                rows.append({"model": label, "seed": seed, "sample": sample, "sites": lab, "fibers": subset,
                             "assay": assay, **summarize(d)})
    return rows


# ---------------------------------------------------------------- C: in-silico flips
def part_c(df, label, seed, sample):
    rows = []
    h1 = per_site_diff_state(df)
    for st in ["bound", "unbound"]:
        s = df[(df.site_set == st) & df.acc]
        for mode in MODES:
            group = "unmeth" if mode.startswith("meth") else "meth"
            g = s[(s.mclass == group) & (s[f"nflip_{mode}"] > 0)]
            for assay in ["ctcf", "atac", "h3k4me3"]:
                g = g.assign(delta=g[f"pred_{assay}__{mode}"] - g[f"pred_{assay}"],
                             delta_per_cpg=(g[f"pred_{assay}__{mode}"] - g[f"pred_{assay}"]) / g[f"nflip_{mode}"])
                per = g.groupby("target").agg(delta=("delta", "median"), dpc=("delta_per_cpg", "median"),
                                              n=("delta", "size"))
                per = per[per.n >= MIN_PER_GROUP]
                r = summarize(per.delta)
                r["frac_neg"] = r.pop("frac_neg")
                rows.append({"model": label, "seed": seed, "sample": sample, "sites": st, "mode": mode,
                             "assay": assay, "fibers": int(per.n.sum()), "mean_nflip": g[f"nflip_{mode}"].mean(),
                             **r, "median_per_cpg": per.dpc.median() if len(per) else np.nan,
                             "h1_footprint_effect": h1})
    return rows


def per_site_diff_state(df):
    """CTCF footprint effect at bound sites (median per-site footprint - open pred CTCF), the scale for C."""
    s = df[(df.site_set == "bound") & df.state.isin(["footprint", "open"])]
    g = s.groupby(["target", "state"]).pred_ctcf.agg(["mean", "size"]).unstack("state")
    ok = (g[("size", "footprint")] >= 3) & (g[("size", "open")] >= 3)
    return float((g[("mean", "footprint")] - g[("mean", "open")])[ok].median())


# ---------------------------------------------------------------- figures
def style(a, axis="y"):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis=axis, color="#e6e6e6", lw=0.6)
    a.set_axisbelow(True)


def fig_a(tables, a1):
    samples = list(tables)
    fig, ax = plt.subplots(1, len(samples), figsize=(3.6 * len(samples), 3.6), squeeze=False)
    for a, sample in zip(ax[0], samples):
        df = tables[sample]
        acc = df[df.acc & df.site_set.isin(["bound", "bound_other_line"]) & (df.mclass != "")].copy()
        acc["fp"] = (acc.state == "footprint").astype(float)
        g = acc.groupby(["site_set", "target", "mclass"]).fp.agg(["mean", "size"]).unstack("mclass")
        ok = (g[("size", "meth")] >= MIN_PER_GROUP) & (g[("size", "unmeth")] >= MIN_PER_GROUP)
        g = g[ok]
        for st, c in [("bound", "#D9534F"), ("bound_other_line", "#F0AD4E")]:
            if st in g.index.get_level_values(0):
                x = g.xs(st, level="site_set")
                a.scatter(x[("mean", "unmeth")], x[("mean", "meth")], s=14, color=c, alpha=0.8, label=st)
        a.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        r = a1[(a1["sample"] == sample) & (a1.sites == "bound + other")].iloc[0]
        a.set_title(f"{sample.split('_')[0]}: MH odds ratio {r.mh_odds_ratio:.2f}\n{r.n_sites} sites, p {r.p:.1e}",
                    fontsize=9)
        a.set_xlabel("footprint fraction, unmethylated fibers")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        style(a, "both")
    ax[0, 0].set_ylabel("footprint fraction, methylated fibers")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Data: accessible fibers at CTCF sites, methylated vs unmethylated (motif ± 100 bp), per site", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cpg_A.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_models(df, panels, fname, title, ylabel):
    """One panel per (filter dict, panel title); x = cells, dots = model x seed medians, filled = p < 0.01."""
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), squeeze=False)
    for a, (flt, ttl) in zip(ax[0], panels):
        s = df.copy()
        for k, v in flt.items():
            s = s[s[k] == v]
        cells = [c for p in CELLS.values() for c in p if c in set(s["sample"])]
        models = [m for m in COL if m in set(s.model)]
        w = 0.8 / max(1, len(models))
        for j, c in enumerate(cells):
            for i, m in enumerate(models):
                v = s[(s["sample"] == c) & (s.model == m)]
                x = j + (i - (len(models) - 1) / 2) * w
                for _, r in v.iterrows():
                    a.scatter(x, r["median"], s=22, zorder=3, color=COL[m] if r.p < 0.01 else "white",
                              edgecolor=COL[m], lw=1.2)
        a.axhline(0, color="#999999", lw=0.8)
        a.set_xticks(range(len(cells)))
        a.set_xticklabels([c.split("_")[0] for c in cells], fontsize=9)
        a.set_title(ttl, fontsize=9)
        style(a)
    ax[0, 0].set_ylabel(ylabel)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[m], label=m) for m in COL if m in set(df.model)],
               loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(COL), frameon=False)
    fig.suptitle(title + " (dots = seeds; filled = p < 0.01)", y=1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    fmt = lambda v: f"{v:.3g}"  # noqa: E731

    brows, crows, tables_a = [], [], {}
    for label, key, pairs, seeds in RUNS:
        for s in seeds:
            d = perturb_dir(key, s)
            print(f"{label} seed {s}: {d}", flush=True)
            for p in pairs:
                for sample in CELLS[p]:
                    df = load(d, sample)
                    if label == "DEV4NF" and s == 919:
                        tables_a[sample] = df
                    brows += part_b(df, label, s, sample)
                    crows += part_c(df, label, s, sample)
    for label, key, seeds in CONTROLS:
        for s in seeds:
            d = molecule_dir(key, "ctcf_sites_chr21", s)
            print(f"{label} seed {s}: {d}", flush=True)
            for sample in ["GM12878", "K562_200U"]:
                brows += part_b(load(d, sample), label, s, sample)

    counts, a1, a2 = part_a(tables_a)
    B, C = pd.DataFrame(brows), pd.DataFrame(crows)
    for name, t in [("cpg_A_counts", counts), ("cpg_A1", a1), ("cpg_A2", a2), ("cpg_B", B), ("cpg_C", C)]:
        t.to_csv(os.path.join(OUT, f"{name}.tsv"), sep="\t", index=False, float_format="%.4g")

    print("\n== A3: fibers and methylation (motif +- 100 bp, >= 3 CpGs)")
    print(counts.to_string(index=False, float_format=fmt))
    print("\n== A1: footprint fraction among accessible fibers, methylated - unmethylated (per site); MH odds ratio")
    print(a1.to_string(index=False, float_format=fmt))
    print("\n== A2: site methylation, bound vs unbound; methylation vs footprint frequency over bound sites")
    print(a2.to_string(index=False, float_format=fmt))
    print("\n== B: pred methylated - unmethylated (per-site median over sites), by model / seed")
    b = B[(B.sites == "bound + other")]
    print(b.pivot_table(index=["fibers", "assay", "model", "seed"], columns="sample", values="median",
                        sort=False).to_string(float_format=fmt))
    print("\n   p values (Wilcoxon over sites)")
    print(b[b.assay == "ctcf"].pivot_table(index=["fibers", "model", "seed"], columns="sample", values="p",
                                           sort=False).to_string(float_format=lambda v: f"{v:.1e}"))
    print("\n   sites per test")
    print(b[(b.assay == "ctcf")].pivot_table(index=["fibers", "model", "seed"], columns="sample", values="n_sites",
                                             sort=False).to_string())
    print("\n== C: in-silico flips, change of pred CTCF / ATAC / H3K4me3 over the motif (median over sites)")
    c = C[C.sites == "bound"]
    print(c.pivot_table(index=["mode", "assay", "model", "seed"], columns="sample", values="median",
                        sort=False).to_string(float_format=fmt))
    print("\n   pred CTCF, relative to the footprint effect (H1) of the same run")
    cc = c[c.assay == "ctcf"].assign(rel=lambda x: x["median"] / x.h1_footprint_effect)
    print(cc.pivot_table(index=["mode", "model", "seed"], columns="sample", values="rel",
                         sort=False).to_string(float_format=fmt))
    print("\n   p values, pred CTCF")
    print(c[c.assay == "ctcf"].pivot_table(index=["mode", "model", "seed"], columns="sample", values="p",
                                           sort=False).to_string(float_format=lambda v: f"{v:.1e}"))
    print("\n   unbound sites, pred CTCF")
    print(C[(C.sites == "unbound") & (C.assay == "ctcf")].pivot_table(
        index=["mode", "model", "seed"], columns="sample", values="median", sort=False).to_string(float_format=fmt))

    fig_a(tables_a, a1)
    bb = B[B.sites == "bound + other"]
    fig_models(bb[bb.assay == "ctcf"], [({"fibers": "accessible"}, "B1: accessible fibers"),
                                       ({"fibers": "footprint"}, "B2: footprint fibers only"),
                                       ({"fibers": "open"}, "B2: open fibers only")],
               "cpg_B.png", "Model on real fibers: pred CTCF, methylated − unmethylated, per site",
               "median over sites")
    cb = C[(C.sites == "bound")]
    fig_models(cb[cb.assay == "ctcf"], [({"mode": m}, f"{m}: Δ pred CTCF") for m in MODES],
               "cpg_C.png", "In-silico CpG flips at bound sites, accessible fibers", "median Δ over sites")
    fig_models(cb[cb.assay == "atac"], [({"mode": m}, f"{m}: Δ pred ATAC") for m in MODES],
               "cpg_C_atac.png", "In-silico CpG flips at bound sites: pred ATAC (specificity)", "median Δ over sites")
    print(f"wrote {OUT}/cpg_A.png, cpg_B.png, cpg_C.png, cpg_C_atac.png", flush=True)


if __name__ == "__main__":
    main()
