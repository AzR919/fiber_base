"""Input ablations of DEV4 (TODO_extended.md T10): bulk accuracy per assay and the pair offsets, DEV4 vs no CpG /
no FIRE / no accessibility (abl_common.MODELS). The single-molecule readouts (CTCF footprints, X inactivation) come
from ctcf_footprints.py / xi_promoters.py run with --model; part C collects their summaries.

Hypotheses (T10, fixed before the runs):
  no CpG    H3K4me3, H3K36me3, H3K9me3, CTCF lose more than ATAC, H3K27ac, H3K4me1; the H1 CTCF shift (models put
            H1 >= K562 at bins where K562's track is higher) shrinks if it comes from H1's higher CpG methylation.
  no FIRE   costs little (FIRE is derived from the m6A we already input; its classifier saw GM12878 DNase + CTCF
            ChIP), so CTCF and GM12878 are checked first.
  no acc.   (CpG + DNA only) accessibility carries most of the active marks; what is left shows what CpG + DNA give.
  m6aCpG    (m6A + CpG + DNA, no fibertools MSP / nucleosome / FIRE calls; added after the first three) the calls are
            derived from the m6A it still sees, so the cost should be small; vs no acc. it shows what raw m6A adds.

A  bulk: per-assay Pearson r, MSE and cell-type r (partial r given the cross-cell baseline) on all chr21 tiles per
   cell line (eval_summary.json), Delta vs DEV4; group means over the dev lines, the held-out three and Panc1.
   CpG-linked vs accessibility-linked assay groups for no CpG (Pearson r).
B  pair offsets: at the pair differential bins of pair_offsets.py (plots/26_nov_13/offsets/access_<pair>.tsv: true
   difference dt, accessibility difference dacc, DEV4's predicted difference), each ablation's predicted difference
   from its paired npz. Per pair x assay: mean predicted vs true difference, slope against dacc (pred / true).
C  molecules: key numbers from ctcf_summary_<model>.tsv and xi_summary_<model>.tsv where they exist.

    python plots_code/26_nov_13/ablations.py
Writes plots/26_nov_13/ablations/: bulk_{r,mse,ctr}.tsv, bulk_groups{,_mse,_ctr}.tsv, offsets.tsv, molecules.tsv,
ablation_bulk{,_mse,_ctr}.png, ablation_offsets.png.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_common import MODELS, NAMES  # noqa: E402

OUT = "plots/26_nov_13/ablations"
OFFSETS = "plots/26_nov_13/offsets"
MOL = "plots/26_nov_13/molecules"
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k4me1", "ctcf", "h3k36me3", "h3k27me3", "h3k9me3"]
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
GROUPS = {"dev": ["GM12878", "K562_200U", "HepG2_200U", "H1_200U"],
          "held-out": ["H9_200U", "HCT116_200U", "MCF_7_200U"],
          "Panc1": ["Panc1_200U"]}
CPG_LINKED, ACC_LINKED = ["h3k4me3", "h3k36me3", "h3k9me3", "ctcf"], ["atac", "h3k27ac", "h3k4me1"]
PAIRS = {"GM12878–HepG2": ("GM12878__HepG2", "eval24_pair_GM12878_HepG2_200U"),
         "K562–H1": ("K562__H1", "eval27_pair_K562_H1_200U")}
ABL = [m for m in MODELS if m != "DEV4"]
COL = {"true": "#333333", "DEV4": "#4C9BE8", "noCpG": "#D9534F", "noFIRE": "#F0AD4E", "noAcc": "#5CB85C",
       "m6aCpG": "#8E6CC0"}


def _partial(r):
    p = (r.get("partial") or {}).get("strata", {}).get("all", {}).get("per_assay", {})
    return {a: v.get("pearson", np.nan) for a, v in p.items()}


def _base(v, key):
    return v.get(key, np.nan) if isinstance(v, dict) else (v if key == "pearson" else np.nan)


# Bulk metrics. MSE = per_assay_loss (lower is better); cell-type r = partial correlation of prediction and target
# given the cross-cell baseline (0 for the baseline by definition, so no baseline value).
METRICS = {
    "r":   {"label": "Pearson r", "short": "r", "suffix": "", "higher_better": True, "fmt": "{:+.2f}", "cap": 0.4,
            "model": lambda r: r["per_assay_pearson"], "baseline": lambda v: _base(v, "pearson")},
    "mse": {"label": "MSE", "short": "MSE", "suffix": "_mse", "higher_better": False, "fmt": "{:+.3f}", "cap": None,
            "model": lambda r: r["per_assay_loss"], "baseline": lambda v: _base(v, "loss")},
    "ctr": {"label": "cell-type r", "short": "cell-type r", "suffix": "_ctr", "higher_better": True,
            "fmt": "{:+.2f}", "cap": 0.4, "model": _partial, "baseline": None},
}


def single_cell_results(run):
    """{cell: result dict} from every eval_summary.json of a run (single-cell evals only; newest dir wins)."""
    out, baseline = {}, {}
    for path in sorted(glob.glob(f"results/*_eval_{run}*/eval_summary.json")):
        for k, r in json.load(open(path))["results"].items():
            if "/" in k:
                continue
            out[k] = r
            if "baseline" in r:
                baseline[k] = r["baseline"].get("per_assay", {})
    return out, baseline


def part_a():
    results = {m: single_cell_results(run) for m, run in MODELS.items()}
    wides = {key: bulk_metric(key, spec, results) for key, spec in METRICS.items()}
    wide = wides["r"]

    # no CpG: CpG-linked vs accessibility-linked loss, per cell line (paired)
    print("\nno CpG: mean Delta r, CpG-linked (H3K4me3, H3K36me3, H3K9me3, CTCF) vs acc-linked (ATAC, H3K27ac, "
          "H3K4me1), per cell line", flush=True)
    for cell in [c for g in GROUPS.values() for c in g]:
        if cell not in wide.index.get_level_values("cell"):
            continue
        s = wide.xs(cell, level="cell")["d_noCpG"]
        c_, a_ = s.reindex(CPG_LINKED).mean(), s.reindex(ACC_LINKED).mean()
        print(f"  {cell:12s} CpG-linked {c_:+.3f}  acc-linked {a_:+.3f}  difference {c_ - a_:+.3f}", flush=True)
    return wide


def bulk_metric(key, spec, results):
    """One metric: per cell x assay table (bulk_<key>.tsv), group means and Delta vs DEV4 (bulk_groups<suffix>.tsv),
    figure ablation_bulk<suffix>.png."""
    rows = []
    for m, (res, base) in results.items():
        for cell, r in res.items():
            for a, v in spec["model"](r).items():
                rows.append({"model": m, "cell": cell, "assay": a, key: v})
        if m == "DEV4" and spec["baseline"]:
            for cell, b in base.items():
                for a, v in b.items():
                    rows.append({"model": "baseline", "cell": cell, "assay": a, key: spec["baseline"](v)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"bulk_{key}.tsv"), sep="\t", index=False, float_format="%.4f")
    wide = df.pivot_table(index=["cell", "assay"], columns="model", values=key)
    for m in ABL:
        wide[f"d_{m}"] = wide[m] - wide["DEV4"]

    g_rows = []
    for gname, cells in GROUPS.items():
        sub = wide[wide.index.get_level_values("cell").isin(cells)]
        for a in ASSAYS:
            s = sub.xs(a, level="assay") if a in sub.index.get_level_values("assay") else None
            if s is None or s.empty:
                continue
            g_rows.append({"group": gname, "assay": a, "n_cells": int(s["DEV4"].notna().sum()),
                           **{m: s[m].mean() for m in ["baseline", "DEV4"] + ABL if m in s},
                           **{f"d_{m}": s[f"d_{m}"].mean() for m in ABL}})
    groups = pd.DataFrame(g_rows)
    groups.to_csv(os.path.join(OUT, f"bulk_groups{spec['suffix']}.tsv"), sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_rows", 100, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(f"\nA. {spec['label']} by group (mean over the group's cell lines) and Delta vs DEV4\n", groups,
              flush=True)
    figure_bulk(wide, spec)
    return wide


def figure_bulk(wide, spec):
    cells = [c for g in GROUPS.values() for c in g if c in wide.index.get_level_values("cell")]
    # two rows: DEV4 top left; the ablations in a 2 x 2 block to its right, in abl_common order (top: fibertools
    # calls removed, bottom: a raw channel removed); bottom-left slot empty
    ncol = (len(ABL) + 1) // 2 + 1
    fig, grid = plt.subplots(2, ncol, figsize=(5.2 * ncol, 10.4))
    slots = [grid[0, 0]] + [grid[r, c] for r in range(2) for c in range(1, ncol)]
    for a in grid.flat:
        if not any(a is s for s in slots[:len(ABL) + 1]):
            a.axis("off")
    ax = slots[:len(ABL) + 1]
    M = wide["DEV4"].unstack("assay").reindex(index=cells, columns=ASSAYS)
    hb = spec["higher_better"]
    # DEV4 panel: brighter = better for every metric; Delta panels: red = worse than DEV4, blue = better
    vmax = 1 if hb else np.nanmax(M.values)
    im = ax[0].imshow(M.values, cmap="viridis" if hb else "viridis_r", vmin=0, vmax=vmax, aspect="auto")
    ax[0].set_title(f"DEV4 {spec['short']} (all chr21 tiles)")
    for (i, k), v in np.ndenumerate(M.values):
        if np.isfinite(v):
            ax[0].text(k, i, spec["fmt"].format(v).lstrip("+"), ha="center", va="center", fontsize=6.5,
                       color="w" if (v < 0.5 * vmax) == hb else "k")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    lim = np.nanmax(np.abs(np.concatenate([wide[f"d_{m}"].values for m in ABL])))
    lim = max(0.01 if spec["cap"] is None else 0.05, lim if spec["cap"] is None else min(lim, spec["cap"]))
    for j, m in enumerate(ABL, 1):
        D = wide[f"d_{m}"].unstack("assay").reindex(index=cells, columns=ASSAYS)
        im = ax[j].imshow(D.values, cmap="RdBu" if hb else "RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax[j].set_title(f"{NAMES[m]} − DEV4 (Δ {spec['short']})")
        for (i, k), v in np.ndenumerate(D.values):
            if np.isfinite(v):
                ax[j].text(k, i, spec["fmt"].format(v), ha="center", va="center", fontsize=6.5)
        plt.colorbar(im, ax=ax[j], fraction=0.046)
    for a in ax:
        a.set_xticks(range(len(ASSAYS)))
        a.set_xticklabels([LAB[x] for x in ASSAYS], rotation=60, ha="right", fontsize=8)
        a.set_yticks(range(len(cells)))
        a.set_yticklabels(cells, fontsize=8)
        a.axhline(len(GROUPS["dev"]) - 0.5, color="w", lw=2)
        a.axhline(len(GROUPS["dev"]) + len(GROUPS["held-out"]) - 0.5, color="w", lw=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"ablation_bulk{spec['suffix']}.png"), dpi=130)
    plt.close(fig)


def fit_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() > 5 else np.nan


def part_b():
    rows = []
    for pname, (bed, ev) in PAIRS.items():
        path = os.path.join(OFFSETS, f"access_{bed}.tsv")
        if not os.path.exists(path):
            print(f"B. missing {path}; run pair_offsets.py first", flush=True)
            continue
        acc = pd.read_csv(path, sep="\t")
        for m in ABL:
            z = np.load(sorted(glob.glob(f"results/*_eval_{MODELS[m]}*/{ev}_paired.npz"))[-1])
            assays = [str(x) for x in z["assays"]]
            k2i = {(c, int(s)): i for i, (c, s) in enumerate(zip(z["tile_chrom"].astype(str), z["tile_start"]))}
            idx = np.array([k2i.get((c, int(s)), -1) for c, s in zip(acc.chrom, acc.start)])
            k = np.array([assays.index(a) for a in acc.assay])
            acc[f"dp_{m}"] = np.where(idx >= 0, z["tile_dp_bin"][np.maximum(idx, 0), k], np.nan)
        for a, g in acc.groupby("assay"):
            ts = fit_slope(g.dacc.values, g.dt.values)
            row = {"pair": pname, "assay": a, "n": len(g), "true_mean": g.dt.mean(), "true_slope": ts}
            for m in ["DEV4"] + ABL:
                p = g[f"dp_{m}"].values
                row[f"{m}_mean"] = np.nanmean(p)
                row[f"{m}_slope_ratio"] = fit_slope(g.dacc.values, p) / ts
                row[f"{m}_r"] = pd.Series(p).corr(pd.Series(g.dt.values))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "offsets.tsv"), sep="\t", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:+.3f}".format):
        print("\nB. pair differential bins: mean difference (A - B), true vs predicted\n",
              df[["pair", "assay", "n", "true_mean"] + [f"{m}_mean" for m in ["DEV4"] + ABL]], flush=True)
        print("\n   slope against the accessibility difference, predicted / true\n",
              df[["pair", "assay"] + [f"{m}_slope_ratio" for m in ["DEV4"] + ABL]], flush=True)
        print("\n   r(predicted, true difference) over the bins\n",
              df[["pair", "assay"] + [f"{m}_r" for m in ["DEV4"] + ABL]], flush=True)
    figure_offsets(df)
    return df


def figure_offsets(df):
    pairs = list(dict.fromkeys(df.pair))
    fig, ax = plt.subplots(len(pairs), 1, figsize=(12, 3.8 * len(pairs)), squeeze=False)
    series = ["true"] + ["DEV4"] + ABL
    w = 0.8 / len(series)
    for i, p in enumerate(pairs):
        g = df[df.pair == p].set_index("assay").reindex(ASSAYS).dropna(how="all")
        x = np.arange(len(g))
        for j, s in enumerate(series):
            ax[i, 0].bar(x + (j - len(series) / 2 + 0.5) * w, g[f"{s}_mean"], w, color=COL[s],
                         label=NAMES.get(s, "true difference"))
        ax[i, 0].axhline(0, color="k", lw=0.8)
        ax[i, 0].set_xticks(x)
        ax[i, 0].set_xticklabels([LAB[a] for a in g.index])
        ax[i, 0].set_ylabel(f"{p}\nmean difference (asinh)")
        ax[i, 0].legend(fontsize=8, ncol=6)
    ax[0, 0].set_title("Pair differential bins: mean true vs predicted difference, DEV4 and its ablations")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_offsets.png"), dpi=130)


def part_c():
    rows = []
    for m in MODELS:
        p = os.path.join(MOL, f"ctcf_summary_{m}.tsv")
        if os.path.exists(p):
            s = pd.read_csv(p, sep="\t")
            for r in s[s.site_set.isin(["bound", "unbound"])].itertuples(index=False):
                rows.append({"test": "ctcf", "model": m, "sample": r.sample, "set": r.site_set,
                             "fp_minus_open_ctcf": r.d_fp_open_ctcf_median, "p_ctcf": r.d_fp_open_ctcf_p,
                             "fp_minus_open_atac": r.d_fp_open_atac_median,
                             "ctcf_minus_atac": r.d_ctcf_minus_atac_median, "p_spec": r.d_ctcf_minus_atac_p,
                             "open_minus_nuc_ctcf": r.d_open_nuc_ctcf_median})
        p = os.path.join(MOL, f"xi_summary_{m}.tsv")
        if os.path.exists(p):
            s = pd.read_csv(p, sep="\t")
            for r in s.itertuples(index=False):
                rows.append({"test": "xi", "model": m, "set": r.group, "column": r.column,
                             "median_d": r.median_d, "frac_expected_sign": r.frac_expected_sign,
                             "wilcoxon_p": r.wilcoxon_p, "n_genes": r.n_genes})
    if not rows:
        print("\nC. no molecule summaries yet", flush=True)
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(MOL, "..", "ablations", "molecules.tsv"), sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 30):
        c = df[df.test == "ctcf"].dropna(axis=1, how="all")
        if len(c):
            print("\nC. CTCF footprints (per-site medians, footprint - open)\n", c, flush=True)
        x = df[(df.test == "xi") & df.column.isin(["pred_atac", "pred_h3k4me3", "pred_h3k27ac", "pred_h3k27me3",
                                                    "pred_h3k9me3", "cpg_meth_t", "msp_t"])].dropna(axis=1, how="all")
        if len(x):
            print("\nC. X inactivation, median Xa - Xi per gene\n",
                  x.pivot_table(index=["set", "column"], columns="model", values="median_d"), flush=True)
            print("\n   fraction of genes with the expected sign\n",
                  x.pivot_table(index=["set", "column"], columns="model", values="frac_expected_sign"), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    part_a()
    part_b()
    part_c()
    print(f"\nwrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
