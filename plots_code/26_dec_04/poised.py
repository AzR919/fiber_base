"""Poised vs Polycomb on single molecules (T5g; reports/TODO_extended.md, defined 27 Nov, fixed before looking).

Bulk ChIP calls some regions poised (H3K4me3 and H3K27me3) and others Polycomb (H3K27me3 only). What do the fibers in
each class look like, is bulk "poised" a mix of open and closed molecules, and does the model separate the classes,
with per-fiber pred H3K4me3 / H3K27me3 following each fiber's own state?

Definitions:
  tiles      data_prep/build_targets.py --sets poised: 1 kb tiles on chr21, classed per cell line from the mean FC of
             its bulk H3K4me3 / H3K27me3: marked >= 2, unmarked < 1; poised, polycomb, active, neither. Robustness:
             the same with marked >= 3 (re-classed here from the k4 / k27 columns).
  fiber      methylated: CpG methylation over the tile > 0.5 from >= 3 covered CpGs (< 0.5 unmethylated);
             open: a large MSP (run >= 151 bp) inside the tile (lmsp_t > 0). Groups UO, UC, MO, MC (Unmethylated /
             Methylated, Open / Closed).
D1  genome fractions per cell and class (all autosomes, bulk only), and the share of each class at CpG islands
D2  fiber states per class (data; the fibers are the same for every model): fraction methylated, open, per group;
    per-tile open fraction (tiles with >= 10 fibers): median and the share of tiles in [0.2, 0.8] (a mix of molecules)
M1  tile-mean predictions (tiles with >= 5 fibers): AUC between classes, main readout poised vs polycomb on pred
    H3K4me3; Spearman of tile-mean pred vs bulk (asinh FC) over the target tiles
M2  within each class, per tile: pred(open) - pred(closed) among unmethylated fibers, pred(meth) - pred(unmeth) among
    closed fibers; >= 3 fibers per group; median over tiles, Wilcoxon
M3  per tile (>= 10 fibers), Spearman across fibers of pred H3K4me3 vs pred H3K27me3; median over tiles
C   in-silico CpG flips over the tile +- 100 bp: meth (unmethylated fibers), unmeth (methylated fibers), meth_distal
    (1.5 kb away); per tile the median per-fiber change, Wilcoxon over tiles; H3K27me3, H3K4me3, ATAC

    python plots_code/26_dec_04/poised.py
Writes plots/26_dec_04/poised/: poised_D1.tsv, poised_D2.tsv, poised_D2_tiles.tsv, poised_M1.tsv, poised_M2.tsv,
poised_M3.tsv, poised_C.tsv, poised_summary.tsv and figures; prints the tables (log_poised.txt via tee).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_13"))
from abl_common import run_name  # noqa: E402
from sm_stats import wilcoxon_p, spearman  # noqa: E402

OUT = "plots/26_dec_04/poised"
TARGETS = "poised_tiles_chr21"
FRACTIONS = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/regions/targets/poised_fractions_chr21.tsv"
CLASSES = ["active", "poised", "polycomb", "neither"]
GROUPS = ["UO", "UC", "MO", "MC"]
MIN_CPG, MIN_PER_GROUP, MIN_TILE, MIN_TILE_M3 = 3, 3, 5, 10
PAIRS = {"A": ["GM12878", "HepG2_200U"], "B": ["K562_200U", "H1_200U"]}
# label, abl_common key, pair, seeds
RUNS = [("DEV4NF", "noFIRE", "A", [919, 2]), ("DEV4NF", "noFIRE", "B", [919, 2]),
        ("R1a", "mixR1a", "A", [919, 2, 3]), ("R1a", "mixR1aB", "B", [919, 2, 3]),
        ("Pair", "pairNF", "A", [919, 2, 3]), ("Pair", "pairNFB", "B", [919, 2, 3]),
        ("MLP", "mlpA", "A", [919, 2, 3]), ("MLP", "mlpB", "B", [919, 2, 3]),
        ("Rules", "rulesA", "A", [919]), ("Rules", "rulesB", "B", [919])]
ORDER = ["Rules", "MLP", "R1a", "Pair", "DEV4NF"]
COL = {"Rules": "#9e9e9e", "MLP": "#4a90d9", "R1a": "#eb6834", "Pair": "#1baf7a", "DEV4NF": "#606060"}
CLS_COL = {"active": "#d62728", "poised": "#9467bd", "polycomb": "#1f77b4", "neither": "#bdbdbd"}
AUC_PAIRS = [("h3k4me3", "active", "poised"), ("h3k4me3", "poised", "polycomb"), ("h3k4me3", "active", "polycomb"),
             ("h3k27me3", "poised", "active"), ("h3k27me3", "polycomb", "active"), ("h3k27me3", "poised", "polycomb")]
FLIP_MODES = ["meth", "unmeth", "meth_distal"]


def cell_of(sample):
    return sample.split("_")[0]


def mol_dir(key, seed, sample):
    """Newest poised molecule dir of a run that holds fibers_<sample>.tsv.gz."""
    dirs = sorted(glob.glob(f"results/*_molecules_{run_name(key, seed)}_{TARGETS}"), reverse=True)
    for d in dirs:
        if os.path.exists(os.path.join(d, f"fibers_{sample}.tsv.gz")):
            return d
    raise FileNotFoundError(f"no {TARGETS} molecules for {key} seed {seed} sample {sample}")


def load(d, sample, marked=2.0):
    df = pd.read_csv(os.path.join(d, f"fibers_{sample}.tsv.gz"), sep="\t")
    cell = cell_of(sample)
    if marked == 2.0:
        df["cls"] = df[f"class_{cell}"]
    else:
        k4, k27 = df[f"k4_{cell}"], df[f"k27_{cell}"]
        df["cls"] = np.select([(k4 >= marked) & (k27 >= marked), (k4 < 1) & (k27 >= marked),
                               (k4 >= marked) & (k27 < 1), (k4 < 1) & (k27 < 1)],
                              ["poised", "polycomb", "active", "neither"], default="mid")
    df = df[df.cls.isin(CLASSES)].copy()
    df["meth"] = np.where(df.cpg_n_t >= MIN_CPG, df.cpg_meth_t, np.nan)
    df["mclass"] = np.select([df.meth > 0.5, df.meth < 0.5], ["M", "U"], default="")
    df["open"] = df.lmsp_t > 0
    df["group"] = np.where(df.mclass != "", df.mclass + np.where(df.open, "O", "C"), "")
    df["bulk_k4"], df["bulk_k27"] = np.arcsinh(df[f"k4_{cell}"]), np.arcsinh(df[f"k27_{cell}"])
    return df


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), from ranks."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = pd.Series(np.r_[pos, neg]).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def summ(x):
    x = pd.Series(x).dropna()
    return {"n_tiles": len(x), "median": x.median() if len(x) else np.nan,
            "frac_pos": (x > 0).mean() if len(x) else np.nan, "p": wilcoxon_p(x) if len(x) >= 5 else np.nan}


# ---------------------------------------------------------------- D: data
def part_d1():
    if not os.path.exists(FRACTIONS):
        print(f"  {FRACTIONS} missing; D1 skipped", flush=True)
        return pd.DataFrame()
    fr = pd.read_csv(FRACTIONS, sep="\t")
    tot = fr.groupby(["cell", "class"])[["n_tiles", "n_cgi", "n_valid", "n_valid_cgi"]].sum().reset_index()
    tot["genome_frac"] = tot.n_tiles / tot.n_valid
    tot["frac_at_cgi"] = tot.n_cgi / tot.n_tiles.where(tot.n_tiles > 0)
    tot["cgi_frac_in_class"] = tot.n_cgi / tot.n_valid_cgi
    c21 = fr[fr.chrom == "chr21"].set_index(["cell", "class"])
    tot["chr21_frac"] = [c21.loc[(c, k), "n_tiles"] / c21.loc[(c, k), "n_valid"] if (c, k) in c21.index else np.nan
                         for c, k in zip(tot.cell, tot["class"])]
    return tot


def part_d2(tables, tag=""):
    rows, tiles = [], []
    for sample, df in tables.items():
        for cls in CLASSES:
            s = df[df.cls == cls]
            called = s[s.mclass != ""]
            r = {"sample": sample, "class": cls, "set": tag, "tiles": s.target.nunique(), "fibers": len(s),
                 "frac_meth": (called.mclass == "M").mean() if len(called) else np.nan,
                 "frac_open": s.open.mean() if len(s) else np.nan,
                 "tiles_cgi": s.groupby("target").cgi.first().mean() if len(s) else np.nan}
            for g in GROUPS:
                r[f"frac_{g}"] = (called.group == g).mean() if len(called) else np.nan
            per = s.groupby("target").agg(open=("open", "mean"), n=("open", "size"),
                                          meth=("meth", "mean"), cgi=("cgi", "first"))
            per = per[per.n >= MIN_TILE_M3]
            r["tile_open_median"] = per.open.median() if len(per) else np.nan
            r["tile_open_mixed"] = ((per.open >= 0.2) & (per.open <= 0.8)).mean() if len(per) else np.nan
            r["tile_open_sd"] = per.open.std() if len(per) > 1 else np.nan
            rows.append(r)
            tiles.append(per.assign(sample=sample, cls=cls).reset_index())
    return pd.DataFrame(rows), pd.concat(tiles, ignore_index=True)


# ---------------------------------------------------------------- M: model on real fibers
def part_m(df, label, seed, sample):
    m1, m2, m3 = [], [], []
    tm = df.groupby("target").agg(cls=("cls", "first"), n=("read", "size"), k4=("pred_h3k4me3", "mean"),
                                  k27=("pred_h3k27me3", "mean"), bulk_k4=("bulk_k4", "first"),
                                  bulk_k27=("bulk_k27", "first"))
    tm = tm[tm.n >= MIN_TILE]
    base = {"model": label, "seed": seed, "sample": sample}
    for assay, a, b in AUC_PAIRS:
        col = "k4" if assay == "h3k4me3" else "k27"
        m1.append({**base, "readout": f"auc_{assay}_{a}>{b}",
                   "value": auc(tm[tm.cls == a][col], tm[tm.cls == b][col])})
    for col, bulk in [("k4", "bulk_k4"), ("k27", "bulk_k27")]:
        m1.append({**base, "readout": f"spearman_{col}_vs_bulk", "value": spearman(tm[col], tm[bulk])})
    for cls in CLASSES:
        for col in ["k4", "k27"]:
            m1.append({**base, "readout": f"mean_{col}_{cls}", "value": tm[tm.cls == cls][col].mean()})
    for cls in CLASSES:
        s = df[(df.cls == cls) & (df.group != "")]
        for assay in ["h3k4me3", "h3k27me3", "atac"]:
            g = s.groupby(["target", "group"])[f"pred_{assay}"].agg(["mean", "size"]).unstack("group")

            def diff(x, y):
                if ("size", x) not in g or ("size", y) not in g:
                    return pd.Series(dtype=float)
                ok = (g[("size", x)] >= MIN_PER_GROUP) & (g[("size", y)] >= MIN_PER_GROUP)
                return (g[("mean", x)] - g[("mean", y)])[ok]
            for contrast, (x, y) in {"open-closed|unmeth": ("UO", "UC"), "meth-unmeth|closed": ("MC", "UC"),
                                     "meth-unmeth|open": ("MO", "UO")}.items():
                m2.append({**base, "class": cls, "assay": assay, "contrast": contrast, **summ(diff(x, y))})
        t = df[df.cls == cls].groupby("target")
        rho = [spearman(g.pred_h3k4me3, g.pred_h3k27me3) for _, g in t if len(g) >= MIN_TILE_M3]
        rho = pd.Series(rho).dropna()
        m3.append({**base, "class": cls, "n_tiles": len(rho), "median_rho": rho.median() if len(rho) else np.nan,
                   "frac_neg": (rho < 0).mean() if len(rho) else np.nan})
    return m1, m2, m3


def part_c(df, label, seed, sample):
    rows = []
    if f"nflip_{FLIP_MODES[0]}" not in df:
        return rows
    for cls in CLASSES:
        s = df[df.cls == cls]
        for mode in FLIP_MODES:
            if f"nflip_{mode}" not in s:
                continue
            group = "U" if mode.startswith("meth") else "M"
            g = s[(s.mclass == group) & (s[f"nflip_{mode}"] > 0)]
            for assay in ["h3k27me3", "h3k4me3", "atac"]:
                g = g.assign(delta=g[f"pred_{assay}__{mode}"] - g[f"pred_{assay}"])
                per = g.groupby("target").delta.agg(["median", "size"])
                per = per[per["size"] >= MIN_PER_GROUP]["median"]
                r = summ(per)
                rows.append({"model": label, "seed": seed, "sample": sample, "class": cls, "mode": mode,
                             "assay": assay, "fibers": len(g), **r})
    return rows


# ---------------------------------------------------------------- summary across seeds
def seed_summary(df, keys, value):
    g = df.groupby(keys)[value]
    out = g.agg(["mean", "std", "size"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["size"])
    return out


def figures(d1, d2, m1, m2, m3, c):
    samples = [s for p in PAIRS.values() for s in p]
    # D: genome fractions and fiber groups per class
    fig, axes = plt.subplots(1, 1 + len(samples), figsize=(4 + 3.2 * len(samples), 3.6))
    if len(d1):
        cells = [cell_of(s) for s in samples]
        x = np.arange(len(cells))
        for i, cls in enumerate(["poised", "polycomb", "active"]):
            v = [d1[(d1.cell == c) & (d1["class"] == cls)].genome_frac.sum() * 100 for c in cells]
            axes[0].bar(x + (i - 1) * 0.27, v, 0.27, color=CLS_COL[cls], label=cls)
        axes[0].set_xticks(x, cells, rotation=30)
        axes[0].set_ylabel("% of autosomal 1 kb tiles")
        axes[0].set_title("D1 genome fractions")
        axes[0].legend(fontsize=7)
    gcol = {"UO": "#f28e2b", "UC": "#4e79a7", "MO": "#edc948", "MC": "#59595c"}
    for ax, s in zip(axes[1:], samples):
        sub = d2[d2["sample"] == s].set_index("class").reindex(CLASSES)
        bottom = np.zeros(len(CLASSES))
        for g in GROUPS:
            v = sub[f"frac_{g}"].fillna(0).values
            ax.bar(CLASSES, v, bottom=bottom, color=gcol[g], label=g)
            bottom += v
        ax.set_title(f"D2 {cell_of(s)}")
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(fontsize=7, title="fiber", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/poised_D.png", dpi=150)
    plt.close(fig)

    # M1: AUCs per model and cell
    reads = [f"auc_{a}_{x}>{y}" for a, x, y in AUC_PAIRS]
    fig, axes = plt.subplots(len(samples), 1, figsize=(11, 2.4 * len(samples)), sharex=True)
    for ax, s in zip(axes, samples):
        sub = m1[(m1["sample"] == s) & m1.readout.isin(reads)]
        for j, lab in enumerate(ORDER):
            q = sub[sub.model == lab]
            for i, r in enumerate(reads):
                v = q[q.readout == r].value
                ax.scatter(np.full(len(v), i + (j - 2) * 0.13), v, color=COL[lab], s=14,
                           label=lab if i == 0 and len(v) else None)
        ax.axhline(0.5, color="k", lw=0.6, ls="--")
        ax.set_ylabel(f"{cell_of(s)}\nAUC")
        ax.set_ylim(0.2, 1.0)
    axes[0].legend(fontsize=7, ncol=5)
    axes[-1].set_xticks(range(len(reads)), [r.replace("auc_", "").replace("h3k", "K") for r in reads], rotation=20)
    fig.suptitle("M1: tile-mean predictions separate the bulk classes")
    fig.tight_layout()
    fig.savefig(f"{OUT}/poised_M1.png", dpi=150)
    plt.close(fig)

    # M2: open - closed (H3K4me3) and meth - unmeth (H3K27me3) at poised / polycomb tiles
    panels = [("h3k4me3", "open-closed|unmeth"), ("h3k27me3", "open-closed|unmeth"),
              ("h3k27me3", "meth-unmeth|closed"), ("h3k4me3", "meth-unmeth|closed")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.4 * len(panels)), sharex=True)
    xt = [(s, cls) for s in samples for cls in ["poised", "polycomb"]]
    for ax, (assay, con) in zip(axes, panels):
        sub = m2[(m2.assay == assay) & (m2.contrast == con)]
        for j, lab in enumerate(ORDER):
            for i, (s, cls) in enumerate(xt):
                v = sub[(sub.model == lab) & (sub["sample"] == s) & (sub["class"] == cls)]["median"]
                ax.scatter(np.full(len(v), i + (j - 2) * 0.13), v, color=COL[lab], s=14,
                           label=lab if i == 0 and len(v) else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"{assay}\n{con}", fontsize=8)
    axes[0].legend(fontsize=7, ncol=5)
    axes[-1].set_xticks(range(len(xt)), [f"{cell_of(s)}\n{c}" for s, c in xt], fontsize=7)
    fig.suptitle("M2: per-tile fiber-group differences (median over tiles)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/poised_M2.png", dpi=150)
    plt.close(fig)

    # M3 and C
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    xt = [(s, cls) for s in samples for cls in CLASSES[:3]]
    for j, lab in enumerate(ORDER):
        for i, (s, cls) in enumerate(xt):
            v = m3[(m3.model == lab) & (m3["sample"] == s) & (m3["class"] == cls)].median_rho
            axes[0].scatter(np.full(len(v), i + (j - 2) * 0.13), v, color=COL[lab], s=12,
                            label=lab if i == 0 and len(v) else None)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(range(len(xt)), [f"{cell_of(s)[:4]} {c[:4]}" for s, c in xt], rotation=60, fontsize=7)
    axes[0].set_title("M3: per-tile Spearman, pred H3K4me3 vs H3K27me3")
    axes[0].legend(fontsize=7)
    if len(c):
        xt = [(s, cls, m) for s in samples for cls in ["poised", "polycomb"] for m in FLIP_MODES]
        sub = c[c.assay == "h3k27me3"]
        for j, lab in enumerate(ORDER):
            for i, (s, cls, m) in enumerate(xt):
                v = sub[(sub.model == lab) & (sub["sample"] == s) & (sub["class"] == cls) & (sub["mode"] == m)]["median"]
                axes[1].scatter(np.full(len(v), i + (j - 2) * 0.13), v, color=COL[lab], s=12)
        axes[1].axhline(0, color="k", lw=0.6)
        axes[1].set_xticks(range(len(xt)), [f"{cell_of(s)[:4]} {c[:4]} {m}" for s, c, m in xt], rotation=70,
                           fontsize=6)
        axes[1].set_title("C: CpG flips, change of pred H3K27me3")
    fig.tight_layout()
    fig.savefig(f"{OUT}/poised_M3_C.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    d1 = part_d1()
    # D2 from DEV4NF seed 919 (the fibers are the same for every model)
    tables = {s: load(mol_dir("noFIRE", 919, s), s) for p in PAIRS.values() for s in p}
    d2, d2t = part_d2(tables, "FC>=2")
    tables3 = {s: load(mol_dir("noFIRE", 919, s), s, marked=3.0) for p in PAIRS.values() for s in p}
    d2b, _ = part_d2(tables3, "FC>=3")
    d2 = pd.concat([d2, d2b], ignore_index=True)
    m1, m2, m3, c = [], [], [], []
    for label, key, pair, seeds in RUNS:
        for seed in seeds:
            for sample in PAIRS[pair]:
                try:
                    d = mol_dir(key, seed, sample)
                except FileNotFoundError as e:
                    print(f"  missing: {e}", flush=True)
                    continue
                df = load(d, sample)
                a, b, cc = part_m(df, label, seed, sample)
                m1 += a; m2 += b; m3 += cc
                c += part_c(df, label, seed, sample)
                df3 = load(d, sample, marked=3.0)
                a3, _, _ = part_m(df3, label, seed, sample)
                m1 += [{**r, "readout": r["readout"] + "@FC3"} for r in a3]
    m1, m2, m3, c = map(pd.DataFrame, (m1, m2, m3, c))
    for name, t in [("D1", d1), ("D2", d2), ("D2_tiles", d2t), ("M1", m1), ("M2", m2), ("M3", m3), ("C", c)]:
        t.to_csv(f"{OUT}/poised_{name}.tsv", sep="\t", index=False)

    s1 = seed_summary(m1, ["model", "sample", "readout"], "value")
    s2 = seed_summary(m2, ["model", "sample", "class", "assay", "contrast"], "median")
    s3 = seed_summary(m3, ["model", "sample", "class"], "median_rho")
    sc = seed_summary(c, ["model", "sample", "class", "mode", "assay"], "median") if len(c) else pd.DataFrame()
    pd.concat([s1.assign(part="M1"), s2.assign(part="M2"), s3.assign(part="M3"), sc.assign(part="C")],
              ignore_index=True).to_csv(f"{OUT}/poised_summary.tsv", sep="\t", index=False)

    print("\n=== D1 genome fractions (autosomes; FC >= 2 marked, < 1 unmarked) ===")
    if len(d1):
        print(d1.pivot(index="cell", columns="class", values="genome_frac").round(4))
        print("share of the class at CpG islands:")
        print(d1.pivot(index="cell", columns="class", values="frac_at_cgi").round(3))
    print("\n=== D2 fiber states per class (DEV4NF seed 919 fibers) ===")
    print(d2.round(3).to_string(index=False))
    print("\n=== M1 (mean over seeds) ===")
    print(s1[~s1.readout.str.startswith("mean_")].pivot_table(index=["sample", "readout"], columns="model",
                                                                values="mean").round(3))
    print("\n=== M1 tile means of pred (mean over seeds) ===")
    print(s1[s1.readout.str.startswith("mean_")].pivot_table(index=["sample", "readout"], columns="model",
                                                               values="mean").round(3))
    print("\n=== M2 (median over tiles, mean over seeds) ===")
    print(s2.pivot_table(index=["sample", "class", "assay", "contrast"], columns="model", values="mean").round(3))
    print("\n=== M3 (median per-tile rho, mean over seeds) ===")
    print(s3.pivot_table(index=["sample", "class"], columns="model", values="mean").round(3))
    if len(sc):
        print("\n=== C flips (median per-tile change, mean over seeds) ===")
        print(sc.pivot_table(index=["sample", "class", "mode", "assay"], columns="model", values="mean").round(3))
    figures(d1, d2[d2.set == "FC>=2"], m1, m2, m3, c)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
