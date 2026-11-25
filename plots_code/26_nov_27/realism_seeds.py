"""Mixture realism test, seed noise (TODO_extended.md section 2b / 2g stage 2): R1a and the pair model on both dev
pairs x 3 training seeds (919, 2, 3), DEV4NF (seeds 919, 2) as the reference, R0 (seed 919, pair A only) for
context. Extends realism_t3.py and realism_bio.py (seed 919, pair A) to all runs.

Pairs (test chr21; the eval configs use their own seed, so every training seed is scored on the same fibers):
  A  GM12878 + HepG2: T3 eval24, library control eval25 (HepG2 100U vs 200U), mixture eval31 (0.5, balanced)
  B  K562 + H1:       T3 eval27, library controls eval28 (K562) and eval29 (H1), mixture eval32 (0.5, balanced)
     H1 has no ATAC track, so pair B means are over the 7 other assays.

Readouts per pair x model x seed (means over the pair's assays):
  T3: fiber-origin AUC, sign accuracy at differential bins, differential r; library |AUC - 0.5| (mean of controls)
  bulk: r on each cell's own track, fraction of the ENCODE replicate ceiling (plots/26_nov_13/ceiling), MSE;
        r on the balanced mixture (eval_summary.json '<cfg>/mixed'; for the seed-919 pair-A runs, whose summary
        predates the key fix, the 'all' row of the eval log)
  CTCF (ctcf_footprints.py, hole sites excluded) on the pair's own fibers, bound sites: H1 footprint - open pred
        CTCF, H2 that minus the ATAC difference, H3 open - nucleosome (per-site medians); H5 share of motif-score
        bins with bound > unbound
  XI (xi_promoters.py) on GM12878 fibers for every model (for pair B models GM12878 is an unseen cell): fraction of
        silenced CGI-promoter genes with the expected sign of Xa - Xi per mark, nulls escape + PAR and chr21, XIST
Seed noise: sd over seeds per model; a model difference (mean over seeds) counts as clear when
|delta| > 2 SE (SE = sqrt(sd_a^2 / n_a + sd_b^2 / n_b)) and every seed pairing has the same sign.

    python plots_code/26_nov_27/realism_seeds.py [--step molecules|summary|all]
'molecules' runs ctcf_footprints.py and xi_promoters.py for every run (molecule_predict output dirs are picked per
sample, newest first, and linked into ignore/seed_molecules/), 'summary' reads their outputs and the evals.
Writes plots/26_nov_27/seeds/: ctcf_<pair>/ and xi/ (per-run outputs and logs), seeds_long.tsv (pair, model,
seed, test, readout, value), seeds_table.tsv (mean, sd, n per pair x model x readout), seeds_deltas.tsv,
seeds_t3.png, seeds_bulk.png, seeds_bio.png; prints the tables (log_seeds.txt via tee).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_13"))
sys.path.insert(0, HERE)
from abl_common import run_name, tag as out_tag  # noqa: E402
from mix_two_pairs import npz_readouts, json_readouts, run_dirs  # noqa: E402
from realism_t3 import mixed_from_log, MODELS as T3_MODELS  # noqa: E402
from realism_bio import xi_groups, MARKS  # noqa: E402

OUT = "plots/26_nov_27/seeds"
LINKS = "ignore/seed_molecules"
HOLES = "plots/26_nov_20/ctcf_noholes/holes.txt"
CEIL = "plots/26_nov_13/ceiling/ceiling_cells.tsv"
ALL_ASSAYS = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]
PAIRS = {
    "A": {"name": "GM12878 + HepG2", "cells": ["GM12878", "HepG2_200U"], "t3": "eval24_pair_GM12878_HepG2_200U",
          "ctrl": ["eval25_pair_HepG2_100U_200U"], "mix": "eval31_mix_bal_GM12878_HepG2_200U_mini",
          "assays": ALL_ASSAYS,
          # label -> (abl_common key, seeds)
          "models": {"DEV4NF": ("noFIRE", [919, 2]), "R0": ("mixR0", [919]), "R1a": ("mixR1a", [919, 2, 3]),
                     "Pair": ("pairNF", [919, 2, 3])}},
    "B": {"name": "K562 + H1", "cells": ["K562_200U", "H1_200U"], "t3": "eval27_pair_K562_H1_200U",
          "ctrl": ["eval28_pair_K562_100U_200U", "eval29_pair_H1_100U_200U"],
          "mix": "eval32_mix_bal_K562_H1_200U_mini", "assays": [a for a in ALL_ASSAYS if a != "atac"],
          "models": {"DEV4NF": ("noFIRE", [919, 2]), "R1a": ("mixR1aB", [919, 2, 3]),
                     "Pair": ("pairNFB", [919, 2, 3])}},
}
# seed-919 pair-A runs: eval31 is read from the eval log (eval_summary.json predates the '<cfg>/mixed' key)
MIX_LOGS = {("A", "DEV4NF", 919): T3_MODELS["DEV4NF"][1], ("A", "R0", 919): T3_MODELS["R0"][1],
            ("A", "R1a", 919): T3_MODELS["R1a"][1], ("A", "Pair", 919): T3_MODELS["Pair"][1]}
COL = {"DEV4NF": "#606060", "R0": "#2a78d6", "R1a": "#eb6834", "Pair": "#1baf7a"}
ORDER = ["DEV4NF", "R0", "R1a", "Pair"]


def runs():
    for p, P in PAIRS.items():
        for label, (key, seeds) in P["models"].items():
            for s in seeds:
                yield p, label, key, s


# ---------------------------------------------------------------- molecules: CTCF and XI per run
def link_dir(key, seed, targets, samples, name):
    """ignore/seed_molecules/<name>/ with fibers_<sample>.tsv.gz linked from the newest molecule dir that has it."""
    d = os.path.join(LINKS, name)
    os.makedirs(d, exist_ok=True)
    dirs = sorted(glob.glob(f"results/*_molecules_{run_name(key, seed)}_{targets}"), reverse=True)
    for s in samples:
        f = next((os.path.join(x, f"fibers_{s}.tsv.gz") for x in dirs
                  if os.path.exists(os.path.join(x, f"fibers_{s}.tsv.gz"))), None)
        if f is None:
            raise FileNotFoundError(f"no {targets} fibers for {s} from {key} seed {seed}")
        dst = os.path.join(d, f"fibers_{s}.tsv.gz")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(f), dst)
        print(f"  {name}: {s} <- {os.path.dirname(f)}", flush=True)
    return d


def molecules():
    py = sys.executable
    for p, label, key, s in runs():
        t = out_tag(key, s)
        out = os.path.join(OUT, f"ctcf_{p}")
        os.makedirs(out, exist_ok=True)
        d = link_dir(key, s, "ctcf_sites_chr21", PAIRS[p]["cells"], f"ctcf_{p}_{t}")
        with open(os.path.join(out, f"log_ctcf_{t}.txt"), "w") as log:
            subprocess.run([py, "-u", os.path.join(HERE, "..", "26_nov_13", "ctcf_footprints.py"), "--model", key,
                            "--seed", str(s), "--dir", d, "--exclude", HOLES, "--out", out],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"ctcf {p} {label} seed {s} done", flush=True)
    done = set()
    for p, label, key, s in runs():
        t = out_tag(key, s)
        if t in done:   # DEV4NF appears under both pairs; XI is on GM12878 fibers either way
            continue
        done.add(t)
        out = os.path.join(OUT, "xi")
        os.makedirs(out, exist_ok=True)
        d = link_dir(key, s, "tss_chrX_chr21", ["GM12878"], f"xi_{t}")
        with open(os.path.join(out, f"log_xi_{t}.txt"), "w") as log:
            subprocess.run([py, "-u", os.path.join(HERE, "..", "26_nov_13", "xi_promoters.py"), "--model", key,
                            "--seed", str(s), "--dir", d, "--out", out],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"xi {label} ({key}) seed {s} done", flush=True)


# ---------------------------------------------------------------- evals
def eval_summary(run, key):
    """Newest eval_summary.json result dict that has key."""
    for d in run_dirs(run):
        f = os.path.join(d, "eval_summary.json")
        if os.path.exists(f):
            res = json.load(open(f))["results"]
            if key in res:
                return res
    return None


def collect():
    ceil = pd.read_csv(CEIL, sep="\t").set_index(["cell", "assay"]).ceiling_r
    rows = []
    for p, label, key, s in runs():
        P, run = PAIRS[p], run_name(key, s)
        A = P["assays"]

        def add(test, readout, vals):
            v = [x for x in vals if x is not None and not np.isnan(x)]
            rows.append({"pair": p, "model": label, "seed": s, "test": test, "readout": readout,
                         "value": float(np.mean(v)) if v else np.nan, "n_assays": len(v)})

        ro, _ = npz_readouts(run, P["t3"])
        diff, _ = json_readouts(run, P["t3"])
        if ro is None:
            print(f"!! no {P['t3']} T3 for {run}", flush=True)
        else:
            for r in ("auc", "sign", "dev"):
                add("pair", r, [ro[a][r] for a in A if a in ro])
            add("pair", "diff_r", [(diff or {}).get(a) for a in A])
        devs = []
        for c in P["ctrl"]:
            cr, _ = npz_readouts(run, c)
            if cr is None:
                print(f"!! no {c} control for {run}", flush=True)
            else:
                devs.append(np.mean([cr[a]["dev"] for a in A if a in cr]))
        add("library", "absdev", devs)
        res = eval_summary(run, f"{P['t3']}/{P['cells'][0]}")
        for c in P["cells"]:
            if res is None:
                continue
            pa, pl = res[f"{P['t3']}/{c}"]["per_assay_pearson"], res[f"{P['t3']}/{c}"]["per_assay_loss"]
            ca = [a for a in A if a in pa and (c, a) in ceil.index]
            add(c, "r", [pa[a] for a in ca])
            add(c, "frac_ceiling", [pa[a] / ceil[(c, a)] for a in ca])
            add(c, "mse", [pl[a] for a in ca if a in pl])
        mres = eval_summary(run, f"{P['mix']}/mixed")
        if mres is not None:
            add("mix", "r", [mres[f"{P['mix']}/mixed"]["per_assay_pearson"].get(a) for a in A])
        elif (p, label, s) in MIX_LOGS:
            m = mixed_from_log(MIX_LOGS[(p, label, s)]).get(P["mix"].split("_")[0], {})
            add("mix", "r", [m.get(a) for a in A])
        else:
            print(f"!! no {P['mix']} mixture result for {run}", flush=True)

        # CTCF, bound sites on the pair's own fibers
        t = out_tag(key, s)
        f = os.path.join(OUT, f"ctcf_{p}", f"ctcf_summary_{t}.tsv")
        if os.path.exists(f):
            cs = pd.read_csv(f, sep="\t")
            b = cs[cs.site_set == "bound"].set_index("sample")
            for c in P["cells"]:
                if c in b.index:
                    for col, ro_ in [("d_fp_open_ctcf_median", "H1"), ("d_ctcf_minus_atac_median", "H2"),
                                     ("d_open_nuc_ctcf_median", "H3")]:
                        add(f"ctcf_{c}", ro_, [b.loc[c, col]])
            h = pd.read_csv(os.path.join(OUT, f"ctcf_{p}", f"ctcf_bound_vs_unbound_by_score_{t}.tsv"), sep="\t")
            for c, g in h.groupby("sample"):
                if c in P["cells"]:
                    add(f"ctcf_{c}", "H5", [float((g.bound_mean > g.unbound_mean).mean())])
        else:
            print(f"!! no CTCF summary {f}", flush=True)
        # XI, GM12878 fibers
        f = os.path.join(OUT, "xi", f"xi_per_gene_{t}.tsv")
        if os.path.exists(f):
            g = xi_groups(pd.read_csv(f, sep="\t"))
            for gname in ["silenced_cgi", "escape_par", "chr21"]:
                fr = [(np.sign(g[gname][f"pred_{a}"].dropna()) == e).mean() for a, (_, e) in MARKS.items()]
                add(f"xi_{gname}", "frac_expected", fr)
                if gname == "silenced_cgi":
                    for (a, (lab, e)), v in zip(MARKS.items(), fr):
                        add(f"xi_{gname}", lab, [v])
            x = g["xist"].iloc[0]
            add("xi_xist", "frac_reversed", [float(np.sign(x[f"pred_{a}"]) == -e) for a, (_, e) in MARKS.items()])
        else:
            print(f"!! no XI output {f}", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables
def table(df):
    g = df.groupby(["pair", "test", "readout", "model"]).value
    t = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size(),
                      "seeds": g.apply(lambda v: " ".join(f"{x:.3f}" for x in v))}).reset_index()
    return t


def deltas(df):
    rows = []
    for (p, test, ro), g in df.groupby(["pair", "test", "readout"]):
        for a, b in [("R1a", "Pair"), ("R1a", "DEV4NF"), ("Pair", "DEV4NF")]:
            va, vb = g[g.model == a].value.dropna().values, g[g.model == b].value.dropna().values
            if len(va) < 2 or len(vb) < 2:
                continue
            d = va.mean() - vb.mean()
            se = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
            allsame = bool(np.all(np.sign(np.subtract.outer(va, vb)) == np.sign(d)))
            rows.append({"pair": p, "test": test, "readout": ro, "a": a, "b": b, "delta": d, "se": se,
                         "z": d / se if se > 0 else np.nan, "all_pairings_same_sign": allsame,
                         "clear": bool(abs(d) > 2 * se and allsame)})
    return pd.DataFrame(rows)


def fmt(t, tests_readouts, p):
    """mean ± sd per model (rows) x readout (columns)."""
    out = {}
    for test, ro, lab in tests_readouts:
        s = t[(t.pair == p) & (t.test == test) & (t.readout == ro)].set_index("model")
        out[lab] = {m: (f"{s.loc[m, 'mean']:.3f} ± {s.loc[m, 'sd']:.3f}" if s.loc[m, "n"] > 1
                        else f"{s.loc[m, 'mean']:.3f}") for m in s.index}
    return pd.DataFrame(out).reindex([m for m in ORDER if m in PAIRS[p]["models"]])


def print_tables(t, dl):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 30)
    for p, P in PAIRS.items():
        c0, c1 = P["cells"]
        print(f"\n======== pair {p}: {P['name']} (mean ± sd over seeds; DEV4NF 2 seeds, R0 1, R1a / Pair 3)")
        print("\n== T3 (mean over assays) and library control")
        print(fmt(t, [("pair", "auc", "AUC"), ("pair", "sign", "sign acc"), ("pair", "diff_r", "diff r"),
                      ("library", "absdev", "library |AUC-0.5|")], p).to_string())
        print("\n== bulk: own tracks (r, fraction of ceiling) and balanced mixture r")
        print(fmt(t, [(c0, "r", f"{c0} r"), (c0, "frac_ceiling", f"{c0} frac"), (c1, "r", f"{c1} r"),
                      (c1, "frac_ceiling", f"{c1} frac"), ("mix", "r", "mixture r")], p).to_string())
        print("\n== CTCF at bound sites (per-site medians; H5 = share of score bins bound > unbound)")
        print(fmt(t, [(f"ctcf_{c}", h, f"{c.split('_')[0]} {h}") for c in P["cells"]
                      for h in ("H1", "H2", "H3", "H5")], p).to_string())
        print("\n== XI on GM12878 fibers: fraction of genes with the expected sign (mean of 5 marks); XIST reversed")
        print(fmt(t, [("xi_silenced_cgi", "frac_expected", "silenced CGI"), ("xi_escape_par", "frac_expected",
                       "escape+PAR"), ("xi_chr21", "frac_expected", "chr21"),
                      ("xi_xist", "frac_reversed", "XIST reversed")], p).to_string())
        print(fmt(t, [("xi_silenced_cgi", lab, lab) for lab, _ in MARKS.values()], p).to_string())
        print("\n== differences of seed means (clear: |delta| > 2 SE and every seed pairing agrees)")
        d = dl[dl.pair == p].copy()
        print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- figures
def dots(ax, df, p, test, ro, title, base=None):
    models = [m for m in ORDER if m in PAIRS[p]["models"]]
    for i, m in enumerate(models):
        v = df[(df.pair == p) & (df.model == m) & (df.test == test) & (df.readout == ro)].value.dropna().values
        if not len(v):
            continue
        ax.bar(i, v.mean(), 0.6, color=COL[m], alpha=0.35)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=COL[m], s=18, zorder=3)
    if base is not None:
        ax.axhline(base, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    vals = df[(df.pair == p) & (df.test == test) & (df.readout == ro)].value.dropna()
    if len(vals):
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.3 + 0.01
        ax.set_ylim(min(lo - pad, base if base is not None else lo - pad), hi + pad)


def figure(df, panels, fname, title):
    fig, ax = plt.subplots(len(PAIRS), len(panels), figsize=(3.1 * len(panels), 3.0 * len(PAIRS)), squeeze=False)
    for r, p in enumerate(PAIRS):
        for k, (test, ro, ttl, base) in enumerate(panels):
            test = test.format(c0=PAIRS[p]["cells"][0], c1=PAIRS[p]["cells"][1])
            dots(ax[r, k], df, p, test, ro, ttl.format(c0=PAIRS[p]["cells"][0].split("_")[0],
                                                     c1=PAIRS[p]["cells"][1].split("_")[0]), base)
        ax[r, 0].set_ylabel(f"pair {p}: {PAIRS[p]['name']}", fontsize=9)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[m], label=m) for m in ORDER],
               loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(ORDER), frameon=False)
    fig.suptitle(title, y=1.07)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{fname}", flush=True)


def summary():
    df = collect()
    df.to_csv(os.path.join(OUT, "seeds_long.tsv"), sep="\t", index=False, float_format="%.4f")
    t = table(df)
    t.to_csv(os.path.join(OUT, "seeds_table.tsv"), sep="\t", index=False, float_format="%.4f")
    dl = deltas(df)
    dl.to_csv(os.path.join(OUT, "seeds_deltas.tsv"), sep="\t", index=False, float_format="%.4f")
    print_tables(t, dl)
    figure(df, [("pair", "auc", "fiber-origin AUC", 0.5), ("pair", "sign", "sign accuracy", 0.5),
                ("pair", "diff_r", "differential r", 0.0), ("library", "absdev", "library |AUC − 0.5|", 0.0)],
           "seeds_t3.png", "T3 readouts per training seed (dots), bar = mean")
    figure(df, [("{c0}", "frac_ceiling", "{c0}: r / ceiling", None), ("{c1}", "frac_ceiling", "{c1}: r / ceiling",
                                                                        None),
                ("mix", "r", "balanced mixture r", None)],
           "seeds_bulk.png", "Bulk accuracy per training seed (dots), bar = mean")
    figure(df, [("ctcf_{c0}", "H1", "{c0}: CTCF H1 (fp − open)", 0.0), ("ctcf_{c1}", "H1", "{c1}: CTCF H1", 0.0),
                ("ctcf_{c0}", "H2", "{c0}: CTCF H2 (− ATAC)", 0.0), ("ctcf_{c1}", "H2", "{c1}: CTCF H2", 0.0),
                ("xi_silenced_cgi", "frac_expected", "XI silenced CGI (GM12878)", 0.5)],
           "seeds_bio.png", "Biology tests per training seed (dots), bar = mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["molecules", "summary", "all"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.step in ("molecules", "all"):
        molecules()
    if args.step in ("summary", "all"):
        summary()


if __name__ == "__main__":
    main()
