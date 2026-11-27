"""Baseline ladder (TODO_extended.md T11): every rung with the same supervision (R1a mixtures, share 0.5, balanced
stacks) on both dev pairs, through the realism-test readouts.
  Rules  rung 1: fixed per-assay rules on the fiber + annotations, per-assay calibration only (model24; seed 919)
  MLP    rung 2: small MLP on 31 hand-crafted per-position features, no annotations (model22; seeds 919, 2, 3)
  R1a    rung 3: DEV4NF's network (model16; seeds 919, 2, 3)
  R1a+F  rung 4: the network + the rung-2 features as extra input channels (model23; seeds 919, 2, 3)
  Pair   the same network trained on each cell's own tracks (the ceiling; seeds 919, 2, 3)
2 -> 3 asks whether deep learning is needed, 3 -> 4 whether the network already has what the features carry.
The floor (rung 0) is the dashed line: AUC / sign accuracy 0.5, differential r 0, CTCF effects 0.

Readouts as realism_seeds.py (T3, library control, bulk r and fraction of the replicate ceiling, balanced mixture r,
CTCF footprints, XI on GM12878 fibers), plus
  per assay: balanced mixture r, sign accuracy, differential r, fiber-origin AUC (which assays need the network?)
  CpG flips (ctcf_cpg.py part C), bound CTCF sites, accessible fibers: per-site median change of pred CTCF, ATAC,
        H3K4me3 when the CpGs over motif +- 100 bp are methylated (meth) / unmethylated (unmeth), or 1.5 kb away
Differences of seed means as in realism_seeds.py (clear: |delta| > 2 SE and every seed pairing agrees); Rules has
one seed, so its differences have no SE.

    python plots_code/26_nov_27/ladder.py [--step molecules|summary|all]
'molecules' runs ctcf_footprints.py and xi_promoters.py for the new rungs (R1a and Pair outputs are copied from
plots/26_nov_27/seeds/). Writes plots/26_nov_27/ladder/: ladder_long.tsv, ladder_table.tsv, ladder_deltas.tsv,
ladder_flips.tsv, ladder_t3.png, ladder_bulk.png, ladder_bio.png, ladder_flips.png, ladder_assays.png; prints the
tables (log_ladder.txt via tee).
"""
import argparse
import os
import shutil
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
import realism_seeds as rs  # noqa: E402
import ctcf_cpg as cc  # noqa: E402
from abl_common import run_name, tag as out_tag  # noqa: E402
from mix_two_pairs import npz_readouts, json_readouts  # noqa: E402

OUT = "plots/26_nov_27/ladder"
SEEDS_OUT = "plots/26_nov_27/seeds"
S3 = [919, 2, 3]
ORDER = ["Rules", "MLP", "R1a", "R1a+F", "Pair"]
COL = {"Rules": "#9a9a9a", "MLP": "#2a78d6", "R1a": "#eb6834", "R1a+F": "#8c3fb8", "Pair": "#1baf7a"}
MODELS = {"A": {"Rules": ("rulesA", [919]), "MLP": ("mlpA", S3), "R1a": ("mixR1a", S3), "R1a+F": ("hcA", S3),
                "Pair": ("pairNF", S3)},
          "B": {"Rules": ("rulesB", [919]), "MLP": ("mlpB", S3), "R1a": ("mixR1aB", S3), "R1a+F": ("hcB", S3),
                "Pair": ("pairNFB", S3)}}
NEW = {"rulesA", "rulesB", "mlpA", "mlpB", "hcA", "hcB"}
DELTAS = [("MLP", "R1a"), ("R1a+F", "R1a"), ("R1a", "Pair"), ("Rules", "MLP"), ("MLP", "Pair")]
FLIPS = [("meth", "ctcf"), ("unmeth", "ctcf"), ("meth_distal", "ctcf"), ("meth", "atac"), ("meth", "h3k4me3")]


def setup(models):
    """Point realism_seeds at the ladder models and output folder (its functions read these module globals)."""
    rs.PAIRS = {p: {**rs.PAIRS_ORIG[p], "models": models[p]} for p in models}
    rs.OUT, rs.ORDER, rs.COL = OUT, ORDER, COL
    rs.MIX_LOGS = {k: v for k, v in rs.MIX_LOGS_ORIG.items() if k[1] in ("R1a", "Pair")}


def runs():
    for p, M in MODELS.items():
        for label, (key, seeds) in M.items():
            for s in seeds:
                yield p, label, key, s


# ---------------------------------------------------------------- molecules
def molecules():
    setup({p: {lab: v for lab, v in M.items() if v[0] in NEW} for p, M in MODELS.items()})
    rs.molecules()
    setup(MODELS)
    # R1a and Pair: already done by realism_seeds.py
    for p, label, key, s in runs():
        if key in NEW:
            continue
        t = out_tag(key, s)
        for sub, names in [(f"ctcf_{p}", [f"ctcf_summary_{t}.tsv", f"ctcf_bound_vs_unbound_by_score_{t}.tsv"]),
                           ("xi", [f"xi_per_gene_{t}.tsv"])]:
            os.makedirs(os.path.join(OUT, sub), exist_ok=True)
            for n in names:
                shutil.copy(os.path.join(SEEDS_OUT, sub, n), os.path.join(OUT, sub, n))


# ---------------------------------------------------------------- per assay and CpG flips
def per_assay():
    rows = []
    for p, label, key, s in runs():
        P, run = rs.PAIRS[p], run_name(key, s)

        def add(a, ro, v):
            if v is not None and not np.isnan(v):
                rows.append({"pair": p, "model": label, "seed": s, "test": a, "readout": ro, "value": float(v)})

        ro, _ = npz_readouts(run, P["t3"])
        diff, _ = json_readouts(run, P["t3"])
        res = rs.eval_summary(run, f"{P['mix']}/mixed")
        mix = res[f"{P['mix']}/mixed"]["per_assay_pearson"] if res else {}
        for a in P["assays"]:
            if ro and a in ro:
                add(a, "auc", ro[a]["auc"])
                add(a, "sign", ro[a]["sign"])
            add(a, "diff_r", (diff or {}).get(a))
            add(a, "mix_r", mix.get(a))
    return pd.DataFrame(rows)


def flips():
    rows, crows = [], []
    for p, label, key, s in runs():
        try:
            d = cc.perturb_dir(key, s)
        except FileNotFoundError as e:
            print(f"!! {e}", flush=True)
            continue
        for sample in rs.PAIRS[p]["cells"]:
            c = pd.DataFrame(cc.part_c(cc.load(d, sample), label, s, sample)).assign(pair=p)
            crows.append(c)
            b = c[c.sites == "bound"].set_index(["mode", "assay"])
            for mode, a in FLIPS:
                rows.append({"pair": p, "model": label, "seed": s, "test": f"flip_{sample}",
                             "readout": f"{mode}_{a}", "value": b.loc[(mode, a), "median"]})
            h1 = b.loc[("meth", "ctcf"), "h1_footprint_effect"]
            rows.append({"pair": p, "model": label, "seed": s, "test": f"flip_{sample}", "readout": "meth_ctcf_rel_h1",
                         "value": b.loc[("meth", "ctcf"), "median"] / h1 if abs(h1) > 1e-6 else np.nan})
        print(f"flips {p} {label} seed {s}: {d}", flush=True)
    return pd.DataFrame(rows), pd.concat(crows, ignore_index=True)


# ---------------------------------------------------------------- tables
def deltas(df):
    rows = []
    for (p, test, ro), g in df.groupby(["pair", "test", "readout"]):
        for a, b in DELTAS:
            va, vb = g[g.model == a].value.dropna().values, g[g.model == b].value.dropna().values
            if not len(va) or not len(vb):
                continue
            d = va.mean() - vb.mean()
            se = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb)) if min(len(va), len(vb)) > 1 else np.nan
            same = bool(np.all(np.sign(np.subtract.outer(va, vb)) == np.sign(d)))
            rows.append({"pair": p, "test": test, "readout": ro, "a": a, "b": b, "n_a": len(va), "n_b": len(vb),
                         "delta": d, "se": se, "z": d / se if se and se > 0 else np.nan,
                         "all_pairings_same_sign": same, "clear": bool(se > 0 and abs(d) > 2 * se and same)
                         if not np.isnan(se) else False})
    return pd.DataFrame(rows)


def assay_table(t, p, ro):
    """model (rows) x assay (columns): mean over seeds (± sd)."""
    s = t[(t.pair == p) & (t.readout == ro) & t.test.isin(rs.ALL_ASSAYS)]
    cell = s.apply(lambda r: f"{r['mean']:.3f}" + (f"±{r['sd']:.3f}" if r["n"] > 1 else ""), axis=1)
    out = s.assign(cell=cell).pivot(index="model", columns="test", values="cell")
    return out.reindex(index=[m for m in ORDER if m in out.index],
                       columns=[a for a in rs.ALL_ASSAYS if a in out.columns])


def print_tables(t, ta, dl, dla):
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.max_rows", 500)
    for p, P in rs.PAIRS.items():
        c0, c1 = P["cells"]
        print(f"\n======== pair {p}: {P['name']} (mean ± sd over seeds; Rules 1 seed, others 3)")
        print("\n== T3 (mean over assays) and library control")
        print(rs.fmt(t, [("pair", "auc", "AUC"), ("pair", "sign", "sign acc"), ("pair", "diff_r", "diff r"),
                         ("library", "absdev", "library |AUC-0.5|")], p).to_string())
        print("\n== bulk: own tracks (r, fraction of ceiling) and balanced mixture r")
        print(rs.fmt(t, [(c0, "r", f"{c0} r"), (c0, "frac_ceiling", f"{c0} frac"), (c1, "r", f"{c1} r"),
                         (c1, "frac_ceiling", f"{c1} frac"), ("mix", "r", "mixture r")], p).to_string())
        for ro, lab in [("mix_r", "balanced mixture r"), ("sign", "sign accuracy"), ("diff_r", "differential r"),
                        ("auc", "fiber-origin AUC")]:
            print(f"\n== per assay: {lab}")
            print(assay_table(ta, p, ro).to_string())
        print("\n== CTCF at bound sites (per-site medians; H5 = share of score bins bound > unbound)")
        print(rs.fmt(t, [(f"ctcf_{c}", h, f"{c.split('_')[0]} {h}") for c in P["cells"]
                         for h in ("H1", "H2", "H3", "H5")], p).to_string())
        print("\n== CpG flips at bound CTCF sites, accessible fibers (median per-site change; rel = meth / H1)")
        print(rs.fmt(t, [(f"flip_{c}", f"{m}_{a}", f"{c.split('_')[0]} {m} {a}") for c in P["cells"]
                         for m, a in FLIPS + [("meth", "ctcf_rel_h1")]], p).T.to_string())
        print("\n== XI on GM12878 fibers: fraction of genes with the expected sign (mean of 5 marks); XIST reversed")
        print(rs.fmt(t, [("xi_silenced_cgi", "frac_expected", "silenced CGI"), ("xi_escape_par", "frac_expected",
                          "escape+PAR"), ("xi_chr21", "frac_expected", "chr21"),
                         ("xi_xist", "frac_reversed", "XIST reversed")], p).to_string())
        print("\n== differences of seed means (clear: |delta| > 2 SE and every seed pairing agrees)")
        print(dl[dl.pair == p].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n== per-assay differences, mixture r and sign accuracy")
        d = dla[(dla.pair == p) & dla.readout.isin(["mix_r", "sign"]) & dla.a.isin(["MLP", "R1a+F", "R1a"])]
        print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- figures
def fig_assays(df):
    reads = [("mix_r", "balanced mixture r"), ("sign", "sign accuracy"), ("diff_r", "differential r")]
    fig, ax = plt.subplots(len(MODELS), len(reads), figsize=(5.2 * len(reads), 3.2 * len(MODELS)), squeeze=False)
    for r, p in enumerate(MODELS):
        A = rs.PAIRS[p]["assays"]
        for k, (ro, ttl) in enumerate(reads):
            a_ = ax[r, k]
            for j, m in enumerate(ORDER):
                g = df[(df.pair == p) & (df.model == m) & (df.readout == ro)].groupby("test").value
                mu, sd = g.mean().reindex(A), g.std(ddof=1).reindex(A)
                x = np.arange(len(A)) + (j - 2) * 0.13
                a_.errorbar(x, mu, yerr=sd.fillna(0), fmt="o", ms=4, color=COL[m], lw=0.8, capsize=0)
            if ro == "sign":
                a_.axhline(0.5, color="#999999", lw=0.8, ls="--")
            a_.set_xticks(range(len(A)))
            a_.set_xticklabels([x.upper().replace("H3K", "K") for x in A], fontsize=7, rotation=30)
            a_.set_title(ttl if r == 0 else "", fontsize=9)
            a_.spines[["top", "right"]].set_visible(False)
            a_.grid(axis="y", color="#e6e6e6", lw=0.6)
            a_.set_axisbelow(True)
        ax[r, 0].set_ylabel(f"pair {p}: {rs.PAIRS[p]['name']}", fontsize=9)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[m], label=m) for m in ORDER],
               loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(ORDER), frameon=False)
    fig.suptitle("Per assay: mean over seeds ± sd (Rules: one seed)", y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder_assays.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/ladder_assays.png", flush=True)


def summary():
    df = rs.collect()
    fl, C = flips()
    df = pd.concat([df, fl], ignore_index=True)
    df.to_csv(os.path.join(OUT, "ladder_long.tsv"), sep="\t", index=False, float_format="%.4f")
    C.to_csv(os.path.join(OUT, "ladder_flips.tsv"), sep="\t", index=False, float_format="%.4g")
    pa = per_assay()
    pa.to_csv(os.path.join(OUT, "ladder_assays_long.tsv"), sep="\t", index=False, float_format="%.4f")
    t, ta = rs.table(df), rs.table(pa)
    pd.concat([t, ta]).to_csv(os.path.join(OUT, "ladder_table.tsv"), sep="\t", index=False, float_format="%.4f")
    dl, dla = deltas(df), deltas(pa)
    pd.concat([dl, dla]).to_csv(os.path.join(OUT, "ladder_deltas.tsv"), sep="\t", index=False, float_format="%.4f")
    print_tables(t, ta, dl, dla)
    rs.figure(df, [("pair", "auc", "fiber-origin AUC", 0.5), ("pair", "sign", "sign accuracy", 0.5),
                   ("pair", "diff_r", "differential r", 0.0), ("library", "absdev", "library |AUC − 0.5|", 0.0)],
              "ladder_t3.png", "Ladder, T3 readouts per training seed (dots), bar = mean")
    rs.figure(df, [("{c0}", "frac_ceiling", "{c0}: r / ceiling", None),
                   ("{c1}", "frac_ceiling", "{c1}: r / ceiling", None), ("mix", "r", "balanced mixture r", None)],
              "ladder_bulk.png", "Ladder, bulk accuracy per training seed (dots), bar = mean")
    rs.figure(df, [("ctcf_{c0}", "H1", "{c0}: CTCF H1 (fp − open)", 0.0), ("ctcf_{c1}", "H1", "{c1}: CTCF H1", 0.0),
                   ("ctcf_{c0}", "H2", "{c0}: CTCF H2 (− ATAC)", 0.0), ("ctcf_{c1}", "H2", "{c1}: CTCF H2", 0.0),
                   ("xi_silenced_cgi", "frac_expected", "XI silenced CGI (GM12878)", 0.5)],
              "ladder_bio.png", "Ladder, biology tests per training seed (dots), bar = mean")
    rs.figure(df, [("flip_{c0}", "meth_ctcf", "{c0}: methylate, Δ CTCF", 0.0),
                   ("flip_{c1}", "meth_ctcf", "{c1}: methylate, Δ CTCF", 0.0),
                   ("flip_{c0}", "unmeth_ctcf", "{c0}: unmethylate, Δ CTCF", 0.0),
                   ("flip_{c1}", "unmeth_ctcf", "{c1}: unmethylate, Δ CTCF", 0.0),
                   ("flip_{c0}", "meth_distal_ctcf", "{c0}: methylate 1.5 kb away", 0.0)],
              "ladder_flips.png", "Ladder, in-silico CpG flips at bound CTCF sites (per-site median change)")
    fig_assays(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["molecules", "summary", "all"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rs.PAIRS_ORIG, rs.MIX_LOGS_ORIG = dict(rs.PAIRS), dict(rs.MIX_LOGS)
    if args.step in ("molecules", "all"):
        molecules()
    setup(MODELS)
    if args.step in ("summary", "all"):
        summary()


if __name__ == "__main__":
    main()
