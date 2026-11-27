"""Bulk metrics of every model in the poised / ladder comparisons, per assay and cell (chr21, the test chromosome).

Models (each on its own pair's evals): DEV4NF (own tracks of the 4 dev cells; seeds 919, 2), Rules (919), MLP, R1a,
R1a+F, Pair (919, 2, 3). Readouts, from the eval outputs realism_seeds.py / ladder.py read:
  r             Pearson r of the prediction against the cell's own ENCODE track (asinh FC, T3 eval of the pair)
  frac_ceiling  r / the ENCODE replicate ceiling of that cell and assay (plots/26_nov_13/ceiling)
  mse           loss on the cell's own track
  mix_r         Pearson r against the balanced 0.5 mixture track (eval31 / eval32, chr21 mini)
Per model: mean and sd over seeds; differences to R1a with SE = sqrt(var_a / n_a + var_b / n_b), clear when |delta| >
2 SE and every seed pairing has the same sign (Rules: one seed, no SE). Pair B has no ATAC (H1 has no ATAC track).

    python plots_code/26_dec_04/bulk_models.py
Writes plots/26_dec_04/bulk/: bulk_long.tsv, bulk_table.tsv, bulk_deltas.tsv, bulk_assays.png; prints the tables.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_13"))
sys.path.insert(0, os.path.join(HERE, "..", "26_nov_27"))
import realism_seeds as rs  # noqa: E402
from abl_common import run_name  # noqa: E402
from realism_t3 import mixed_from_log  # noqa: E402

OUT = "plots/26_dec_04/bulk"
S3 = [919, 2, 3]
MODELS = {"A": {"DEV4NF": ("noFIRE", [919, 2]), "Rules": ("rulesA", [919]), "MLP": ("mlpA", S3),
                "R1a": ("mixR1a", S3), "R1a+F": ("hcA", S3), "Pair": ("pairNF", S3)},
          "B": {"DEV4NF": ("noFIRE", [919, 2]), "Rules": ("rulesB", [919]), "MLP": ("mlpB", S3),
                "R1a": ("mixR1aB", S3), "R1a+F": ("hcB", S3), "Pair": ("pairNFB", S3)}}
ORDER = ["Rules", "MLP", "R1a", "R1a+F", "Pair", "DEV4NF"]
COL = {"Rules": "#9e9e9e", "MLP": "#4a90d9", "R1a": "#eb6834", "R1a+F": "#8c3fb8", "Pair": "#1baf7a",
       "DEV4NF": "#606060"}
READOUTS = ["r", "frac_ceiling", "mse", "mix_r"]


def collect():
    ceil = pd.read_csv(rs.CEIL, sep="\t").set_index(["cell", "assay"]).ceiling_r
    rows = []
    for p, M in MODELS.items():
        P = rs.PAIRS[p]
        for label, (key, seeds) in M.items():
            for s in seeds:
                run = run_name(key, s)
                res = rs.eval_summary(run, f"{P['t3']}/{P['cells'][0]}")
                if res is None:
                    print(f"!! no {P['t3']} for {label} seed {s} ({run})", flush=True)
                else:
                    for c in P["cells"]:
                        pa = res[f"{P['t3']}/{c}"]["per_assay_pearson"]
                        pl = res[f"{P['t3']}/{c}"]["per_assay_loss"]
                        for a in P["assays"]:
                            if a not in pa:
                                continue
                            base = {"pair": p, "model": label, "seed": s, "cell": c, "assay": a}
                            rows.append({**base, "readout": "r", "value": pa[a]})
                            if (c, a) in ceil.index:
                                rows.append({**base, "readout": "frac_ceiling", "value": pa[a] / ceil[(c, a)]})
                            if a in pl:
                                rows.append({**base, "readout": "mse", "value": pl[a]})
                mres = rs.eval_summary(run, f"{P['mix']}/mixed")
                if mres is not None:
                    mix = mres[f"{P['mix']}/mixed"]["per_assay_pearson"]
                elif (p, label, s) in rs.MIX_LOGS:
                    mix = mixed_from_log(rs.MIX_LOGS[(p, label, s)]).get(P["mix"].split("_")[0], {})
                else:
                    print(f"!! no {P['mix']} mixture for {label} seed {s}", flush=True)
                    mix = {}
                for a in P["assays"]:
                    if mix.get(a) is not None:
                        rows.append({"pair": p, "model": label, "seed": s, "cell": "mixture", "assay": a,
                                     "readout": "mix_r", "value": mix[a]})
    return pd.DataFrame(rows)


def table(df):
    """Mean and sd over seeds per pair, model, cell, readout and assay, plus the assay mean ('all')."""
    per_seed_all = df.groupby(["pair", "model", "seed", "cell", "readout"]).value.mean().reset_index()
    per_seed_all["assay"] = "all"
    full = pd.concat([df, per_seed_all], ignore_index=True)
    t = full.groupby(["pair", "cell", "readout", "assay", "model"]).value.agg(["mean", "std", "size"]).reset_index()
    return full, t


def deltas(full, ref="R1a"):
    rows = []
    for (p, c, ro, a), g in full.groupby(["pair", "cell", "readout", "assay"]):
        if ref not in set(g.model):
            continue
        rv = g[g.model == ref].set_index("seed").value
        for m in ORDER:
            if m == ref or m not in set(g.model):
                continue
            mv = g[g.model == m].set_index("seed").value
            d = mv.mean() - rv.mean()
            se = np.sqrt(mv.var(ddof=1) / len(mv) + rv.var(ddof=1) / len(rv)) if len(mv) > 1 and len(rv) > 1 else np.nan
            signs = {np.sign(x - y) for x, y in itertools.product(mv.values, rv.values)}
            rows.append({"pair": p, "cell": c, "readout": ro, "assay": a, "model": m, "ref": ref, "delta": d,
                         "se": se, "clear": bool(not np.isnan(se) and abs(d) > 2 * se and len(signs) == 1),
                         "n": len(mv), "n_ref": len(rv)})
    return pd.DataFrame(rows)


def figure(t):
    cells = [("A", "GM12878"), ("A", "HepG2_200U"), ("B", "K562_200U"), ("B", "H1_200U"), ("A", "mixture"),
             ("B", "mixture")]
    fig, axes = plt.subplots(len(cells), 1, figsize=(11, 2.3 * len(cells)), sharex=True)
    assays = rs.ALL_ASSAYS + ["all"]
    for ax, (p, c) in zip(axes, cells):
        ro = "mix_r" if c == "mixture" else "r"
        sub = t[(t.pair == p) & (t.cell == c) & (t.readout == ro)]
        for j, m in enumerate(ORDER):
            q = sub[sub.model == m].set_index("assay")
            x = [i + (j - 2.5) * 0.12 for i, a in enumerate(assays) if a in q.index]
            y = [q.loc[a, "mean"] for a in assays if a in q.index]
            e = [q.loc[a, "std"] if q.loc[a, "size"] > 1 else 0 for a in assays if a in q.index]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=3.5, color=COL[m], lw=0.8, label=m)
        ax.set_ylabel(f"{c.split('_')[0]}{' (pair ' + p + ')' if c == 'mixture' else ''}\n{ro}", fontsize=8)
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=7, ncol=6)
    axes[-1].set_xticks(range(len(assays)), assays)
    fig.suptitle("Bulk r on each cell's own track and on the balanced mixture (chr21; mean ± sd over seeds)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bulk_assays.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    df = collect()
    df.to_csv(f"{OUT}/bulk_long.tsv", sep="\t", index=False)
    full, t = table(df)
    t.to_csv(f"{OUT}/bulk_table.tsv", sep="\t", index=False)
    d = deltas(full)
    d.to_csv(f"{OUT}/bulk_deltas.tsv", sep="\t", index=False)
    for ro in READOUTS:
        sub = t[t.readout == ro]
        if not len(sub):
            continue
        print(f"\n=== {ro} (mean over seeds) ===")
        print(sub.pivot_table(index=["pair", "cell", "assay"], columns="model", values="mean")
              .reindex(columns=[m for m in ORDER if m in set(sub.model)]).round(3).to_string())
    print("\n=== differences to R1a, assay mean ('all'): delta (SE), * = clear ===")
    s = d[d.assay == "all"].copy()
    s["txt"] = [f"{x:+.3f}" + ("" if np.isnan(e) else f" ({e:.3f})") + ("*" if c else "")
                for x, e, c in zip(s.delta, s.se, s.clear)]
    print(s.pivot_table(index=["pair", "cell", "readout"], columns="model", values="txt", aggfunc="first").to_string())
    print("\n=== clear per-assay differences to R1a (r and mix_r) ===")
    q = d[(d.assay != "all") & d.clear & d.readout.isin(["r", "mix_r"])]
    print(q[["pair", "cell", "readout", "assay", "model", "delta", "se"]].round(3).to_string(index=False))
    figure(t)
    print(f"\nWrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
