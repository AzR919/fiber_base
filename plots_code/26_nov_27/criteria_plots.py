"""Figures for the two open decisions on the success criteria (TODO_extended.md section 2f, after the seed-noise
stage; reports/Nov_27.md section 3). Reads realism_seeds.py's outputs and the eval npz files; run it after
realism_seeds.py.

1. Sign accuracy and differential r: criterion or reported cost?
   criteria_gaps.png      model - Pair per readout (seed means, bars = 2 SE), with the criterion-1 tolerance
                          (2 seed SDs of Pair) shaded, for both dev pairs
   criteria_per_assay.png sign accuracy and differential r per assay, dots = seeds: where the gap to Pair is
2. CTCF footprint (H1): per seed or on the seed mean?
   criteria_ctcf.png      H1 per seed and cell (filled = p < 0.01 per seed), seed mean ± sd; -log10 p per seed

    python plots_code/26_nov_27/criteria_plots.py
Writes plots/26_nov_27/criteria/: the three figures, criteria_gaps.tsv, criteria_per_assay.tsv,
criteria_ctcf.tsv.
"""
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
sys.path.insert(0, HERE)
from abl_common import run_name, tag as out_tag  # noqa: E402
from mix_two_pairs import npz_readouts, json_readouts  # noqa: E402
from realism_seeds import PAIRS, COL  # noqa: E402

SEEDS_DIR = "plots/26_nov_27/seeds"
OUT = "plots/26_nov_27/criteria"
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
MODELS = ["DEV4NF", "R1a", "Pair"]
P_PASS = 0.01


def style(a, axis="y"):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis=axis, color="#e6e6e6", lw=0.6)
    a.set_axisbelow(True)


def legend(fig, models, marker="o", y=1.02):
    fig.legend(handles=[Line2D([], [], marker=marker, ls="", color=COL[m], markersize=8, label=m) for m in models],
               loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(models), frameon=False)


# ---------------------------------------------------------------- 1a. gaps against the criterion band
def gaps(df):
    rows = []
    for p, P in PAIRS.items():
        c0, c1 = P["cells"]
        readouts = [("pair", "auc", "fiber-origin AUC"), (c0, "r", f"{c0.split('_')[0]} own-track r"),
                    (c1, "r", f"{c1.split('_')[0]} own-track r"), ("mix", "r", "mixture r"),
                    ("pair", "sign", "sign accuracy"), ("pair", "diff_r", "differential r")]
        for test, ro, lab in readouts:
            g = df[(df.pair == p) & (df.test == test) & (df.readout == ro)]
            vp = g[g.model == "Pair"].value.dropna().values
            for m in ["R1a", "DEV4NF"]:
                vm = g[g.model == m].value.dropna().values
                d = vm.mean() - vp.mean()
                se = np.sqrt(vm.var(ddof=1) / len(vm) + vp.var(ddof=1) / len(vp))
                rows.append({"pair": p, "readout": lab, "model": m, "delta_vs_pair": d, "se": se,
                             "tol_2sd_pair": 2 * vp.std(ddof=1),
                             "within_tolerance": bool(abs(d) <= 2 * vp.std(ddof=1)),
                             "clear_below_pair": bool(d + 2 * se < 0)})
    return pd.DataFrame(rows)


def plot_gaps(g):
    fig, ax = plt.subplots(1, len(PAIRS), figsize=(13, 4.2), sharey=False)
    for a, p in zip(ax, PAIRS):
        s = g[g.pair == p]
        labs = list(dict.fromkeys(s.readout))
        y = np.arange(len(labs))[::-1]
        for yi, lab in zip(y, labs):
            tol = s[(s.readout == lab)].tol_2sd_pair.iloc[0]
            a.add_patch(plt.Rectangle((-tol, yi - 0.38), 2 * tol, 0.76, color="#1baf7a", alpha=0.15, lw=0))
        for k, m in enumerate(["DEV4NF", "R1a"]):
            r = s[s.model == m].set_index("readout").reindex(labs)
            off = 0.14 if m == "R1a" else -0.14
            a.errorbar(r.delta_vs_pair, y + off, xerr=2 * r.se, fmt="o", color=COL[m], ms=6, capsize=3, lw=1.2)
        a.axvline(0, color="#1baf7a", lw=1)
        a.axhline(1.5, color="#bbbbbb", lw=0.8, ls=":")
        a.set_yticks(y)
        a.set_yticklabels(labs, fontsize=9)
        a.set_xlabel("model − Pair (seed means; bars = 2 SE)")
        a.set_title(f"{PAIRS[p]['name']}", fontsize=10)
        style(a, "x")
        a.text(0.99, 0.02, "below dotted line: the two\nreadouts the decision is about", transform=a.transAxes,
               ha="right", va="bottom", fontsize=7.5, color="#777777")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[m], markersize=7, label=f"{m} − Pair")
                        for m in ["R1a", "DEV4NF"]] +
               [plt.Rectangle((0, 0), 1, 1, color="#1baf7a", alpha=0.15, label="criterion 1: ± 2 seed SDs of Pair")],
               loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle("Gap to the pair model, 3 seeds each (DEV4NF 2)", y=1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "criteria_gaps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/criteria_gaps.png", flush=True)


# ---------------------------------------------------------------- 1b. per assay
def per_assay():
    rows = []
    for p, P in PAIRS.items():
        for label, (key, seeds) in P["models"].items():
            if label not in MODELS:
                continue
            for s in seeds:
                run = run_name(key, s)
                ro, _ = npz_readouts(run, P["t3"])
                diff, _ = json_readouts(run, P["t3"])
                for a in P["assays"]:
                    if ro and a in ro:
                        rows.append({"pair": p, "model": label, "seed": s, "assay": a, "readout": "sign",
                                     "value": ro[a]["sign"]})
                    if diff and diff.get(a) is not None:
                        rows.append({"pair": p, "model": label, "seed": s, "assay": a, "readout": "diff_r",
                                     "value": diff[a]})
    return pd.DataFrame(rows)


def plot_per_assay(pa):
    fig, ax = plt.subplots(2, len(PAIRS), figsize=(15, 7), squeeze=False)
    for c, p in enumerate(PAIRS):
        A = PAIRS[p]["assays"]
        x = np.arange(len(A))
        w = 0.8 / len(MODELS)
        for r, (ro, title) in enumerate([("sign", "sign accuracy at differential bins"),
                                         ("diff_r", "differential r")]):
            a = ax[r, c]
            for i, m in enumerate(MODELS):
                s = pa[(pa.pair == p) & (pa.model == m) & (pa.readout == ro)]
                xm = x + (i - (len(MODELS) - 1) / 2) * w
                means = s.groupby("assay").value.mean().reindex(A)
                a.bar(xm, means, w * 0.9, color=COL[m], alpha=0.35)
                for j, asy in enumerate(A):
                    v = s[s.assay == asy].value.values
                    a.scatter(np.full(len(v), xm[j]), v, color=COL[m], s=10, zorder=3)
            if ro == "sign":
                a.axhline(0.5, color="#999999", lw=0.8, ls="--")
                a.set_ylim(0.3, 1.0)
            a.set_xticks(x)
            a.set_xticklabels([LAB[s] for s in A], fontsize=8)
            a.set_title(f"{PAIRS[p]['name']}: {title}", fontsize=10)
            style(a)
    legend(fig, MODELS, marker="s")
    fig.suptitle("Where the gap to the pair model is, per assay (bar = seed mean, dots = seeds)", y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "criteria_per_assay.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/criteria_per_assay.png", flush=True)


# ---------------------------------------------------------------- 2. CTCF per seed
def ctcf():
    rows = []
    for p, P in PAIRS.items():
        for label, (key, seeds) in P["models"].items():
            if label not in MODELS:
                continue
            for s in seeds:
                cs = pd.read_csv(os.path.join(SEEDS_DIR, f"ctcf_{p}", f"ctcf_summary_{out_tag(key, s)}.tsv"),
                                 sep="\t")
                b = cs[cs.site_set == "bound"].set_index("sample")
                for c in P["cells"]:
                    rows.append({"pair": p, "cell": c, "model": label, "seed": s,
                                 "h1": b.loc[c, "d_fp_open_ctcf_median"], "p": b.loc[c, "d_fp_open_ctcf_p"],
                                 "frac_pos": b.loc[c, "d_fp_open_ctcf_frac_pos"]})
    return pd.DataFrame(rows)


def plot_ctcf(cd):
    cells = [(p, c) for p in PAIRS for c in PAIRS[p]["cells"]]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    w = 0.8 / len(MODELS)
    for j, (p, c) in enumerate(cells):
        for i, m in enumerate(MODELS):
            s = cd[(cd.pair == p) & (cd.cell == c) & (cd.model == m)]
            xm = j + (i - (len(MODELS) - 1) / 2) * w
            ax[0].bar(xm, s.h1.mean(), w * 0.9, color=COL[m], alpha=0.3)
            ax[0].errorbar(xm, s.h1.mean(), yerr=s.h1.std(ddof=1), color=COL[m], lw=1, capsize=3)
            jit = np.linspace(-0.06, 0.06, len(s))
            for k, (_, r) in enumerate(s.iterrows()):
                ok = r.p < P_PASS and r.h1 > 0
                ax[0].scatter(xm + jit[k], r.h1, s=26, zorder=3, color=COL[m] if ok else "white",
                              edgecolor=COL[m], lw=1.3)
                ax[1].scatter(xm + jit[k], -np.log10(max(r.p, 1e-30)) * np.sign(r.h1), s=26, zorder=3,
                              color=COL[m] if ok else "white", edgecolor=COL[m], lw=1.3)
            n_ok = int(((s.p < P_PASS) & (s.h1 > 0)).sum())
            ax[0].text(xm, -0.32, f"{n_ok}/{len(s)}", ha="center", fontsize=8, color=COL[m])
    ax[0].axhline(0, color="#999999", lw=0.8)
    ax[0].set_ylim(-0.4, 1.5)
    ax[0].set_ylabel("H1: footprint − open, pred CTCF\n(median over bound sites)")
    ax[0].set_title("CTCF footprint effect per training seed (filled = passes in that seed, p < 0.01; "
                    "n/N = seeds passing); bar = seed mean ± sd", fontsize=10)
    ax[1].axhline(-np.log10(P_PASS), color="#d9534f", lw=0.8, ls="--")
    ax[1].axhline(0, color="#999999", lw=0.8)
    ax[1].set_ylabel("−log10 p × sign of effect\n(Wilcoxon over sites)")
    ax[1].set_xticks(range(len(cells)))
    ax[1].set_xticklabels([f"{c.split('_')[0]}\n({PAIRS[p]['name']} models)" for p, c in cells], fontsize=9)
    for a in ax:
        style(a)
    legend(fig, MODELS)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "criteria_ctcf.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/criteria_ctcf.png", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(SEEDS_DIR, "seeds_long.tsv"), sep="\t")
    g = gaps(df)
    g.to_csv(os.path.join(OUT, "criteria_gaps.tsv"), sep="\t", index=False, float_format="%.4f")
    pa = per_assay()
    pa.to_csv(os.path.join(OUT, "criteria_per_assay.tsv"), sep="\t", index=False, float_format="%.4f")
    cd = ctcf()
    cd.to_csv(os.path.join(OUT, "criteria_ctcf.tsv"), sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print("\n== gaps to Pair\n" + g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    t = pa.groupby(["pair", "readout", "assay", "model"]).value.mean().unstack("model")[MODELS]
    t["R1a-Pair"] = t.R1a - t.Pair
    print("\n== per assay, seed means\n" + t.to_string(float_format=lambda v: f"{v:.3f}"))
    s = cd.assign(ok=(cd.p < P_PASS) & (cd.h1 > 0)).groupby(["model"]).agg(
        cases=("ok", "size"), pass_per_seed=("ok", "sum"))
    m = cd.groupby(["pair", "cell", "model"]).h1.agg(["mean", "std", "min"]).reset_index()
    m["seed_mean_minus_2se"] = m["mean"] - 2 * m["std"] / np.sqrt(cd.groupby(["pair", "cell", "model"]).size().values)
    print("\n== CTCF H1: per-seed passes\n" + s.to_string())
    print("\n== CTCF H1: seed mean, sd, min, mean - 2 SE\n" + m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_gaps(g)
    plot_per_assay(pa)
    plot_ctcf(cd)


if __name__ == "__main__":
    main()
