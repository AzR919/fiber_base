"""Mixture realism test, first run (TODO_extended.md section 2b, 23 Nov; GM12878 + HepG2, seed 919, DEV4NF's
channels): T3 readouts and bulk metrics for the four models. Companion of realism_bio.py (the biology tests).

Models: DEV4NF (4 dev cells, own tracks), R0 (mixtures, share redrawn per window, data27), R1a (share fixed at 0.5,
balanced stack, data36), Pair (own tracks of GM12878 and HepG2, data28). For reference, the Nov 13 Mix A / Pair A
(model15, with the FIRE channel; one seed) on the same pair.

Readouts (newest results/*_eval_<run>*/ dir that has them):
  T3, eval24 (GM12878 vs HepG2, chr21 tiles), from <cfg>_paired.npz and eval_summary.json:
    fiber-origin AUC, |AUC - 0.5|, sign accuracy at the pair's differential bins, differential r (bp level)
  library control, eval25 (HepG2 100U vs 200U): |AUC - 0.5| and AUC (should sit at chance)
  bulk per cell, eval24: per-assay Pearson r and MSE on each cell's own track; fraction of the ENCODE replicate
    ceiling (plots/26_nov_13/ceiling/ceiling_cells.tsv, same chr21 positions)
  bulk on mixtures (chr21 mini2000): eval26 (share redrawn per window) and eval31 (0.5, balanced stack). Both are
    keyed 'mixed' in eval_summary.json (the second overwrites the first), so the per-assay r is read from the
    "all" row the eval log prints for each config.

    python plots_code/26_nov_27/realism_t3.py
Writes plots/26_nov_27/realism/: t3_results.tsv (long: model, test, assay, readout, value), t3_summary.png,
t3_bulk.png; prints the tables (log_t3.txt via tee).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "26_nov_13"))
from mix_two_pairs import npz_readouts, json_readouts, run_dirs  # noqa: E402

OUT = "plots/26_nov_27/realism"
M16 = "model16_uct_mid_8A_no_fire_msp_train04"
# label -> (run name, eval log with the eval26 / eval31 sections or None)
MODELS = {
    "DEV4NF": (f"ABL_V1_data24_dev_loo_Panc1_200U_{M16}", "logs/2026-11-23_06_eval_dev4nf_pair_a.out"),
    "R0": (f"MIXR0_NF_data27_mix_GM12878_HepG2_200U_{M16}", "logs/2026-11-23_03_eval_r0_nf.out"),
    "R1a": (f"MIXR1A_NF_data36_mix_bal_GM12878_HepG2_200U_{M16}", "logs/2026-11-23_04_eval_r1a_nf.out"),
    "Pair": (f"PAIR_NF_data28_GM12878_HepG2_200U_{M16}", "logs/2026-11-23_05_eval_pair_nf.out"),
}
REF = {"Mix A (Nov, model15)": "MIX_V1_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train04",
       "Pair A (Nov, model15)": "PAIR_V1_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train04"}
COL = {"DEV4NF": "#606060", "R0": "#2a78d6", "R1a": "#eb6834", "Pair": "#1baf7a"}
PAIR, CTRL = "eval24_pair_GM12878_HepG2_200U", "eval25_pair_HepG2_100U_200U"
MIXES = {"eval26": "share redrawn", "eval31": "0.5 balanced"}
CELLS = ["GM12878", "HepG2_200U"]
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}


def bulk_cells(run):
    """{cell: {assay: (r, mse)}} on eval24 from the newest eval_summary.json that has it."""
    import json
    for d in run_dirs(run):
        p = os.path.join(d, "eval_summary.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        res = j["results"]
        if f"{PAIR}/{CELLS[0]}" not in res:
            continue
        out = {c: {a: (res[f"{PAIR}/{c}"]["per_assay_pearson"][a], res[f"{PAIR}/{c}"]["per_assay_loss"][a])
                   for a in ASSAYS} for c in CELLS}
        return out, {k: j.get(k) for k in ("checkpoint", "baseline_train_cells", "baseline_include_own")}
    return None, None


def mixed_from_log(path):
    """{eval26|eval31: {assay: r}} from the 'all' row after each mixture config in an eval log."""
    out, cur, header = {}, None, None
    for line in open(path):
        m = re.search(r"Config: configs/evals/(eval\d+)_", line)
        if m:
            cur, header = m.group(1), None
            continue
        if cur in MIXES and line.strip().startswith("region"):
            header = line.split()[2:]
        elif cur in MIXES and header and line.strip().startswith("all "):
            vals = line.split()[2:]
            out[cur] = {a: float(v) for a, v in zip(header, vals) if a != "mean"}
            cur = None
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ceil = pd.read_csv("plots/26_nov_13/ceiling/ceiling_cells.tsv", sep="\t").set_index(["cell", "assay"])
    rows = []

    def add(model, test, assay, readout, value):
        rows.append({"model": model, "test": test, "assay": assay, "readout": readout, "value": value})

    for label, run in [(k, v[0]) for k, v in MODELS.items()] + list(REF.items()):
        ro, _ = npz_readouts(run, PAIR)
        diff, _ = json_readouts(run, PAIR)
        ctrl, _ = npz_readouts(run, CTRL)
        for a in ASSAYS:
            if ro and a in ro:
                add(label, "pair", a, "auc", ro[a]["auc"])
                add(label, "pair", a, "absdev", ro[a]["dev"])
                add(label, "pair", a, "sign", ro[a]["sign"])
            if diff:
                add(label, "pair", a, "diff_r", diff.get(a))
            if ctrl and a in ctrl:
                add(label, "library", a, "auc", ctrl[a]["auc"])
                add(label, "library", a, "absdev", ctrl[a]["dev"])
        b, meta = bulk_cells(run)
        if b:
            print(f"{label}: {meta}", flush=True)
            for c in CELLS:
                for a in ASSAYS:
                    r, mse = b[c][a]
                    add(label, c, a, "r", r)
                    add(label, c, a, "mse", mse)
                    add(label, c, a, "frac_ceiling", r / ceil.loc[(c, a), "ceiling_r"])
        if label in MODELS:
            for e, v in mixed_from_log(MODELS[label][1]).items():
                for a in ASSAYS:
                    add(label, f"mix_{e}", a, "r", v[a])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "t3_results.tsv"), sep="\t", index=False, float_format="%.4f")

    order = list(MODELS) + list(REF)
    mean = df.groupby(["model", "test", "readout"]).value.mean().unstack(["test", "readout"]).reindex(order)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n== T3, GM12878 vs HepG2 (chr21 tiles), mean over 8 assays; library control HepG2 100U vs 200U")
    print(mean[[("pair", "auc"), ("pair", "absdev"), ("pair", "sign"), ("pair", "diff_r"), ("library", "absdev"),
                ("library", "auc")]].to_string(float_format="%.3f"))
    print("\n== bulk, mean over 8 assays: own tracks (r, fraction of replicate ceiling, MSE) and mixtures (r)")
    cols = [(c, x) for c in CELLS for x in ("r", "frac_ceiling", "mse")] + [(f"mix_{e}", "r") for e in MIXES]
    print(mean[[c for c in cols if c in mean.columns]].to_string(float_format="%.3f"))
    for test, ro in [("pair", "auc"), ("pair", "sign"), ("pair", "diff_r"), ("GM12878", "r"), ("HepG2_200U", "r"),
                     ("GM12878", "frac_ceiling"), ("HepG2_200U", "frac_ceiling"), ("mix_eval26", "r"),
                     ("mix_eval31", "r")]:
        t = df[(df.test == test) & (df.readout == ro)].pivot(index="model", columns="assay", values="value")
        t = t.reindex(index=[m for m in order if m in t.index], columns=ASSAYS).rename(columns=LAB)
        print(f"\n== {test} {ro} per assay")
        print(t.to_string(float_format="%.3f"))
    plot_summary(df)
    plot_bulk(df)


def _mean(df, model, test, ro):
    v = df[(df.model == model) & (df.test == test) & (df.readout == ro)].value
    return v.mean() if len(v) else np.nan


def _style(a):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e6e6e6", lw=0.6)
    a.set_axisbelow(True)


def _legend(fig):
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=COL[m], markersize=9, label=m) for m in MODELS],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(MODELS), frameon=False)


def plot_summary(df):
    """T3 means over assays: AUC, sign accuracy, differential r on the pair; |AUC - 0.5| on the library control."""
    panels = [("pair", "auc", "fiber-origin AUC", 0.5), ("pair", "sign", "sign accuracy, differential bins", 0.5),
              ("pair", "diff_r", "differential r (bp)", 0.0), ("library", "absdev", "library control |AUC − 0.5|", 0.0)]
    fig, ax = plt.subplots(1, len(panels), figsize=(15, 3.6))
    for a, (test, ro, title, base) in zip(ax, panels):
        v = [_mean(df, m, test, ro) for m in MODELS]
        a.bar(range(len(MODELS)), np.array(v) - base, bottom=base, color=[COL[m] for m in MODELS])
        for i, x in enumerate(v):
            a.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
        lo = min(v) - base
        a.set_ylim(base + min(0, lo) * 1.2, base + (max(v) - base) * 1.15)
        a.set_xticks(range(len(MODELS)))
        a.set_xticklabels(list(MODELS), fontsize=9)
        a.set_title(title, fontsize=10)
        _style(a)
    ax[0].set_ylabel("mean over 8 assays")
    _legend(fig)
    fig.suptitle("T3 readouts, GM12878 vs HepG2 on chr21, seed 919", y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "t3_summary.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}/t3_summary.png", flush=True)


def plot_bulk(df):
    """Per assay: fraction of the replicate ceiling on each cell's own track, and r on the two mixture evals."""
    panels = [("GM12878", "frac_ceiling", "GM12878 own track: r / replicate ceiling"),
              ("HepG2_200U", "frac_ceiling", "HepG2 own track: r / replicate ceiling"),
              ("mix_eval26", "r", "mixtures, share redrawn (eval26): r"),
              ("mix_eval31", "r", "mixtures, 0.5 balanced (eval31): r")]
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), sharey=False)
    x = np.arange(len(ASSAYS))
    w = 0.8 / len(MODELS)
    for a, (test, ro, title) in zip(ax.flat, panels):
        for i, m in enumerate(MODELS):
            t = df[(df.model == m) & (df.test == test) & (df.readout == ro)].set_index("assay").value
            a.bar(x + (i - (len(MODELS) - 1) / 2) * w, [t.get(s, np.nan) for s in ASSAYS], w * 0.92, color=COL[m])
        a.set_xticks(x)
        a.set_xticklabels([LAB[s] for s in ASSAYS], fontsize=8)
        a.set_title(title, fontsize=10)
        a.set_ylim(0, 1.05)
        _style(a)
    _legend(fig)
    fig.suptitle("Bulk accuracy on chr21, seed 919", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "t3_bulk.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}/t3_bulk.png", flush=True)


if __name__ == "__main__":
    main()
