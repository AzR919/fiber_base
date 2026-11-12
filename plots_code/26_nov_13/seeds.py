"""Second training seed of DEV4 and its four input ablations (Nov_13_extended.md sections 6-9): how much do the
reported numbers move between two trainings of the same model, and which model differences survive that?

Seed 919 = the runs in sections 6-9; seed 2 = the same configs retrained with --seed 2 (run prefix *_seed2,
abl_common.run_name). The test evals use the eval config's own seed, so both seeds are scored on the same fibers.

A  bulk: per model x seed x cell x assay Pearson r, MSE and cell-type r from eval_summary.json (as ablations.py).
   Seed noise of a reported entry (a group mean for one model and assay): sd_seed = sqrt(mean over the 5 models of
   d^2 / 2), d = seed 2 - seed 919 group mean, i.e. the SD of one training. A model difference averaged over both
   seeds (each model's mean of 2 seeds) has SD sd_seed; an effect counts as clear if |mean Delta| > 2 sd_seed and
   both seeds give the same sign.
B  ceiling: the fractions of section 9 (ceiling.py's ceiling_cells.tsv) per seed and averaged over the two seeds.
C  molecules: CTCF footprint response at bound sites (footprint - open, median per-site difference; ctcf_summary_*)
   and X inactivation (fraction of subject genes in the expected direction, XIST Xa - Xi; xi_summary_* and
   xi_per_gene_*) for both seeds. Run ctcf_footprints.py / xi_promoters.py --model <m> --seed 2 first.

    python plots_code/26_nov_13/seeds.py
Writes plots/26_nov_13/seeds/: seeds_bulk.tsv (long), seeds_groups.tsv (group means per seed, seed difference),
seeds_noise.tsv (sd_seed per metric x group x assay), seeds_effects.tsv (ablation - DEV4 per seed, mean, clear),
seeds_ceiling.tsv, seeds_molecules.tsv, seeds_scatter.png, seeds_effects_{r,ctr}.png, seeds_molecules.png,
log_seeds.txt.
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
from abl_common import MODELS, NAMES, SEEDS, run_name, tag  # noqa: E402

OUT = "plots/26_nov_13/seeds"
MOL = "plots/26_nov_13/molecules"
CEIL = "plots/26_nov_13/ceiling/ceiling_cells.tsv"
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k4me1", "ctcf", "h3k36me3", "h3k27me3", "h3k9me3"]
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
GROUPS = {"dev": ["GM12878", "K562_200U", "HepG2_200U", "H1_200U"],
          "held-out": ["H9_200U", "HCT116_200U", "MCF_7_200U"],
          "Panc1": ["Panc1_200U"]}
GROUP_OF = {c: g for g, cells in GROUPS.items() for c in cells}
COL = {"DEV4": "#4C9BE8", "noCpG": "#D9534F", "noFIRE": "#F0AD4E", "noAcc": "#5CB85C", "m6aCpG": "#8E6CC0"}
GCOL = {"dev": "#4C9BE8", "held-out": "#D9534F", "Panc1": "#333333"}
ABL = [m for m in MODELS if m != "DEV4"]
METRICS = {"r": "Pearson r", "mse": "MSE", "ctr": "cell-type r"}
XI_ASSAYS = ["pred_atac", "pred_h3k4me3", "pred_h3k27ac", "pred_h3k27me3", "pred_h3k9me3"]


def single_cell_results(run):
    """{cell: result dict} from every eval_summary.json of a run (single-cell evals only; newest dir wins)."""
    out = {}
    for path in sorted(glob.glob(f"results/*_eval_{run}*/eval_summary.json"), key=os.path.getmtime):
        for k, r in json.load(open(path))["results"].items():
            if "/" not in k:
                out[k] = r
    return out


def load_bulk():
    rows = []
    for m in MODELS:
        for s in SEEDS:
            res = single_cell_results(run_name(m, s))
            if not res:
                raise FileNotFoundError(f"no evals for {m} seed {s} ({run_name(m, s)})")
            for cell, r in res.items():
                ct = (r.get("partial") or {}).get("strata", {}).get("all", {}).get("per_assay", {})
                for a, v in r["per_assay_pearson"].items():
                    rows.append({"model": m, "seed": s, "cell": cell, "group": GROUP_OF[cell], "assay": a,
                                 "r": v, "mse": r["per_assay_loss"].get(a, np.nan),
                                 "ctr": ct.get(a, {}).get("pearson", np.nan)})
    return pd.DataFrame(rows)


def group_means(bulk):
    """Mean over each group's cell types (and all 8) per model, seed and assay, plus 'all' assays."""
    out = []
    for g in list(GROUPS) + ["all"]:
        sub = bulk if g == "all" else bulk[bulk.group == g]
        for a in ASSAYS + ["all"]:
            s = sub if a == "all" else sub[sub.assay == a]
            m = s.groupby(["model", "seed"])[list(METRICS)].mean().reset_index()
            m["group"], m["assay"] = g, a
            out.append(m)
    return pd.concat(out, ignore_index=True)


def noise_and_effects(gm):
    """sd_seed per metric x group x assay; ablation - DEV4 per seed, mean over seeds, and whether it is clear."""
    wide = gm.pivot_table(index=["group", "assay", "model"], columns="seed", values=list(METRICS))
    noise, eff = [], []
    for key in METRICS:
        d = (wide[(key, 2)] - wide[(key, 919)]).rename("d").reset_index()
        for (g, a), s in d.groupby(["group", "assay"]):
            sd = float(np.sqrt((s.d ** 2).mean() / 2))
            noise.append({"metric": key, "group": g, "assay": a, "sd_seed": sd, "mean_abs_d": s.d.abs().mean(),
                          "max_abs_d": s.d.abs().max(), "worst_model": s.loc[s.d.abs().idxmax(), "model"]})
            per = wide[key].xs((g, a), level=["group", "assay"])
            for m in ABL:
                d919, d2 = per.loc[m, 919] - per.loc["DEV4", 919], per.loc[m, 2] - per.loc["DEV4", 2]
                mean = (d919 + d2) / 2
                eff.append({"metric": key, "group": g, "assay": a, "model": m, "delta_919": d919, "delta_2": d2,
                            "delta_mean": mean, "sd_seed": sd, "z": mean / sd if sd > 0 else np.nan,
                            "clear": bool(abs(mean) > 2 * sd and np.sign(d919) == np.sign(d2))})
    return pd.DataFrame(noise), pd.DataFrame(eff)


def ceiling_fractions(bulk):
    c = pd.read_csv(CEIL, sep="\t")[["cell", "assay", "ceiling_r", "ceiling_ct_r", "mse_floor"]]
    b = bulk.merge(c, on=["cell", "assay"])
    b["frac_r"], b["frac_ctr"], b["mse_x_floor"] = b.r / b.ceiling_r, b.ctr / b.ceiling_ct_r, b.mse / b.mse_floor
    out = []
    for g in list(GROUPS) + ["all"]:
        sub = b if g == "all" else b[b.group == g]
        for a in ASSAYS + ["all"]:
            s = sub if a == "all" else sub[sub.assay == a]
            m = s.groupby(["model", "seed"])[["frac_r", "frac_ctr", "mse_x_floor"]].mean().reset_index()
            m["seed"] = m.seed.astype(str)
            mm = s.groupby("model")[["frac_r", "frac_ctr", "mse_x_floor"]].mean().reset_index().assign(seed="mean")
            out.append(pd.concat([m, mm]).assign(group=g, assay=a))
    return pd.concat(out, ignore_index=True)


def molecules():
    rows = []
    for m in MODELS:
        for s in SEEDS:
            t = tag(m, s)
            p = os.path.join(MOL, f"ctcf_summary_{t}.tsv")
            if os.path.exists(p):
                c = pd.read_csv(p, sep="\t")
                for _, r in c[c.site_set == "bound"].iterrows():
                    for a in ("ctcf", "atac", "h3k4me3"):
                        rows.append({"model": m, "seed": s, "test": "ctcf_footprint", "sample": r["sample"],
                                     "readout": f"fp-open {a}", "value": r[f"d_fp_open_{a}_median"],
                                     "p": r[f"d_fp_open_{a}_p"]})
                for _, r in c[c.site_set == "unbound"].iterrows():
                    rows.append({"model": m, "seed": s, "test": "ctcf_footprint", "sample": r["sample"],
                                 "readout": "fp-open ctcf (unbound)", "value": r["d_fp_open_ctcf_median"],
                                 "p": r["d_fp_open_ctcf_p"]})
            p = os.path.join(MOL, f"xi_summary_{t}.tsv")
            if os.path.exists(p):
                x = pd.read_csv(p, sep="\t")
                for _, r in x[x.group.isin(["subject", "subject_cgi"]) & x.column.isin(XI_ASSAYS)].iterrows():
                    rows.append({"model": m, "seed": s, "test": "xi", "sample": r["group"],
                                 "readout": f"{r['column'][5:]} frac expected", "value": r["frac_expected_sign"],
                                 "p": r["wilcoxon_p"]})
            p = os.path.join(MOL, f"xi_per_gene_{t}.tsv")
            if os.path.exists(p):
                g = pd.read_csv(p, sep="\t")
                for _, r in g[g["class"] == "xist"].iterrows():
                    for col in ("pred_atac", "pred_h3k4me3"):
                        rows.append({"model": m, "seed": s, "test": "xi", "sample": "XIST",
                                     "readout": f"{col[5:]} Xa-Xi", "value": r[col], "p": np.nan})
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_scatter(bulk):
    """Seed 2 against seed 919 per cell type x assay, one panel per model and metric (r, cell-type r)."""
    w = bulk.pivot_table(index=["model", "cell", "group", "assay"], columns="seed", values=["r", "ctr"]).reset_index()
    fig, ax = plt.subplots(2, len(MODELS), figsize=(3.6 * len(MODELS), 7.4))
    for i, key in enumerate(["r", "ctr"]):
        lo = -0.25 if key == "ctr" else 0
        for j, m in enumerate(MODELS):
            s = w[w.model == m]
            for g in GROUPS:
                t = s[s.group == g]
                ax[i, j].scatter(t[(key, 919)], t[(key, 2)], s=14, color=GCOL[g], label=g, alpha=0.85)
            ax[i, j].plot([lo, 1], [lo, 1], color="#999999", lw=0.8)
            d = (s[(key, 2)] - s[(key, 919)]).dropna()
            ax[i, j].text(0.04, 0.96, f"|seed 2 - seed 919|\nmedian {d.abs().median():.3f}, max {d.abs().max():.2f}",
                          transform=ax[i, j].transAxes, va="top", fontsize=7.5)
            ax[i, j].set_xlim(lo, 0.95)
            ax[i, j].set_ylim(lo, 0.95)
            ax[i, j].set_aspect("equal")
            if i == 0:
                ax[i, j].set_title(NAMES[m], fontsize=10)
            ax[i, j].set_xlabel(f"{METRICS[key]}, seed 919", fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel(f"{METRICS[key]}, seed 2", fontsize=8)
            style(ax[i, j])
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Two trainings of each model, every cell type x assay on chr21", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seeds_scatter.png"), dpi=130)
    plt.close(fig)


def plot_effects(eff, key):
    """Ablation - DEV4 per assay for each seed, with the +-2 sd_seed band; rows dev / held out, one column per
    ablation."""
    groups = ["dev", "held-out"]
    fig, ax = plt.subplots(len(groups), len(ABL), figsize=(4.6 * len(ABL), 3.9 * len(groups)), sharey=True)
    x = np.arange(len(ASSAYS))
    for i, g in enumerate(groups):
        for j, m in enumerate(ABL):
            t = eff[(eff.metric == key) & (eff.group == g) & (eff.model == m)].set_index("assay").reindex(ASSAYS)
            ax[i, j].bar(x, 4 * t.sd_seed, 0.86, bottom=-2 * t.sd_seed, color="#e4e4e4", label="±2 SD of one training")
            ax[i, j].bar(x - 0.19, t.delta_919, 0.36, color=COL[m], label="seed 919")
            ax[i, j].bar(x + 0.19, t.delta_2, 0.36, color=COL[m], alpha=0.45, hatch="///", edgecolor="white",
                         label="seed 2")
            ax[i, j].axhline(0, color="#333333", lw=0.8)
            ax[i, j].set_xticks(x, [LAB[a] for a in ASSAYS], rotation=45, ha="right", fontsize=8)
            if i == 0:
                ax[i, j].set_title(f"{NAMES[m]} − DEV4", fontsize=10)
            style(ax[i, j])
        ax[i, 0].set_ylabel(f"Δ {METRICS[key]}, {'dev' if g == 'dev' else 'held out (H9, HCT116, MCF-7)'}",
                            fontsize=9)
    ax[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Ablation effects on {METRICS[key]}, each seed paired with the DEV4 of the same seed", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"seeds_effects_{key}.png"), dpi=130)
    plt.close(fig)


def plot_molecules(mol):
    """Per model, seed 919 vs seed 2: CTCF footprint - open at bound sites (GM12878, K562) and the fraction of
    subject CGI promoters in the expected direction for H3K9me3 and H3K27me3 (chance 0.5)."""
    panels = [("ctcf_footprint", "GM12878", "fp-open ctcf", "CTCF footprint − open, GM12878 bound sites", 0),
              ("ctcf_footprint", "K562_200U", "fp-open ctcf", "CTCF footprint − open, K562 bound sites", 0),
              ("xi", "subject_cgi", "h3k9me3 frac expected", "Xi H3K9me3: CGI promoters lower on Xa", 0.5),
              ("xi", "subject_cgi", "h3k27me3 frac expected", "Xi H3K27me3: CGI promoters lower on Xa", 0.5)]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9))
    x = np.arange(len(MODELS))
    for k, (test, sample, readout, title, ref) in enumerate(panels):
        t = mol[(mol.test == test) & (mol["sample"] == sample) & (mol.readout == readout)]
        for j, s in enumerate(SEEDS):
            v = t[t.seed == s].set_index("model").value.reindex(list(MODELS))
            ax[k].bar(x + (j - 0.5) * 0.38, v, 0.36, color=[COL[m] for m in MODELS], alpha=1 if j == 0 else 0.45,
                      hatch=None if j == 0 else "///", edgecolor="white", label=f"seed {s}")
        ax[k].axhline(ref, color="#333333", lw=0.8, ls="-" if ref == 0 else "--")
        ax[k].set_xticks(x, [NAMES[m].replace(" (all inputs)", "") for m in MODELS], rotation=35, ha="right",
                         fontsize=8)
        ax[k].set_title(title, fontsize=9)
        if ref:
            ax[k].set_ylim(0, 1)
        style(ax[k])
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color="#777777", label="seed 919"),
                          Patch(facecolor="#777777", alpha=0.45, hatch="///", edgecolor="white", label="seed 2")],
                 fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Single-molecule readouts (sections 7-8) in two trainings of each model", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seeds_molecules.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    bulk = load_bulk()
    gm = group_means(bulk)
    noise, eff = noise_and_effects(gm)
    ceil = ceiling_fractions(bulk)
    mol = molecules()
    bulk.to_csv(os.path.join(OUT, "seeds_bulk.tsv"), sep="\t", index=False, float_format="%.4f")
    gm.to_csv(os.path.join(OUT, "seeds_groups.tsv"), sep="\t", index=False, float_format="%.4f")
    noise.to_csv(os.path.join(OUT, "seeds_noise.tsv"), sep="\t", index=False, float_format="%.4f")
    eff.to_csv(os.path.join(OUT, "seeds_effects.tsv"), sep="\t", index=False, float_format="%.4f")
    ceil.to_csv(os.path.join(OUT, "seeds_ceiling.tsv"), sep="\t", index=False, float_format="%.4f")
    mol.to_csv(os.path.join(OUT, "seeds_molecules.tsv"), sep="\t", index=False, float_format="%.4g")
    plot_scatter(bulk)
    for key in ("r", "ctr"):
        plot_effects(eff, key)
    if len(mol):
        plot_molecules(mol)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 40)
    L = [f"bulk rows: {len(bulk)} ({bulk.groupby(['model', 'seed']).size().to_dict()})"]
    d = bulk.pivot_table(index=["model", "cell", "assay"], columns="seed", values=list(METRICS))
    for key in METRICS:
        dd = (d[(key, 2)] - d[(key, 919)]).dropna()
        L.append(f"per cell x assay |seed 2 - seed 919| {key}: median {dd.abs().median():.4f}, "
                 f"90th pct {dd.abs().quantile(0.9):.4f}, max {dd.abs().max():.4f} ({dd.abs().idxmax()})")
    for key in METRICS:
        L += [f"\n=== sd_seed of a group-mean entry, {METRICS[key]}",
              noise[noise.metric == key].pivot(index="assay", columns="group", values="sd_seed")
              .reindex(ASSAYS + ["all"])[list(GROUPS) + ["all"]].round(4).to_string()]
    for key in METRICS:
        L.append(f"\n=== group means per seed, {METRICS[key]} (all assays)")
        L.append(gm[gm.assay == "all"].pivot_table(index="model", columns=["group", "seed"], values=key)
                 .reindex(list(MODELS)).round(4).to_string())
    for key in METRICS:
        e = eff[eff.metric == key]
        L += [f"\n=== ablation - DEV4, {METRICS[key]}: seed 919 / seed 2 / mean / z (mean / sd_seed); * = clear",
              e.assign(cell=lambda t: t.apply(lambda r: f"{r.delta_919:+.3f} {r.delta_2:+.3f} {r.delta_mean:+.3f}"
                                              f" z{r.z:+.1f}{'*' if r.clear else ''}", axis=1))
              .pivot_table(index=["group", "assay"], columns="model", values="cell", aggfunc="first")[ABL]
              .to_string()]
    L.append("\n=== fraction of the ceiling, mean of the two seeds (all assays)")
    c = ceil[(ceil.assay == "all")]
    for key in ("frac_r", "frac_ctr", "mse_x_floor"):
        L.append(f"--- {key}")
        L.append(c.pivot_table(index="model", columns=["group", "seed"], values=key, aggfunc="first")
                 .reindex(list(MODELS)).round(3).to_string())
    L.append("\n=== fraction of the ceiling per assay, mean of the two seeds")
    for g in ["dev", "held-out", "Panc1"]:
        for key in ("frac_r", "frac_ctr"):
            t = ceil[(ceil.group == g) & (ceil.seed == "mean")].pivot(index="assay", columns="model", values=key)
            L += [f"--- {g} {key}", t.reindex(ASSAYS + ["all"])[list(MODELS)].round(3).to_string()]
    if len(mol):
        L += ["\n=== molecules per seed",
              mol.pivot_table(index=["test", "sample", "readout"], columns=["model", "seed"], values="value")
              .round(3).to_string()]
    text = "\n".join(L)
    open(os.path.join(OUT, "log_seeds.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
