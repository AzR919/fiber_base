"""Nov 13 report: the two-pair mixture test (T3 on GM12878-HepG2 and K562-H1), figures + all numbers in one place.

Models (all model15 / train04): Mix A / Mix B (mixtures of the pair only), Pair A / Pair B (same cells, own tracks),
DEV4 (4 dev cell types, own tracks). Each is evaluated on both pairs (eval24, eval27) and the three library controls
(eval25 HepG2, eval28 K562, eval29 H1: 100U vs 200U of the same cell type).

Sources per model: every results/*_eval_<run>*/ dir (timed-out evals were rerun into a second dir), newest first.
  fiber-origin AUC, sign accuracy, |AUC - 0.5|   <config>_paired.npz (as evaluator._paired_summary)
  differential r (bp level), recovery r          eval_summary.json

Writes to --out_dir:
  mix_summary.png             mean over assays: AUC, sign accuracy, differential r; both test pairs
  mix_per_assay_auc.png       fiber-origin AUC per assay, one panel per test pair
  mix_per_assay_diff_r.png    differential r per assay, one panel per test pair
  mix_library_control.png     |AUC - 0.5|: same cell type / two libraries vs the real pairs
  mix_bin_scatter_<pair>.png  predicted vs true difference per differential bin: mixture model on its own pair vs
                              the mixture model that never saw the pair
  mix_results.csv             every number above, long format (model, test, assay, readout, value)
and prints the markdown tables used in reports/Nov_13.md.

Run from the repo root on nibi (CPU, ~1 min):  python plots_code/26_nov_13/mix_two_pairs.py
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ---- models and configs ----
MODELS = [  # key, label, run name, supervision, pair it was trained on (None = all 4 dev)
    ("mixA", "Mix A", "MIX_V1_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train04", "mix", "GM12878–HepG2"),
    ("pairA", "Pair A", "PAIR_V1_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train04", "own", "GM12878–HepG2"),
    ("mixB", "Mix B", "MIX_V1_data35_mix_K562_H1_200U_model15_uct_mid_8A_train04", "mix", "K562–H1"),
    ("pairB", "Pair B", "PAIR_V1_data34_K562_H1_200U_model15_uct_mid_8A_train04", "own", "K562–H1"),
    ("dev4", "DEV4", "LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04", "dev4", None),
]
PAIRS = {"GM12878–HepG2": "eval24_pair_GM12878_HepG2_200U", "K562–H1": "eval27_pair_K562_H1_200U"}
CONTROLS = {"HepG2": "eval25_pair_HepG2_100U_200U", "K562": "eval28_pair_K562_100U_200U",
            "H1": "eval29_pair_H1_100U_200U"}
ORDER = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]  # narrow, then broad
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}

# ---- style (reference palette slots 1-3; text in ink, never series colour) ----
COLOR = {"mix": "#2a78d6", "own": "#eb6834", "dev4": "#1baf7a"}
SUP_LABEL = {"mix": "Mixtures only", "own": "Own tracks", "dev4": "DEV4 (all 4 dev, own tracks)"}
CONTROL_COLOR = "#b7b5ad"
REAL_COLOR = "#52514e"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---- loading ----
def run_dirs(run):
    return sorted(glob.glob(f"results/*_eval_{run}*"), reverse=True)


def npz_readouts(run, cfg):
    """{assay: {auc, sign, dev, n}} from the newest <cfg>_paired.npz, plus the bin-level arrays."""
    for d in run_dirs(run):
        path = os.path.join(d, f"{cfg}_paired.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        assays = [str(a) for a in z["assays"]]
        auc, td, dp, nb = z["auc"], z["tile_d"], z["tile_dp_bin"], z["tile_n_bin"]
        dt = z["tile_dt_bin"]
        out, bins = {}, {}
        for k, a in enumerate(assays):
            has = (td[:, k] != 0) & (nb[:, k] > 0)
            ak = auc[auc[:, 1] == k] if len(auc) else auc
            out[a] = {"n": int(has.sum()),
                      "sign": float(np.mean(np.sign(dp[has, k]) == np.sign(td[has, k]))) if has.any() else None,
                      "auc": float(ak[:, 3].mean()) if len(ak) else None,
                      "dev": float(np.abs(ak[:, 2] - 0.5).mean()) if len(ak) else None}
            bins[a] = (td[has, k], dt[has, k], dp[has, k])
        return out, bins
    return None, None


def json_readouts(run, cfg):
    """differential r per assay and recovery r per cell (mean over assays and per assay), newest JSON that has cfg."""
    for d in run_dirs(run):
        path = os.path.join(d, "eval_summary.json")
        if not os.path.exists(path):
            continue
        res = json.load(open(path))["results"]
        if f"{cfg}/paired" not in res:
            continue
        p = res[f"{cfg}/paired"]
        diff = {a: v["diff_pearson"] for a, v in p["strata"]["all"]["per_assay"].items()}
        rec = {}
        for c in p["cells"]:
            pa = res.get(f"{cfg}/{c}", {}).get("per_assay", {})
            rec[c] = {a: v.get("pearson") for a, v in pa.items()}
        return diff, rec
    return None, None


def mean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else None


def load_all():
    data = {}
    for key, _, run, _, _ in MODELS:
        for name, cfg in {**PAIRS, **CONTROLS}.items():
            ro, bins = npz_readouts(run, cfg)
            diff, rec = json_readouts(run, cfg) if name in PAIRS else (None, None)
            if ro is None:
                print(f"MISSING npz: {key} {cfg}")
            if name in PAIRS and diff is None:
                print(f"MISSING json: {key} {cfg}")
            data[(key, name)] = {"ro": ro, "bins": bins, "diff": diff, "rec": rec}
    return data


def assays_of(data, pair):
    ro = data[(MODELS[0][0], pair)]["ro"]
    return [a for a in ORDER if a in ro and ro[a]["auc"] is not None]


def value(data, key, pair, readout, assay):
    e = data[(key, pair)]
    if readout == "diff":
        return e["diff"].get(assay) if e["diff"] else None
    return e["ro"][assay][readout] if e["ro"] else None


# ---- drawing helpers ----
def style(ax, h0=None):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TEXT_2)
    ax.spines["bottom"].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2)
    if h0 is not None:
        ax.axhline(h0, color=TEXT, ls="--", lw=1.2, zorder=4)


def model_order(pair):
    """Bars per test pair: trained on it (mixture, own tracks), DEV4, never saw it (mixture, own tracks)."""
    seen = [m for m in MODELS if m[4] == pair]
    unseen = [m for m in MODELS if m[4] not in (pair, None)]
    dev = [m for m in MODELS if m[4] is None]
    return [(m, True) for m in seen] + [(m, True) for m in dev] + [(m, False) for m in unseen]


def bar(ax, x, h, w, model, seen):
    c = COLOR[model[3]]
    if seen:
        ax.bar(x, h, w, color=c, zorder=3, linewidth=0)
    else:
        ax.bar(x, h, w, facecolor=matplotlib.colors.to_rgba(c, 0.18), edgecolor=c, hatch="////", linewidth=0,
               zorder=3)


def legend_handles():
    hs = [Patch(color=COLOR[s], label=SUP_LABEL[s]) for s in ("mix", "own", "dev4")]
    hs.append(Patch(facecolor="white", edgecolor=TEXT_2, hatch="////", linewidth=0,
                    label="hatched: trained on the other pair (never saw these cell types)"))
    return hs


# ---- figures ----
READOUTS = [("auc", "Fiber-origin AUC", 0.5, (0.45, 0.72)),
            ("sign", "Sign accuracy at differential bins", 0.5, (0.45, 0.88)),
            ("diff", "Differential r (bp level)", 0.0, (0.0, 0.40))]


def fig_summary(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    w, gap = 0.17, 0.012
    for ax, (ro, title, h0, ylim) in zip(axes, READOUTS):
        for gi, pair in enumerate(PAIRS):
            assays = assays_of(data, pair)
            for bi, (m, seen) in enumerate(model_order(pair)):
                v = mean(value(data, m[0], pair, ro, a) for a in assays)
                x = gi + (bi - 2) * (w + gap)
                bar(ax, x, v, w, m, seen)
                ax.text(x, v + 0.004, f"{v:.2f}" if ro != "auc" else f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color=TEXT if seen else TEXT_2, rotation=90 if ro == "auc" else 0)
                ax.text(x, ylim[0] + 0.003, m[1], ha="center", va="bottom", fontsize=7.5, color="white" if seen
                        else TEXT_2, rotation=90, zorder=5)
        style(ax, h0 if h0 > 0 else None)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(PAIRS)))
        ax.set_xticklabels([f"tested on {p}\n({len(assays_of(data, p))} assays)" for p in PAIRS], fontsize=11,
                           color=TEXT)
        ax.set_title(title + "\n", fontsize=13, fontweight="bold", color=TEXT, loc="left")
        ax.text(0, 1.0, "mean over assays" + (f";  dashed: chance = {h0:g}" if h0 > 0 else ""),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=TEXT_2)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.13, wspace=0.18)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def fig_per_assay(data, ro, title, h0, ylim, out):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5))
    w, gap = 0.15, 0.01
    for ax, pair in zip(axes, PAIRS):
        assays = assays_of(data, pair)
        groups = assays + ["mean"]
        xs = np.array(list(range(len(assays))) + [len(assays) + 0.5], dtype=float)
        for gi, (g, xg) in enumerate(zip(groups, xs)):
            for bi, (m, seen) in enumerate(model_order(pair)):
                if g == "mean":
                    v = mean(value(data, m[0], pair, ro, a) for a in assays)
                else:
                    v = value(data, m[0], pair, ro, g)
                if v is None:
                    continue
                bar(ax, xg + (bi - 2) * (w + gap), v, w, m, seen)
        style(ax, h0 if h0 > 0 else None)
        ax.set_ylim(*ylim)
        ax.set_xticks(xs)
        n = data[(MODELS[0][0], pair)]["ro"]
        ax.set_xticklabels([f"{LAB[a]}\n{n[a]['n']} bins" for a in assays] + ["Mean"], fontsize=10.5, color=TEXT)
        ax.get_xticklabels()[-1].set_fontweight("bold")
        ax.set_title(f"Tested on {pair}", fontsize=13, fontweight="bold", color=TEXT, loc="left")
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, fontweight="bold", color=TEXT, y=0.995)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.965))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07, hspace=0.42)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def fig_library(data, out):
    comps = [(c, "control") for c in CONTROLS] + [(p, "real") for p in PAIRS]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = np.array([0, 1, 2, 3.5, 4.5], dtype=float)
    for x, (name, kind) in zip(xs, comps):
        vals = []
        for m in MODELS:
            ro = data[(m[0], name)]["ro"]
            vals.append(mean(v["dev"] for v in ro.values()) if ro else None)
        v = mean(vals)
        ax.bar(x, v, 0.6, color=CONTROL_COLOR if kind == "control" else REAL_COLOR, zorder=3, linewidth=0)
        ok = [u for u in vals if u is not None]
        ax.scatter(np.full(len(ok), x) + np.linspace(-0.18, 0.18, len(ok)), ok, s=22, color="white",
                   edgecolor=TEXT, linewidth=0.9, zorder=5)
        ax.text(x, max(ok) + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5, color=TEXT)
    style(ax)
    ax.set_ylim(0, 0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n100U vs 200U" for c in CONTROLS] + list(PAIRS), fontsize=10.5, color=TEXT)
    ax.set_ylabel("|AUC − 0.5|, fiber origin", color=TEXT_2)
    for lo, hi, txt in ((xs[0], xs[2], "same cell type, two libraries"), (xs[3], xs[4], "two cell types")):
        ax.annotate("", xy=(lo - 0.35, -0.2), xytext=(hi + 0.35, -0.2), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=1))
        ax.text((lo + hi) / 2, -0.23, txt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10.5,
                color=TEXT_2)
    ax.set_title("Fibers separate by cell type, not by library\n", fontsize=13, fontweight="bold", color=TEXT,
                 loc="left")
    ax.text(0, 1.0, "bar: mean over the 5 models; dots: each model; mean over tiles and assays",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=TEXT_2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.25)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def fig_bin_scatter(data, pair, out):
    """Rows: the mixture model trained on this pair, the one trained on the other pair. One dot per differential bin,
    x = true difference (A - B, asinh), y = predicted, coloured by which cell type's track is higher."""
    a_cell, b_cell = pair.split("–")
    own = next(m for m in MODELS if m[3] == "mix" and m[4] == pair)
    other = next(m for m in MODELS if m[3] == "mix" and m[4] not in (pair, None))
    assays = assays_of(data, pair)
    cols = {True: "#2a78d6", False: "#eb6834"}  # higher in A / higher in B
    fig, axes = plt.subplots(2, len(assays), figsize=(2.55 * len(assays), 5.8), squeeze=False)
    for r, (m, tag) in enumerate(((own, "trained on this pair"), (other, "trained on the other pair"))):
        e = data[(m[0], pair)]
        for c, a in enumerate(assays):
            ax = axes[r][c]
            d, x, y = e["bins"][a]
            for pos in (True, False):
                s = (d > 0) == pos
                ax.scatter(x[s], y[s], s=4, alpha=0.35, color=cols[pos], linewidth=0, rasterized=True)
            lim = max(np.percentile(np.abs(x), 99), np.percentile(np.abs(y), 99), 0.5) * 1.05
            ax.plot([-lim, lim], [-lim, lim], ls="--", color=TEXT_2, lw=0.8)
            ax.axhline(0, color=GRID, lw=0.8, zorder=0)
            ax.axvline(0, color=GRID, lw=0.8, zorder=0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            rr = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
            ax.set_title(f"{LAB[a]}\nr {rr:.2f}, sign {e['ro'][a]['sign']:.2f}", fontsize=9.5, color=TEXT)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7.5, colors=TEXT_2)
            if c == 0:
                ax.set_ylabel(f"{m[1]} ({tag})\npredicted {a_cell} − {b_cell}", fontsize=9.5, color=TEXT)
            if r == 1:
                ax.set_xlabel(f"true {a_cell} − {b_cell}", fontsize=9, color=TEXT_2)
    fig.legend(handles=[Patch(color=cols[True], label=f"higher in {a_cell}"),
                        Patch(color=cols[False], label=f"higher in {b_cell}")],
               loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"{pair}: predicted vs true difference per differential 1 kb bin (dashed: y = x)", x=0.01,
                 ha="left", fontsize=12.5, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.1, wspace=0.35, hspace=0.55)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# ---- tables ----
def write_csv(data, out):
    rows = []
    for key, label, run, sup, trained in MODELS:
        for name in list(PAIRS) + list(CONTROLS):
            e = data[(key, name)]
            if e["ro"]:
                for a, v in e["ro"].items():
                    for ro in ("auc", "sign", "dev", "n"):
                        rows.append((label, trained or "all 4 dev", name, a, ro, v[ro]))
            for a, v in (e["diff"] or {}).items():
                rows.append((label, trained or "all 4 dev", name, a, "diff_r", v))
            for cell, pa in (e["rec"] or {}).items():
                for a, v in pa.items():
                    rows.append((label, trained or "all 4 dev", name, a, f"recovery_r_{cell}", v))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "trained_on", "tested_on", "assay", "readout", "value"])
        w.writerows(rows)
    print(f"saved {out} ({len(rows)} rows)")


def print_tables(data):
    f = lambda v, nd=3: "–" if v is None else f"{v:.{nd}f}"
    print("\n### Mean over assays: AUC / sign accuracy / differential r\n")
    print("| Model | " + " | ".join(PAIRS) + " |\n|---" * 1 + "|---" * len(PAIRS) + "|")
    for key, label, _, _, trained in MODELS:
        cells = []
        for p in PAIRS:
            A = assays_of(data, p)
            cells.append(" / ".join(f(mean(value(data, key, p, ro, a) for a in A)) for ro in ("auc", "sign", "diff")))
        print(f"| {label} | " + " | ".join(cells) + " |")
    for ro, nd, name in (("auc", 3, "Fiber-origin AUC"), ("diff", 2, "Differential r")):
        for p in PAIRS:
            A = assays_of(data, p)
            print(f"\n### {name} per assay, {p}\n")
            print("| Model | " + " | ".join(LAB[a] for a in A) + " |\n" + "|---" * (len(A) + 1) + "|")
            for key, label, *_ in MODELS:
                print(f"| {label} | " + " | ".join(f(value(data, key, p, ro, a), nd) for a in A) + " |")
            if ro == "auc":
                n = data[(MODELS[0][0], p)]["ro"]
                print("| *bins* | " + " | ".join(str(n[a]["n"]) for a in A) + " |")
    print("\n### Library control: mean |AUC - 0.5|\n")
    names = list(CONTROLS) + list(PAIRS)
    print("| Model | " + " | ".join(names) + " |\n" + "|---" * (len(names) + 1) + "|")
    for key, label, *_ in MODELS:
        print(f"| {label} | " + " | ".join(
            f(mean(v["dev"] for v in data[(key, n)]["ro"].values())) if data[(key, n)]["ro"] else "–"
            for n in names) + " |")
    print("\n### Recovery r (own fibers vs own track, mean over assays)\n")
    for key, label, *_ in MODELS:
        parts = []
        for p in PAIRS:
            rec = data[(key, p)]["rec"] or {}
            parts.append(", ".join(f"{c} {f(mean(rec[c].values()))}" for c in rec))
        print(f"| {label} | " + " | ".join(parts) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="plots/26_nov_13/mix")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_all()
    print_tables(data)
    write_csv(data, os.path.join(args.out_dir, "mix_results.csv"))
    fig_summary(data, os.path.join(args.out_dir, "mix_summary.png"))
    fig_per_assay(data, "auc", "Fiber-origin AUC per assay (dashed: chance = 0.5)", 0.5, (0.4, 0.82),
                  os.path.join(args.out_dir, "mix_per_assay_auc.png"))
    fig_per_assay(data, "diff", "Differential r per assay (bp level)", 0.0, (0.0, 0.7),
                  os.path.join(args.out_dir, "mix_per_assay_diff_r.png"))
    fig_library(data, os.path.join(args.out_dir, "mix_library_control.png"))
    for p in PAIRS:
        tag = p.replace("–", "_")
        fig_bin_scatter(data, p, os.path.join(args.out_dir, f"mix_bin_scatter_{tag}.png"))


if __name__ == "__main__":
    main()
