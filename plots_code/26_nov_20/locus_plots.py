"""Single-molecule locus plots for reports/Nov_20_figs.md: the per-bp predictions of DEV4 and DEV4NF on individual
GM12878 fibers at the loci picked by select_loci.py (CTCF motif sites, X-inactivation promoters).

Input: molecule_predict.py --save_arrays output of molecules_loci.sh (results/*_molecules_<run>_loci_{ctcf,tss}/):
arrays_GM12878.npz (per target: pred (n, K, L), inputs (n, 5, L), mask (n, L), reads, hp, window) and
fibers_GM12878.tsv.gz (per-fiber input summaries, for the CTCF fiber states). Fibers are the same in every run
(same window, same reads); they are matched by read name.

Figures (plots/26_nov_20/loci/):
  ctcf_<site>.png   fibers sorted by state at the motif (footprint / open / nucleosome, as in
                    plots_code/26_nov_13/ctcf_footprints.py); columns: measured input, predicted CTCF for DEV4 and
                    DEV4NF in both seeds, predicted ATAC for DEV4NF; bottom row: mean per state, with the
                    ENCODE GM12878 bulk track of the same assay (asinh fold change, the training target's units)
                    and the Pearson r between each state's mean profile and it over the plotted +-300 bp
  xi_<gene>.png     phased fibers split by haplotype (Xa = HP1 on top, Xi = HP2 below; on chr21 an arbitrary pair);
                    columns: measured input, DEV4NF's five predicted marks; bottom row: mean per haplotype for
                    DEV4NF (solid) and DEV4 (dashed), seed 919
  ctcf_loci_summary.png  per site, mean predicted CTCF over the motif per state, all four runs
  ctcf_loci_shape.png    per site, Pearson r of each state's mean predicted CTCF profile with the bulk CTCF profile
  xi_loci_summary.png    per gene and mark, Xa - Xi over TSS +- 500 bp, all four runs

    python plots_code/26_nov_20/locus_plots.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pyBigWig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "26_nov_13"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repo root (utils)
from utils import load_cell_assays, load_metapaths
from abl_common import molecule_dir
from ctcf_footprints import fiber_state

OUT = "plots/26_nov_20/loci"
SAMPLE = "GM12878"
# (label, abl_common key, seed); seed-919 first
RUNS = [("DEV4", "DEV4", 919), ("DEV4NF", "noFIRE", 919), ("DEV4 seed 2", "DEV4", 2), ("DEV4NF seed 2", "noFIRE", 2)]
M6A, CPG, MSP, NUC = 0, 1, 2, 3
CTCF_HALF, XI_HALF, TSS_HALF = 300, 1500, 500
STATE_ORDER = ["footprint", "open", "nucleosome", "other"]
STATE_COL = {"footprint": "#eb6834", "open": "#2a78d6", "nucleosome": "#8c8c8c", "other": "#dddddd"}
HP_COL = {"Xa": "#2a78d6", "Xi": "#eb6834"}
XI_MARKS = [("atac", "ATAC", "higher on Xa"), ("h3k4me3", "H3K4me3", "higher on Xa"),
            ("h3k27ac", "H3K27ac", "higher on Xa"), ("h3k9me3", "H3K9me3", "lower on Xa"),
            ("h3k27me3", "H3K27me3", "lower on Xa")]
CMAP = "Blues"
# input image: 0 outside the fiber, 1 linker / unannotated, 2 MSP, 3 nucleosome, 4 m6A call. CpG calls are drawn
# as dots on top (cpg_dots), single-bp pixels vanish at figure resolution.
IN_COLORS = ["#ffffff", "#f4f4f4", "#9cc8f2", "#d4d4d4", "#1a1a1a"]
IN_CMAP = ListedColormap(IN_COLORS)
CPG_COL = {"methylated": "#1a1a1a", "unmethylated": "#ffffff"}
BULK_COL = "#1a1a1a"
_BW = {}


def bulk_track(assay, chrom, start, end):
    """ENCODE GM12878 bulk signal over [start, end), asinh(fold change) as in training (NaN -> 0)."""
    if assay not in _BW:
        m = load_metapaths("configs/metapaths.yaml")
        _BW[assay] = pyBigWig.open(load_cell_assays(m["cell_types"][SAMPLE])[assay])
    v = np.nan_to_num(np.asarray(_BW[assay].values(chrom, start, end, numpy=True), dtype=np.float64))
    return np.arcsinh(v)


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else np.nan


def load_run(key, seed, targets):
    d = molecule_dir(key, targets, seed)
    z = np.load(os.path.join(d, f"arrays_{SAMPLE}.npz"), allow_pickle=True)
    assays = json.load(open(os.path.join(d, "run.json")))["assays"]
    out = {}
    for k in z.files:
        name, field = k.split("::")
        out.setdefault(name, {})[field] = z[k]
    fib = pd.read_csv(os.path.join(d, f"fibers_{SAMPLE}.tsv.gz"), sep="\t")
    return out, list(assays), fib


def load_all(targets):
    runs = {}
    for label, key, seed in RUNS:
        runs[label] = load_run(key, seed, targets)
    return runs


def aligned(runs, name):
    """Reads present in every run at this target, in the first run's order, and per-run index arrays."""
    first = runs[RUNS[0][0]][0][name]["reads"]
    common = [r for r in dict.fromkeys(first) if all(r in set(runs[l][0][name]["reads"]) for l, _, _ in RUNS)]
    idx = {}
    for label, _, _ in RUNS:
        reads = list(runs[label][0][name]["reads"])
        pos = {r: i for i, r in reversed(list(enumerate(reads)))}
        idx[label] = np.array([pos[r] for r in common])
    return common, idx


def pred_image(a, assays, assay, rows, lo, hi):
    """(n, hi - lo) predictions with positions outside each fiber's scored span set to NaN."""
    p = a["pred"][rows, assays.index(assay), lo:hi].astype(np.float32)
    p[~a["mask"][rows, lo:hi]] = np.nan
    return p


def input_image(a, rows, lo, hi, cpg=False):
    x, m = a["inputs"][rows, :, lo:hi], a["mask"][rows, lo:hi]
    img = np.where(m, 1, 0)
    img[(x[:, MSP] == 1) & m] = 2
    img[(x[:, NUC] == 1) & m] = 3
    if not cpg:
        img[(x[:, M6A] == 1) & m] = 4
    return img


def cpg_dots(ax, a, rows, lo, hi, x0):
    """CpG calls of the plotted fibers as dots (row = fiber, x = position)."""
    x, m = a["inputs"][rows, CPG, lo:hi], a["mask"][rows, lo:hi]
    for val, key in [(-1, "unmethylated"), (1, "methylated")]:
        r, c = np.nonzero((x == val) & m)
        ax.scatter(c + x0 + 0.5, r + 0.5, s=2.5, color=CPG_COL[key], edgecolor="#555555" if val == -1 else "none",
                   linewidths=0.2, zorder=3)


def nanmean_rows(v, min_n=3):
    n = np.isfinite(v).sum(0)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(v, 0) if len(v) else np.full(v.shape[1], np.nan)
    out[n < min_n] = np.nan
    return out


def shared_limits(imgs):
    v = np.concatenate([i[np.isfinite(i)] for i in imgs])
    return np.percentile(v, 1), np.percentile(v, 99)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)


def heat(ax, img, x0, x1, vmin=None, vmax=None, cmap=CMAP, **kw):
    kw.setdefault("interpolation", "nearest")   # no blended edges at fiber ends
    return ax.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=(x0, x1, len(img), 0), **kw)


# ---------------------------------------------------------------------------------------------------------- CTCF

def ctcf_site(runs, name, sel):
    a0 = runs[RUNS[0][0]][0][name]
    chrom, ws, we = a0["window"]
    ws = int(ws)
    s = sel.set_index("name").loc[name]
    mid = (int(s.start) + int(s.end)) // 2
    lo, hi = mid - CTCF_HALF - ws, mid + CTCF_HALF - ws
    common, idx = aligned(runs, name)
    fib = runs[RUNS[0][0]][2]
    fib = fib[fib.target == name].drop_duplicates("read").copy()
    fib["state"] = fiber_state(fib)
    state = fib.set_index("read").state.reindex(common).fillna("other").values
    start = np.array([np.flatnonzero(a0["mask"][i])[0] for i in idx[RUNS[0][0]]])
    order = np.lexsort((start, [STATE_ORDER.index(x) for x in state]))
    state = state[order]
    rows = {l: idx[l][order] for l in idx}
    xs = np.arange(lo, hi) + ws - mid

    cols = [(l, "ctcf") for l, _, _ in RUNS] + [("DEV4NF", "atac")]
    bulk = {a: bulk_track(a, chrom, ws + lo, ws + hi) for a in ("ctcf", "atac")}
    shape = []
    imgs = {c: pred_image(runs[c[0]][0][name], runs[c[0]][1], c[1], rows[c[0]], lo, hi) for c in cols}
    vmin, vmax = shared_limits([imgs[c] for c in cols if c[1] == "ctcf"])
    avmin, avmax = shared_limits([imgs[("DEV4NF", "atac")]])

    w = [0.15, 1] + [1] * len(cols)
    fig, ax = plt.subplots(2, len(w), figsize=(3 * len(w), 9.5), gridspec_kw={"width_ratios": w, "height_ratios": [3, 1]})
    n = len(state)
    ax[0, 0].imshow(np.array([[STATE_ORDER.index(x)] for x in state]), aspect="auto",
                    cmap=ListedColormap([STATE_COL[k] for k in STATE_ORDER]), vmin=0, vmax=3, extent=(0, 1, n, 0))
    ax[0, 0].set_xticks([]); ax[0, 0].set_ylabel(f"{n} fibers, sorted by state at the motif")
    ax[1, 0].axis("off")
    heat(ax[0, 1], input_image(a0, rows[RUNS[0][0]], lo, hi), xs[0], xs[-1] + 1, 0, 4, IN_CMAP,
         interpolation="antialiased")
    ax[0, 1].set_title("measured input\n(m6A calls on MSP / nucleosome)", fontsize=9)
    m6 = a0["inputs"][rows[RUNS[0][0]], M6A, lo:hi].astype(float)
    m6[m6 == 0] = np.nan
    m6 = (m6 == 1).astype(float) + np.where(np.isnan(m6), np.nan, 0)
    for st in ["footprint", "open", "nucleosome"]:
        k = state == st
        prof = pd.Series(nanmean_rows(m6[k])).rolling(21, center=True, min_periods=5).mean()
        ax[1, 1].plot(xs, prof, color=STATE_COL[st], lw=1.5)
    ax[1, 1].set_ylabel("m6A rate (21 bp)"); ax[1, 1].set_title("measured", fontsize=9)
    for j, c in enumerate(cols):
        a = ax[0, j + 2]
        img = imgs[c]
        lim = (vmin, vmax) if c[1] == "ctcf" else (avmin, avmax)
        im = heat(a, img, xs[0], xs[-1] + 1, *lim)
        a.set_title(f"{c[0]}\npredicted {c[1].upper()}", fontsize=9, fontweight="bold" if c == ("DEV4NF", "ctcf") else None)
        a.set_yticks([])
        if j == 0 or c[1] == "atac":
            fig.colorbar(im, ax=a, fraction=0.05, pad=0.02).ax.tick_params(labelsize=7)
        r = {}
        for st in ["footprint", "open", "nucleosome"]:
            prof = nanmean_rows(img[state == st])
            ax[1, j + 2].plot(xs, prof, color=STATE_COL[st], lw=1.5)
            r[st] = pearson(prof, bulk[c[1]])
            shape.append({"site": name, "group": s.group, "run": c[0], "assay": c[1], "state": st, "r_with_bulk": r[st]})
        ax[1, j + 2].plot(xs, bulk[c[1]], color=BULK_COL, lw=1.6, ls="--")
        if np.all(bulk[c[1]] == 0):
            ax[1, j + 2].set_title(f"mean per state\n(bulk {c[1].upper()} = 0 here: no data)", fontsize=8)
        else:
            ax[1, j + 2].set_title(f"mean per state; r with bulk:\nfootprint {r['footprint']:.2f}, "
                                   f"open {r['open']:.2f}", fontsize=8)
    ylim = [min(ax[1, j + 2].get_ylim()[0] for j in range(len(cols) - 1)),
            max(ax[1, j + 2].get_ylim()[1] for j in range(len(cols) - 1))]
    for j in range(len(cols) - 1):
        ax[1, j + 2].set_ylim(ylim)
    for a in list(ax[0, 1:]) + list(ax[1, 1:]):
        a.axvline(-7.5, color="k", lw=0.6, ls=":"); a.axvline(7.5, color="k", lw=0.6, ls=":")
        a.set_xlim(xs[0], xs[-1] + 1)
    for a in ax[1, 1:]:
        style(a); a.set_xlabel("bp from motif centre", fontsize=8); a.tick_params(labelsize=7)
    for a in ax[0, 1:]:
        a.set_xticks([])
    counts = ", ".join(f"{(state == k).sum()} {k}" for k in STATE_ORDER)
    handles = [Patch(color=STATE_COL[k], label=k) for k in STATE_ORDER] + \
              [Line2D([], [], color=BULK_COL, lw=1.6, ls="--", label="ENCODE GM12878 bulk (same assay)"),
               Patch(color=IN_COLORS[2], label="MSP (input)"), Patch(color=IN_COLORS[3], label="nucleosome (input)"),
               Patch(color=IN_COLORS[4], label="m6A call (input)")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    kind = "bound in GM12878" if s.group == "bound" else "unbound motif (no peak in any dev line)"
    fig.suptitle(f"{name}: CTCF motif {chrom}:{int(s.start):,}-{int(s.end):,} ({kind}, motif score {s.motif_score}), "
                 f"GM12878 fibers ({counts}); dotted lines = motif", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"ctcf_{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # per-state mean over the motif for every run (summary figure)
    rec = []
    for label, _, _ in RUNS:
        p = pred_image(runs[label][0][name], runs[label][1], "ctcf", rows[label], lo + CTCF_HALF - 7, lo + CTCF_HALF + 8)
        v = np.nanmean(p, 1)
        for st in ["footprint", "open", "nucleosome"]:
            rec.append({"site": name, "group": s.group, "run": label, "state": st,
                        "mean_pred_ctcf": float(np.nanmean(v[state == st])), "n": int((state == st).sum()),
                        "bulk_ctcf_motif": float(bulk["ctcf"][CTCF_HALF - 7:CTCF_HALF + 8].mean())})
    return rec, shape


def ctcf_shape(shape):
    """Pearson r of each state's mean predicted CTCF profile with the bulk CTCF profile, per site and run."""
    df = pd.DataFrame(shape)
    df.to_csv(os.path.join(OUT, "ctcf_loci_shape.tsv"), sep="\t", index=False, float_format="%.4g")
    d = df[df.assay == "ctcf"]
    sites = list(dict.fromkeys(d.site))
    labels = [l for l, _, _ in RUNS]
    fig, ax = plt.subplots(1, len(sites), figsize=(4 * len(sites), 3.8), sharey=True)
    for k, site in enumerate(sites):
        a = ax[k]
        e = d[d.site == site]
        for j, st in enumerate(["footprint", "open", "nucleosome"]):
            v = [e[(e.run == l) & (e.state == st)].r_with_bulk.iloc[0] for l in labels]
            a.plot(np.arange(len(labels)) + (j - 1) * 0.12, v, "o", color=STATE_COL[st], ms=7, mec="white", mew=1)
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        a.set_title(f"{site} ({e.group.iloc[0]})", fontsize=10)
        a.axhline(0, color="#999999", lw=0.8)
        style(a)
    ax[0].set_ylabel("Pearson r, mean predicted CTCF profile\nvs ENCODE bulk CTCF (± 300 bp)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=STATE_COL[s], label=s)
                        for s in ["footprint", "open", "nucleosome"]],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ctcf_loci_shape.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(df.pivot_table(index=["site", "assay", "state"], columns="run", values="r_with_bulk", sort=False).round(2),
          flush=True)


def ctcf_summary(rec):
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(OUT, "ctcf_loci_summary.tsv"), sep="\t", index=False, float_format="%.4g")
    sites = list(dict.fromkeys(df.site))
    fig, ax = plt.subplots(1, len(sites), figsize=(4 * len(sites), 3.8), sharey=True)
    labels = [l for l, _, _ in RUNS]
    for k, site in enumerate(sites):
        a = ax[k]
        d = df[df.site == site]
        for j, st in enumerate(["footprint", "open", "nucleosome"]):
            v = [d[(d.run == l) & (d.state == st)].mean_pred_ctcf.iloc[0] for l in labels]
            a.plot(np.arange(len(labels)) + (j - 1) * 0.12, v, "o", color=STATE_COL[st], ms=7, mec="white", mew=1)
        b = d.bulk_ctcf_motif.iloc[0]
        if b > 0:
            a.axhline(b, color=BULK_COL, lw=1.2, ls="--")
        else:
            a.text(0.5, 0.97, "bulk CTCF = 0 over ± 20 kb\n(no ChIP signal: no data, not unbound)", fontsize=8,
                   ha="center", va="top", transform=a.transAxes)
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        a.set_title(f"{site} ({d.group.iloc[0]})", fontsize=10)
        style(a)
    ax[0].set_ylabel("mean predicted CTCF over the motif")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=STATE_COL[s], label=s)
                        for s in ["footprint", "open", "nucleosome"]] +
                       [Line2D([], [], color=BULK_COL, lw=1.2, ls="--", label="ENCODE GM12878 bulk CTCF at the motif")],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ctcf_loci_summary.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(df.pivot_table(index=["site", "state"], columns="run", values="mean_pred_ctcf", sort=False).round(2), flush=True)


# ------------------------------------------------------------------------------------------- X inactivation

def xi_gene(runs, name, sel):
    s = sel.set_index("name").loc[name]
    a0 = runs["DEV4NF"][0][name]
    chrom, ws, we = a0["window"]
    ws = int(ws)
    tss = (int(s.start) + int(s.end)) // 2
    lo, hi = tss - XI_HALF - ws, tss + XI_HALF - ws
    common, idx = aligned(runs, name)
    hp = np.array([str(x) for x in a0["hp"][idx["DEV4NF"]]])
    keep = np.isin(hp, ["1", "1.0", "2", "2.0"])
    hapl = np.where(np.isin(hp, ["1", "1.0"]), "Xa", "Xi")
    start = np.array([np.flatnonzero(a0["mask"][i])[0] for i in idx["DEV4NF"]])
    order = [i for i in np.lexsort((start, (hapl == "Xi").astype(int))) if keep[i]]   # Xa block, then Xi
    hapl = hapl[order]
    rows = {l: idx[l][order] for l in idx}
    xs = np.arange(lo, hi) + ws - tss
    n_xa = int((hapl == "Xa").sum())

    w = [0.15, 1] + [1] * len(XI_MARKS)
    fig, ax = plt.subplots(2, len(w), figsize=(3 * len(w), 9.5), gridspec_kw={"width_ratios": w, "height_ratios": [3, 1]})
    n = len(hapl)
    ax[0, 0].imshow(np.array([[0 if h == "Xa" else 1] for h in hapl]), aspect="auto",
                    cmap=ListedColormap([HP_COL["Xa"], HP_COL["Xi"]]), vmin=0, vmax=1, extent=(0, 1, n, 0))
    ax[0, 0].set_xticks([]); ax[0, 0].set_ylabel(f"{n} phased fibers: {n_xa} Xa (HP1), {n - n_xa} Xi (HP2)")
    ax[1, 0].axis("off")
    heat(ax[0, 1], input_image(a0, rows["DEV4NF"], lo, hi, cpg=True), xs[0], xs[-1] + 1, 0, 4, IN_CMAP,
         interpolation="antialiased")
    cpg_dots(ax[0, 1], a0, rows["DEV4NF"], lo, hi, xs[0])
    ax[0, 1].set_title("measured input\n(CpG calls on MSP / nucleosome)", fontsize=9)
    cg = a0["inputs"][rows["DEV4NF"], CPG, lo:hi].astype(float)
    called = cg != 0
    for h in ["Xa", "Xi"]:
        k = hapl == h
        meth = ((cg[k] == 1).sum(0), called[k].sum(0))
        m = pd.Series(meth[0]).rolling(201, center=True, min_periods=1).sum() / \
            pd.Series(meth[1]).rolling(201, center=True, min_periods=1).sum().replace(0, np.nan)
        ax[1, 1].plot(xs, m, color=HP_COL[h], lw=1.5)
    ax[1, 1].set_ylabel("CpG methylation (200 bp)"); ax[1, 1].set_ylim(0, 1); ax[1, 1].set_title("measured", fontsize=9)
    for j, (assay, lab, exp) in enumerate(XI_MARKS):
        a = ax[0, j + 2]
        img = pred_image(a0, runs["DEV4NF"][1], assay, rows["DEV4NF"], lo, hi)
        vmin, vmax = shared_limits([img])
        im = heat(a, img, xs[0], xs[-1] + 1, vmin, vmax)
        if s.group == "XIST":
            exp = exp.replace("higher", "HIGHER").replace("lower", "higher").replace("HIGHER", "lower") + " (reversed)"
        elif s.group != "silenced CGI":
            exp = "no difference"
        a.set_title(f"DEV4NF predicted {lab}\n(expected {exp})", fontsize=9)
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.05, pad=0.02).ax.tick_params(labelsize=7)
        b = ax[1, j + 2]
        for label, ls in [("DEV4NF", "-"), ("DEV4", "--")]:
            p = pred_image(runs[label][0][name], runs[label][1], assay, rows[label], lo, hi)
            for h in ["Xa", "Xi"]:
                b.plot(xs, nanmean_rows(p[hapl == h]), color=HP_COL[h], lw=1.5 if ls == "-" else 1.1, ls=ls)
        b.set_title("mean per haplotype", fontsize=8)
    for a in list(ax[0, 1:]) + list(ax[1, 1:]):
        a.axvspan(-TSS_HALF, TSS_HALF, color="#000000", alpha=0.04, lw=0)
        a.axvline(0, color="k", lw=0.6, ls=":")
        a.set_xlim(xs[0], xs[-1] + 1)
    for a in ax[0, 1:]:
        a.axhline(n_xa, color="k", lw=1.2)
        a.set_xticks([])
    for a in ax[1, 1:]:
        style(a); a.set_xlabel(f"bp from TSS (genomic orientation, strand {s.strand})", fontsize=8)
        a.tick_params(labelsize=7)
    hl = ("HP1", "HP2") if chrom != "chrX" else ("Xa (HP1)", "Xi (HP2)")
    handles = [Line2D([], [], color=HP_COL["Xa"], lw=2, label=hl[0]), Line2D([], [], color=HP_COL["Xi"], lw=2, label=hl[1]),
               Line2D([], [], color="#606060", lw=1.5, label="DEV4NF"), Line2D([], [], color="#606060", lw=1.1, ls="--", label="DEV4"),
               Line2D([], [], marker="o", ls="", color=CPG_COL["methylated"], label="methylated CpG"), Line2D([], [], marker="o", ls="", markerfacecolor=CPG_COL["unmethylated"], markeredgecolor="#555555", label="unmethylated CpG"),
               Patch(color=IN_COLORS[2], label="MSP"), Patch(color=IN_COLORS[3], label="nucleosome")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    what = {"silenced CGI": "silenced gene, CGI promoter", "escape": "escape gene (both copies active)",
            "chr21": "chr21 control (HP1 / HP2 arbitrary)", "XIST": "XIST (active only on the Xi)"}[s.group]
    fig.suptitle(f"{s.gene}: {what}; {chrom}:{tss:,} TSS, GM12878 phased fibers; shaded = TSS ± 500 bp "
                 f"(the test window); heatmaps seed 919", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"xi_{s.gene}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    rec = []
    t0, t1 = tss - TSS_HALF - ws, tss + TSS_HALF - ws
    for label, _, _ in RUNS:
        a = runs[label][0][name]
        for assay, lab, _ in XI_MARKS:
            p = pred_image(a, runs[label][1], assay, rows[label], t0, t1)
            v = np.nanmean(p, 1)
            rec.append({"gene": s.gene, "group": s.group, "run": label, "assay": lab,
                        "d_xa_minus_xi": float(np.nanmean(v[hapl == "Xa"]) - np.nanmean(v[hapl == "Xi"]))})
    return rec


def xi_summary(rec):
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(OUT, "xi_loci_summary.tsv"), sep="\t", index=False, float_format="%.4g")
    genes = list(dict.fromkeys(df.gene))
    labels = [l for l, _, _ in RUNS]
    mk = {"DEV4": ("o", "#eb6834"), "DEV4NF": ("o", "#2a78d6"), "DEV4 seed 2": ("s", "#eb6834"),
          "DEV4NF seed 2": ("s", "#2a78d6")}
    fig, ax = plt.subplots(1, len(XI_MARKS), figsize=(4 * len(XI_MARKS), 4), sharey=False)
    for k, (_, lab, exp) in enumerate(XI_MARKS):
        a = ax[k]
        for j, l in enumerate(labels):
            v = [df[(df.gene == g) & (df.run == l) & (df.assay == lab)].d_xa_minus_xi.iloc[0] for g in genes]
            m, c = mk[l]
            a.plot(np.arange(len(genes)) + (j - 1.5) * 0.12, v, m, color=c, ms=6, mec="white", mew=0.8,
                   alpha=1 if "seed" not in l else 0.75)
        a.axhline(0, color="#999999", lw=0.8)
        a.set_xticks(range(len(genes))); a.set_xticklabels(genes, rotation=35, ha="right", fontsize=8)
        a.set_title(f"{lab} (expected {exp} at silenced genes)", fontsize=9)
        style(a)
    ax[0].set_ylabel("Xa − Xi, mean over TSS ± 500 bp")
    fig.legend(handles=[Line2D([], [], marker=mk[l][0], ls="", color=mk[l][1], label=l) for l in labels],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xi_loci_summary.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(df.pivot_table(index=["gene", "assay"], columns="run", values="d_xa_minus_xi", sort=False).round(2), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = pd.read_csv(os.path.join(OUT, "selection.tsv"), sep="\t")
    runs = load_all("loci_ctcf")
    rec, shape = [], []
    for name in sel[sel.test == "ctcf"].name:
        r, sh = ctcf_site(runs, name, sel)
        rec += r
        shape += sh
        print(f"wrote ctcf_{name}.png", flush=True)
    ctcf_summary(rec)
    ctcf_shape(shape)
    print(pd.DataFrame(rec).groupby("site").bulk_ctcf_motif.first().round(2), flush=True)
    runs = load_all("loci_tss")
    rec = []
    for name in sel[sel.test == "xi"].name:
        rec += xi_gene(runs, name, sel)
        print(f"wrote xi for {name}", flush=True)
    xi_summary(rec)


if __name__ == "__main__":
    main()
