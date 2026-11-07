"""Nov 13 report: the CTCF decomposition (ctcf_offset.py) for every assay on both pairs.

For each pair (GM12878-HepG2, K562-H1) and assay, at the pair's differential bins:
  1. track scale: each of the two lines minus the mean of the dev lines at shared sites (all dev lines with the
     assay >= min_signal on chr21); a line far from 0 has a globally weaker / stronger track
  2. fiber accessibility difference (MSP fraction, as data_prep/fiber_qc.py) of the two samples per bin
  3. for the true difference and each model's predicted difference: the mean over the bins (a predicted mean of the
     wrong sign = an offset between the two cell types) and the slope against the accessibility difference
     (predicted / true slope < 1 = compressed). The intercept difference ("shift" column in offsets.tsv) is kept
     but not reported: where the predicted slope is flatter, it mostly measures that compression extrapolated to
     zero accessibility difference.
Bins: up to --max_bins per assay (random, seed 919), all of them when fewer.

    python plots_code/26_nov_13/pair_offsets.py [--workers 8] [--max_bins 200]
Writes plots/26_nov_13/offsets/: offsets.tsv (one row per pair x assay x model), access_<pair>.tsv, pair_offsets.png.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctcf_offset import BIN, MIN_SIGNAL, MAX_FIBERS, DEV, MODELS, setup, track_table  # noqa: E402

PAIRS = {"GM12878–HepG2": {"cells": ("GM12878", "HepG2"), "samples": ("GM12878", "HepG2_200U"),
                           "bed": "GM12878__HepG2", "eval": "eval24_pair_GM12878_HepG2_200U"},
         "K562–H1": {"cells": ("K562", "H1"), "samples": ("K562_200U", "H1_200U"),
                     "bed": "K562__H1", "eval": "eval27_pair_K562_H1_200U"}}
ASSAYS = ["atac", "h3k4me3", "h3k27ac", "ctcf", "h3k4me1", "h3k36me3", "h3k27me3", "h3k9me3"]
LAB = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3", "h3k4me1": "H3K4me1",
       "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
TRAINED_ON = {"Mix A": "GM12878–HepG2", "Pair A": "GM12878–HepG2", "Mix B": "K562–H1", "Pair B": "K562–H1",
              "DEV4": None}


def tile_access(args):
    """MSP fraction per requested bin, reading each 5 kb tile once. items: [(chrom, start, end, [bin_starts])]."""
    sample, items = args
    import pyft
    import pysam
    from pathlib import Path
    from utils import load_metapaths
    from fiber_utils import get_msp, get_fiber_data
    m = load_metapaths("configs/metapaths.yaml")
    bam = pyft.Fiberbam(str(Path(m["fiber_paths"][sample]).resolve()))
    fasta = pysam.FastaFile(m["fasta_path"])
    out = []
    for chrom, s, e, bins in items:
        x, _, n, cov, mask = get_fiber_data(bam, chrom, s, e, fasta, MAX_FIBERS, e - s, [get_msp],
                                            min_overlap=(e - s) // 8)
        for b in bins:
            o = b - s
            if n == 0:
                out.append((chrom, b, np.nan)); continue
            msp, mb = x[:n, 0, o:o + BIN], mask[:n, o:o + BIN]
            cov_ = mb.sum(1) >= BIN // 5
            if cov_.sum() < 5:
                out.append((chrom, b, np.nan)); continue
            out.append((chrom, b, float(((msp[cov_] * mb[cov_]).sum(1) / mb[cov_].sum(1)).mean())))
    return sample, out


def shared_site_scale(tr, lines):
    """{(line, assay): median of line - mean of the dev lines at sites where every dev line with the assay >= 1.5}."""
    out = {}
    for a in ASSAYS:
        dev = [l for l in DEV if (l, a) in tr]
        V = np.stack([tr[(l, a)] for l in dev])
        sites = np.all(V >= MIN_SIGNAL, axis=0)
        ref = V[:, sites].mean(0)
        for l in lines:
            if (l, a) in tr:
                out[(l, a)] = (float(np.median(tr[(l, a)][sites] - ref)), int(sites.sum()), len(dev))
    return out


def fit(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_bins", type=int, default=200)
    ap.add_argument("--out_dir", default="plots/26_nov_13/offsets")
    ap.add_argument("--replot", action="store_true", help="tables and figures from the saved .tsv files only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.replot:
        df = pd.read_csv(os.path.join(a.out_dir, "offsets.tsv"), sep="\t")
        report(df)
        figure(df, a.out_dir)
        access_scatter(a.out_dir)
        return
    m, lines = setup()
    tr, _ = track_table(lines, ASSAYS)
    scale = shared_site_scale(tr, [c for p in PAIRS.values() for c in p["cells"]])
    rng = np.random.default_rng(919)

    # bins per pair x assay (subsampled), and the tiles to read per sample
    sel, jobs = {}, []
    for pname, P in PAIRS.items():
        bed = pd.read_csv(os.path.join(m["regions_root"], "loci", "pairs", f"{P['bed']}.bed"), sep="\t",
                          comment="#", header=None, names=["chrom", "start", "end", "region_id", "assay", "d",
                                                           "direction", "bin_start"])
        for asy in ASSAYS:
            g = bed[bed.assay == asy]
            if len(g) == 0:
                continue
            if len(g) > a.max_bins:
                g = g.iloc[np.sort(rng.choice(len(g), a.max_bins, replace=False))]
            sel[(pname, asy)] = g.reset_index(drop=True)
        tiles = {}
        for (pn, asy), g in sel.items():
            if pn != pname:
                continue
            for r in g.itertuples(index=False):
                tiles.setdefault((r.chrom, int(r.start), int(r.end)), set()).add(int(r.bin_start))
        items = [(c, s, e, sorted(b)) for (c, s, e), b in tiles.items()]
        per = max(1, a.workers // 2)
        for sample in P["samples"]:
            jobs += [(sample, items[i::per]) for i in range(per)]
        print(f"{pname}: {sum(len(g) for (pn, _), g in sel.items() if pn == pname)} bins in {len(items)} tiles",
              flush=True)

    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("forkserver")) as ex:
        res = list(ex.map(tile_access, jobs))
    acc = {}
    for sample, rows in res:
        for chrom, b, f in rows:
            acc[(sample, chrom, b)] = f

    # model predictions per bin from the paired npz (newest dir that has the eval)
    rows = []
    for pname, P in PAIRS.items():
        A, B = P["cells"]
        sa, sb = P["samples"]
        npz = {}
        for mname, run in MODELS.items():
            z = np.load(sorted(glob.glob(f"results/*_eval_{run}*/{P['eval']}_paired.npz"))[-1])
            npz[mname] = z
        z0 = next(iter(npz.values()))
        assays = [str(x) for x in z0["assays"]]
        key = pd.DataFrame({"chrom": z0["tile_chrom"].astype(str), "start": z0["tile_start"].astype(int)})
        dump = []
        for asy in ASSAYS:
            if (pname, asy) not in sel or asy not in assays:
                continue
            k = assays.index(asy)
            g = sel[(pname, asy)].copy()
            g["fa"] = [acc.get((sa, r.chrom, int(r.bin_start)), np.nan) for r in g.itertuples()]
            g["fb"] = [acc.get((sb, r.chrom, int(r.bin_start)), np.nan) for r in g.itertuples()]
            g["dacc"] = g.fa - g.fb
            k2i = {(c, s): i for i, (c, s) in enumerate(key.itertuples(index=False))}
            idx = np.array([k2i.get((r.chrom, int(r.start)), -1) for r in g.itertuples()])
            g = g[idx >= 0].copy(); idx = idx[idx >= 0]
            g["dt"] = z0["tile_dt_bin"][idx, k]
            for mname, z in npz.items():
                g[f"dp_{mname}"] = z["tile_dp_bin"][idx, k]
            g = g[(z0["tile_n_bin"][idx, k] > 0) & g.dacc.notna()]
            if len(g) < 10:
                continue
            at, bt = fit(g.dacc, g.dt)
            base = {"pair": pname, "assay": asy, "n": len(g), "pct_higher_A": float(np.mean(g.d > 0)),
                    f"scale_A": scale.get((A, asy), (np.nan,))[0], f"scale_B": scale.get((B, asy), (np.nan,))[0],
                    "dacc_mean": float(g.dacc.mean()), "r_acc_true": float(np.corrcoef(g.dacc, g.dt)[0, 1]),
                    "true_icpt": at, "true_slope": bt, "true_mean": float(g.dt.mean())}
            for mname in npz:
                ap_, bp_ = fit(g.dacc, g[f"dp_{mname}"])
                rows.append({**base, "model": mname,
                             "seen": "own" if TRAINED_ON[mname] == pname else ("dev4" if mname == "DEV4" else "unseen"),
                             "pred_icpt": ap_, "pred_slope": bp_, "pred_mean": float(g[f"dp_{mname}"].mean()),
                             "shift": ap_ - at,
                             "r_pred_true": float(np.corrcoef(g[f"dp_{mname}"], g.dt)[0, 1]),
                             "r_pred_acc": float(np.corrcoef(g[f"dp_{mname}"], g.dacc)[0, 1])})
            g["assay"] = asy
            dump.append(g)
        if dump:
            pd.concat(dump).to_csv(os.path.join(a.out_dir, f"access_{P['bed']}.tsv"), sep="\t", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "offsets.tsv"), sep="\t", index=False)
    report(df)
    figure(df, a.out_dir)
    access_scatter(a.out_dir)


def report(df):
    for pname, P in PAIRS.items():
        A, B = P["cells"]
        d = df[df.pair == pname]
        print(f"\n### {pname} (A = {A}, B = {B})\n")
        print("| Assay | bins | % higher in A | shared-site scale A / B | Δaccess mean | r(Δaccess, true) | "
              "true mean | predicted mean: Mix A / Pair A / Mix B / Pair B / DEV4 | slope pred/true, same order |")
        print("|---|---|---|---|---|---|---|---|---|")
        for asy in ASSAYS:
            s = d[d.assay == asy]
            if s.empty:
                continue
            r0 = s.iloc[0]
            pm = " / ".join(f"{s[s.model == mm]['pred_mean'].iloc[0]:+.2f}" for mm in MODELS)
            sl = " / ".join(f"{s[s.model == mm]['pred_slope'].iloc[0] / r0.true_slope:.2f}" for mm in MODELS)
            print(f"| {LAB[asy]} | {int(r0.n)} | {r0.pct_higher_A:.0%} | {r0.scale_A:+.2f} / {r0.scale_B:+.2f} | "
                  f"{r0.dacc_mean:+.3f} | {r0.r_acc_true:.2f} | {r0.true_mean:+.2f} | {pm} | {sl} |")


def access_scatter(out_dir):
    """Per pair: predicted A - B against the fibers' accessibility difference, per assay. Rows: the mixture model
    trained on this pair, the one trained on the other pair. Solid: fit; dashed: the true difference's fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    HI, LO = "#6250d6", "#e34948"
    for pname, P in PAIRS.items():
        path = os.path.join(out_dir, f"access_{P['bed']}.tsv")
        if not os.path.exists(path):
            continue
        g = pd.read_csv(path, sep="\t")
        A, B = P["cells"]
        assays = [x for x in ASSAYS if x in set(g.assay)]
        own = next(mm for mm, t in TRAINED_ON.items() if t == pname and mm.startswith("Mix"))
        other = next(mm for mm, t in TRAINED_ON.items() if t not in (pname, None) and mm.startswith("Mix"))
        fig, axes = plt.subplots(2, len(assays), figsize=(2.55 * len(assays), 5.9), squeeze=False)
        for r, (mm, tag) in enumerate(((own, "trained on this pair"), (other, "trained on the other pair"))):
            for c, asy in enumerate(assays):
                ax = axes[r][c]
                s = g[g.assay == asy]
                x, y = s.dacc.to_numpy(), s[f"dp_{mm}"].to_numpy()
                for pos, colr in ((True, HI), (False, LO)):
                    k = (s.d > 0).to_numpy() == pos
                    ax.scatter(x[k], y[k], s=5, alpha=0.4, color=colr, linewidth=0, rasterized=True)
                xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 50)
                ap_, bp_ = fit(x, y)
                at, bt = fit(x, s.dt.to_numpy())
                ax.plot(xs, ap_ + bp_ * xs, color=TEXT, lw=1.4)
                ax.plot(xs, at + bt * xs, color=TEXT, lw=1.0, ls="--")
                ax.axhline(0, color=GRID, lw=0.8, zorder=0)
                ax.axvline(0, color=GRID, lw=0.8, zorder=0)
                lim = max(np.nanpercentile(np.abs(s.dt), 99), np.nanpercentile(np.abs(y), 99), 0.5) * 1.05
                ax.set_ylim(-lim, lim)
                ax.set_title(f"{LAB[asy]}\nmean {np.mean(y):+.2f} (true {s.dt.mean():+.2f})", fontsize=9.5,
                             color=TEXT)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(labelsize=7.5, colors=TEXT_2)
                if c == 0:
                    ax.set_ylabel(f"{mm} ({tag})\npredicted {A} − {B}", fontsize=9.5, color=TEXT)
                if r == 1:
                    ax.set_xlabel(f"accessibility {A} − {B}", fontsize=8.5, color=TEXT_2)
        fig.legend(handles=[Patch(color=HI, label=f"track higher in {A}"), Patch(color=LO, label=f"track higher in {B}")],
                   loc="upper right", ncol=2, frameon=False, fontsize=10)
        fig.suptitle(f"{pname}: predicted difference vs fiber accessibility difference (MSP fraction) per differential "
                     f"bin; solid: fit, dashed: true difference's fit", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=TEXT)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.1, wspace=0.35, hspace=0.6)
        out = os.path.join(out_dir, f"access_scatter_{P['bed']}.png")
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"saved {out}")


def figure(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    COLOR = {"Mix": "#2a78d6", "Pair": "#eb6834", "DEV4": "#1baf7a"}
    TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5))
    w, gap = 0.15, 0.01
    for ax, pname in zip(axes, PAIRS):
        d = df[df.pair == pname]
        assays = [x for x in ASSAYS if x in set(d.assay)]
        order = [mm for mm in MODELS if TRAINED_ON[mm] == pname] + ["DEV4"] + \
                [mm for mm in MODELS if TRAINED_ON[mm] not in (pname, None)]
        for gi, asy in enumerate(assays):
            t = d[d.assay == asy].true_mean.iloc[0]
            ax.plot([gi - 2.6 * (w + gap), gi + 2.6 * (w + gap)], [t, t], color=TEXT, lw=2.2, zorder=5,
                    solid_capstyle="butt")
            for bi, mm in enumerate(order):
                v = d[(d.assay == asy) & (d.model == mm)]["pred_mean"].iloc[0]
                c = COLOR["DEV4" if mm == "DEV4" else mm.split()[0]]
                seen = TRAINED_ON[mm] in (pname, None)
                x = gi + (bi - 2) * (w + gap)
                if seen:
                    ax.bar(x, v, w, color=c, zorder=3, linewidth=0)
                else:
                    ax.bar(x, v, w, facecolor=matplotlib.colors.to_rgba(c, 0.18), edgecolor=c, hatch="////",
                           linewidth=0, zorder=3)
        ax.axhline(0, color=TEXT, lw=1.0, zorder=4)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=TEXT_2)
        A, B = PAIRS[pname]["cells"]
        sc = d.drop_duplicates("assay").set_index("assay")
        ax.set_xticks(range(len(assays)))
        ax.set_xticklabels([f"{LAB[x]}\n{A} {sc.loc[x, 'scale_A']:+.2f}\n{B} {sc.loc[x, 'scale_B']:+.2f}"
                            for x in assays], fontsize=9.5, color=TEXT)
        for t in ax.get_xticklabels():
            t.set_linespacing(1.3)
        ax.set_ylabel(f"mean {A} − {B} (asinh)", color=TEXT_2)
        ax.set_title(f"Tested on {pname}", fontsize=13, fontweight="bold", color=TEXT, loc="left")
    fig.suptitle("Mean difference between the two cell types over the differential bins: predicted (bars) vs "
                 "true (black line)", x=0.04, ha="left", fontsize=13, fontweight="bold", color=TEXT, y=0.995)
    fig.text(0.04, 0.955, "under each assay: shared-site track scale of each line (median line − dev-line mean; "
                          "far from 0 = a globally weak or strong track)", fontsize=10, color=TEXT_2)
    hs = [plt.Line2D([], [], color=TEXT, lw=2.2, label="true"),
          Patch(color=COLOR["Mix"], label="Mixtures only"), Patch(color=COLOR["Pair"], label="Own tracks"),
          Patch(color=COLOR["DEV4"], label="DEV4"),
          Patch(facecolor="white", edgecolor=TEXT_2, hatch="////", label="hatched: trained on the other pair")]
    fig.legend(handles=hs, loc="upper left", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.035, 0.94))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.11, hspace=0.62)
    out = os.path.join(out_dir, "pair_offsets.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
