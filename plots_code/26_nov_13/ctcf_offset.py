"""Nov 13 report: why every model gets CTCF backwards on K562-H1 (fiber-origin AUC 0.46-0.48, sign accuracy ~0.4).

82% of the K562-H1 CTCF differential bins are higher in K562, yet all models predict H1 >= K562 there. Two
explanations, tested here:
  (a) track scale: K562's CTCF ChIP is more enriched than H1's everywhere (antibody / library quality), so the
      "differential" bins are mostly shared CTCF sites at a different scale, which fibers cannot show;
  (b) the model's CTCF output follows fiber accessibility, and H1's fibers are more accessible at these sites.

Part 1 (tracks + model, login node, < 1 min):
  1. chr21 scale of each line's CTCF / H3K4me3 / ATAC track (bins >= min_signal, upper percentiles)
  2. at constitutive CTCF sites (all 4 dev lines >= min_signal): each line minus the 4-line mean
  3. at the K562-H1 CTCF differential bins: every line's level, whether H1 also has a peak there
  4. the models' predicted vs true K562 - H1 difference per assay (mean over bins), from the eval27 npz
Part 2 (--fibers, CPU job): fiber accessibility (fraction of the bin in MSPs, as data_prep/fiber_qc.py) of the
  K562_200U and H1_200U fibers at the CTCF differential bins and at constitutive sites, and how the models'
  predicted difference relates to the accessibility difference vs the true difference.

    python plots_code/26_nov_13/ctcf_offset.py [--fibers --workers 4]
Writes plots/26_nov_13/ctcf/ (tables as .tsv, figure ctcf_offset.png).
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

BIN, MIN_SIGNAL, MAX_FIBERS = 1000, 1.5, 300
DEV = ["GM12878", "K562", "HepG2", "H1"]
PAIR = ("K562", "H1")
SAMPLES = {"K562": "K562_200U", "H1": "H1_200U"}
EVAL = "eval27_pair_K562_H1_200U"
MODELS = {"Mix A": "MIX_V1_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train04",
          "Pair A": "PAIR_V1_data28_GM12878_HepG2_200U_model15_uct_mid_8A_train04",
          "Mix B": "MIX_V1_data35_mix_K562_H1_200U_model15_uct_mid_8A_train04",
          "Pair B": "PAIR_V1_data34_K562_H1_200U_model15_uct_mid_8A_train04",
          "DEV4": "LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04"}


def setup():
    from utils import load_metapaths
    m = load_metapaths("configs/metapaths.yaml")
    lines = {}
    for sample, d in (m.get("cell_types") or {}).items():
        lines.setdefault(os.path.basename(d.rstrip("/")), d)
    return m, lines


def manifest(d):
    return json.load(open(os.path.join(d, "manifest.json")))["assays"]


def chrom_bins(path, chrom, n_bins):
    import pyBigWig
    bw = pyBigWig.open(path)
    try:
        v = np.asarray(bw.values(chrom, 0, n_bins * BIN, numpy=True), dtype=np.float64)
    finally:
        bw.close()
    return np.arcsinh(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).reshape(n_bins, BIN).mean(1))


def track_table(lines, assays, chrom="chr21"):
    import pyBigWig
    any_path = next(os.path.join(d, a["file"]) for d in lines.values() for a in manifest(d).values())
    bw = pyBigWig.open(any_path)
    n_bins = bw.chroms()[chrom] // BIN
    bw.close()
    out = {}
    for line, d in lines.items():
        man = manifest(d)
        for a in assays:
            if a in man:
                out[(line, a)] = chrom_bins(os.path.join(d, man[a]["file"]), chrom, n_bins)
    return out, n_bins


def part1(m, lines, out_dir):
    tr, n_bins = track_table(lines, ["ctcf", "h3k4me3", "atac"])

    print("\n### 1. chr21 track scale (asinh of 1 kb bin means)\n")
    rows = []
    for (line, a), v in sorted(tr.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append({"assay": a, "line": line, "bins>=1.5": int((v >= MIN_SIGNAL).sum()),
                     "p99": np.percentile(v, 99), "p99.9": np.percentile(v, 99.9),
                     "top200_mean": np.sort(v)[-200:].mean()})
    scale = pd.DataFrame(rows)
    print(scale.to_string(index=False, float_format="%.2f"))
    scale.to_csv(os.path.join(out_dir, "track_scale.tsv"), sep="\t", index=False)

    print("\n### 2. Constitutive sites (all 4 dev lines >= 1.5): line minus the 4-line mean (asinh)\n")
    const = {}
    for a in ("ctcf", "h3k4me3"):
        V = np.stack([tr[(l, a)] for l in DEV])
        sites = np.all(V >= MIN_SIGNAL, axis=0)
        const[a] = np.where(sites)[0]
        dev = V[:, sites] - V[:, sites].mean(0)
        print(f"{a}: {sites.sum()} sites | " + ", ".join(
            f"{l} median {np.median(dev[i]):+.2f} (above mean at {np.mean(dev[i] > 0):.0%})" for i, l in enumerate(DEV)))
        kd = V[1, sites] - V[3, sites]
        print(f"   K562 - H1 at these sites: median {np.median(kd):+.2f}, K562 higher at {np.mean(kd > 0):.0%}, "
              f"|d| >= 0.5 at {np.mean(np.abs(kd) >= 0.5):.0%}")

    print("\n### 3. The K562-H1 CTCF differential bins\n")
    bed = pd.read_csv(os.path.join(m["regions_root"], "loci", "pairs", "K562__H1.bed"), sep="\t", comment="#",
                      header=None, names=["chrom", "start", "end", "region_id", "assay", "d", "direction",
                                          "bin_start"])
    c = bed[bed.assay == "ctcf"].copy()
    idx = (c.bin_start // BIN).to_numpy()
    for line in [l for l in lines if (l, "ctcf") in tr]:
        c[line] = tr[(line, "ctcf")][idx]
    hi = c[c.d > 0]
    lo = c[c.d < 0]
    print(f"{len(c)} bins: {len(hi)} higher in K562, {len(lo)} higher in H1")
    for name, g in (("K562-higher", hi), ("H1-higher", lo)):
        print(f"  {name}: median asinh K562 {g.K562.median():.2f}, H1 {g.H1.median():.2f}, GM12878 "
              f"{g.GM12878.median():.2f}, HepG2 {g.HepG2.median():.2f}; H1 also >= 1.5 at {np.mean(g.H1 >= 1.5):.0%}, "
              f"K562 also >= 1.5 at {np.mean(g.K562 >= 1.5):.0%}")
    dev4 = hi[DEV].to_numpy()
    print(f"  K562-higher bins: K562 is the highest of the 4 dev lines at {np.mean(dev4.argmax(1) == 1):.0%}, "
          f"H1 the lowest at {np.mean(dev4.argmin(1) == 3):.0%}")
    in_const = np.isin(idx, const["ctcf"])
    print(f"  bins that are constitutive sites (all 4 dev >= 1.5): {in_const.mean():.0%} "
          f"({in_const[c.d.to_numpy() > 0].mean():.0%} of the K562-higher ones)")
    c.to_csv(os.path.join(out_dir, "ctcf_bins.tsv"), sep="\t", index=False)

    print("\n### 4. Models: mean predicted vs mean true K562 - H1 at the differential bins, per assay\n")
    pred = {}
    for name, run in MODELS.items():
        path = sorted(glob.glob(f"results/*_eval_{run}*/{EVAL}_paired.npz"))[-1]
        z = np.load(path)
        assays = [str(a) for a in z["assays"]]
        cells = [str(x) for x in z["cells"]]
        row = {}
        for k, a in enumerate(assays):
            has = (z["tile_d"][:, k] != 0) & (z["tile_n_bin"][:, k] > 0)
            if has.sum() == 0:
                continue
            dt, dp = z["tile_dt_bin"][has, k], z["tile_dp_bin"][has, k]
            row[a] = f"{dp.mean():+.2f} / {dt.mean():+.2f}"
            if a == "ctcf":
                pred[name] = pd.DataFrame({"chrom": z["tile_chrom"][has], "start": z["tile_start"][has],
                                           "dt": dt, "dp": dp, "d": z["tile_d"][has, k]})
        print(f"{name} (cells {cells}): " + ", ".join(f"{a} {v}" for a, v in row.items()))
    return c, const, tr, pred


def bin_access(args):
    """f_msp per (chrom, tile_start, tile_end, bin_start) for one sample (as data_prep/fiber_qc.py)."""
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
    for chrom, s, e, b in items:
        x, _, n, cov, mask = get_fiber_data(bam, chrom, s, e, fasta, MAX_FIBERS, e - s, [get_msp],
                                            min_overlap=(e - s) // 8)
        o = b - s
        if n == 0:
            out.append((chrom, b, np.nan, 0)); continue
        msp, mb = x[:n, 0, o:o + BIN], mask[:n, o:o + BIN]
        covered = mb.sum(1) >= BIN // 5
        if covered.sum() < 5:
            out.append((chrom, b, np.nan, int(covered.sum()))); continue
        f = ((msp[covered] * mb[covered]).sum(1) / mb[covered].sum(1)).mean()
        out.append((chrom, b, float(f), int(covered.sum())))
    return sample, out


def part2(m, c, const, pred, out_dir, workers):
    tiles = pd.read_csv(os.path.join(m["regions_root"], "windows", "test_chr21_tiles.bed"), sep="\t", comment="#",
                        header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    rng = np.random.default_rng(919)
    # constitutive CTCF sites inside test tiles, 300 at random, as the reference
    tb = {}
    for t in tiles.itertuples(index=False):
        for b in range(t.start, t.end, BIN):
            tb[b // BIN] = (t.chrom, t.start, t.end, b)
    cs = [tb[i] for i in const["ctcf"] if i in tb]
    cs = [cs[i] for i in rng.choice(len(cs), size=min(300, len(cs)), replace=False)]
    items = [(r.chrom, int(r.start), int(r.end), int(r.bin_start)) for r in c.itertuples(index=False)] + cs
    print(f"\n### 5. Fiber accessibility: {len(c)} differential bins + {len(cs)} constitutive sites, "
          f"{len(SAMPLES)} samples", flush=True)
    with ProcessPoolExecutor(min(workers, 2 * len(SAMPLES)), mp_context=mp.get_context("forkserver")) as ex:
        chunks = [(s, items[i::2]) for s in SAMPLES.values() for i in range(2)]
        res = list(ex.map(bin_access, chunks))
    acc = {}
    for sample, rows in res:
        for chrom, b, f, n in rows:
            acc[(sample, chrom, b)] = f
    for line, s in SAMPLES.items():
        c[f"f_msp_{line}"] = [acc.get((s, r.chrom, int(r.bin_start)), np.nan) for r in c.itertuples()]
    const_df = pd.DataFrame(cs, columns=["chrom", "start", "end", "bin_start"])
    for line, s in SAMPLES.items():
        const_df[f"f_msp_{line}"] = [acc.get((s, r.chrom, int(r.bin_start)), np.nan) for r in const_df.itertuples()]
    c["df_msp"] = c.f_msp_K562 - c.f_msp_H1
    const_df["df_msp"] = const_df.f_msp_K562 - const_df.f_msp_H1
    for name, g in (("constitutive sites", const_df), ("K562-higher bins", c[c.d > 0]), ("H1-higher bins", c[c.d < 0])):
        print(f"  {name}: f_msp K562 {g.f_msp_K562.median():.3f}, H1 {g.f_msp_H1.median():.3f}; "
              f"K562 - H1 median {g.df_msp.median():+.3f}, H1 more accessible at {np.mean(g.df_msp < 0):.0%}")

    print("\n### 6. Predicted difference vs true difference and vs accessibility difference (CTCF bins)\n")
    for name, p in pred.items():
        j = c.merge(p, left_on=["chrom", "start"], right_on=["chrom", "start"], how="inner")
        j = j.dropna(subset=["df_msp"])
        r_true = np.corrcoef(j.dp, j.dt)[0, 1]
        r_acc = np.corrcoef(j.dp, j.df_msp)[0, 1]
        r_acc_true = np.corrcoef(j.df_msp, j.dt)[0, 1]
        print(f"  {name}: n {len(j)} | r(pred, true) {r_true:.2f} | r(pred, access diff) {r_acc:.2f} | "
              f"r(access diff, true) {r_acc_true:.2f}")
        j.to_csv(os.path.join(out_dir, f"ctcf_bins_{name.replace(' ', '_')}.tsv"), sep="\t", index=False)
    const_df.to_csv(os.path.join(out_dir, "ctcf_constitutive_access.tsv"), sep="\t", index=False)
    return c, const_df


def figure(tr, const, out_dir):
    """(a) constitutive CTCF sites: each line minus the 4 dev lines' mean; (b) true and (c) predicted (Pair B, Mix A)
    K562 - H1 difference at the CTCF differential bins against the fibers' accessibility difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    HI, LO = "#6250d6", "#e34948"   # higher in K562 / in H1, as in mix_bin_scatter
    path_b = os.path.join(out_dir, "ctcf_bins_Pair_B.tsv")
    if not os.path.exists(path_b):
        print("no part-2 outputs yet; run with --fibers first")
        return
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})

    ax = axes[0]
    sites = const["ctcf"]
    ref = np.stack([tr[(l, "ctcf")][sites] for l in DEV]).mean(0)
    order = DEV + sorted(l for (l, a) in tr if a == "ctcf" and l not in DEV)
    vals = [tr[(l, "ctcf")][sites] - ref for l in order]
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=TEXT, lw=1.5))
    for i, (b, l) in enumerate(zip(bp["boxes"], order)):
        b.set_facecolor(LO if l == "H1" else ("#b7b5ad" if l in DEV else "#e4e3df"))
        b.set_edgecolor(TEXT_2)
    ax.axhline(0, color=TEXT_2, lw=0.8, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right", rotation_mode="anchor", fontsize=10, color=TEXT)
    ax.set_ylabel("line − mean of the 4 dev lines (asinh FC)", color=TEXT_2)
    ax.set_title(f"(a) CTCF at {len(sites)} shared sites (chr21)\n", fontsize=12, fontweight="bold", color=TEXT,
                 loc="left")
    ax.text(0, 1.0, "sites with signal in all 4 dev lines; light: held out", transform=ax.transAxes, fontsize=9.5,
            color=TEXT_2, va="bottom")

    panels = [("dt", "(b) True K562 − H1", path_b),
              ("dp", "(c) Pair B predicted (trained on both)", path_b),
              ("dp", "(d) Mix A predicted (never saw them)", os.path.join(out_dir, "ctcf_bins_Mix_A.tsv"))]
    for ax, (col, title, path) in zip(axes[1:], panels):
        j = pd.read_csv(path, sep="\t").dropna(subset=["df_msp"])
        d = j["d_x"] if "d_x" in j else j["d"]   # loci d (the merge with the npz table suffixes it)
        for pos, colr in ((True, HI), (False, LO)):
            s = (d > 0) == pos
            ax.scatter(j.df_msp[s], j[col][s], s=7, alpha=0.45, color=colr, linewidth=0)
        b, a = np.polyfit(j.df_msp, j[col], 1)
        xs = np.linspace(j.df_msp.quantile(0.01), j.df_msp.quantile(0.99), 50)
        ax.plot(xs, a + b * xs, color=TEXT, lw=1.6)
        if col == "dp":
            bt, at = np.polyfit(j.df_msp, j.dt, 1)
            ax.plot(xs, at + bt * xs, color=TEXT, lw=1.2, ls="--")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel("fiber accessibility, K562 − H1 (MSP fraction)", color=TEXT_2)
        ax.set_title(title + "\n", fontsize=12, fontweight="bold", color=TEXT, loc="left")
        sub = f"fit: {a:+.2f} + {b:.1f}·x" + ("   (dashed: true fit)" if col == "dp" else "")
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=9.5, color=TEXT_2, va="bottom")
        if col == "dt":
            ax.set_ylabel("K562 − H1 at the bin (asinh)", color=TEXT_2)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=TEXT_2)
    fig.legend(handles=[Patch(color=HI, label="track higher in K562"), Patch(color=LO, label="track higher in H1")],
               loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.2, wspace=0.28)
    out = os.path.join(out_dir, "ctcf_offset.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fibers", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="plots/26_nov_13/ctcf")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    m, lines = setup()
    c, const, tr, pred = part1(m, lines, a.out_dir)
    if a.fibers:
        part2(m, c, const, pred, a.out_dir, a.workers)
    figure(tr, const, a.out_dir)


if __name__ == "__main__":
    main()
