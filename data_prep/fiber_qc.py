"""
Fiber-seq data QC: per sample, library stats and how well its own accessibility agrees with ENCODE. Answers "is a
cell type's problem in its Fiber-seq library, or a mismatch between its fibers and its ENCODE tracks?". Not part of
setup_data.py (it needs the data built first and takes ~10-20 min on 8 CPUs); rerun when samples or tracks change.

Run from the repo root (CPU job: sbatch data_prep/slurm_fiber_qc.sh):
    python -u data_prep/fiber_qc.py [--windows 300] [--workers 8] [--samples GM12878 Panc1_200U ...] [--out DIR]

On the same chr21 windows (a seeded subset of test_chr21_mini2000) for each sample in configs/metapaths.yaml:
1. Library stats: fibers per 5 kb window (up to 300), scored span per fiber, m6A rate (methylated / all A-T calls),
   CpG methylation rate, fraction of the span in MSPs, in FIREs, in nucleosomes.
2. Agreement of the sample's own accessibility with ENCODE: per 1 kb bin, fiber accessibility (mean over the fibers
   covering the bin of the fraction of the bin inside an MSP / a FIRE) against asinh(mean FC) of every cell line's
   ATAC and H3K4me3 track. A sample should agree best with its own line; if it agrees better with another line, its
   fibers and its ENCODE data do not describe the same cells.
Writes <out>/library_stats.tsv and <out>/agreement_<assay>_<f_fire|f_msp>.tsv (default out: DATA_FIBER/qc/, next to
the region files) and prints the tables.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BIN = 1000
MAX_FIBERS = 300


def windows(n):
    from regions import Regions
    from utils import load_metapaths
    m = load_metapaths("configs/metapaths.yaml")
    return Regions(m["regions_root"]).window_set("test_chr21_mini2000", max_regions=n, seed=919)


def sample_stats(args):
    """Library stats and per-bin fiber accessibility for one sample over the windows."""
    sample, wins = args
    import pyft
    import pysam
    from pathlib import Path
    from utils import load_metapaths
    from fiber_utils import get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp, get_fiber_data
    m = load_metapaths("configs/metapaths.yaml")
    bam = pyft.Fiberbam(str(Path(m["fiber_paths"][sample]).resolve()))
    fasta = pysam.FastaFile(m["fasta_path"])
    feats = [get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp]
    n_fib, span, rows = [], [], []
    tot = dict(m6a_pos=0, m6a_all=0, cpg_pos=0, cpg_all=0, msp=0, fire=0, nuc=0, scored=0)
    for chrom, s, e in wins:
        c = (s + e) // 2
        start, end = c - 2500, c + 2500
        x, _, n, cov, mask = get_fiber_data(bam, chrom, start, end, fasta, MAX_FIBERS, end - start, feats,
                                            min_overlap=(end - start) // 8)
        n_fib.append(n)
        if n == 0:
            for b in range(0, end - start, BIN):
                rows.append((chrom, start + b, np.nan, np.nan, 0))
            continue
        x, mask = x[:n], mask[:n]                       # (n, 5, L), (n, L)
        span.extend(mask.sum(1).tolist())
        m6a, cpg, msp, nuc, fire = (x[:, k] for k in range(5))
        tot["m6a_pos"] += int(((m6a == 1) & mask).sum()); tot["m6a_all"] += int(((m6a != 0) & mask).sum())
        tot["cpg_pos"] += int(((cpg == 1) & mask).sum()); tot["cpg_all"] += int(((cpg != 0) & mask).sum())
        tot["msp"] += int((msp * mask).sum()); tot["fire"] += int((fire * mask).sum())
        tot["nuc"] += int((nuc * mask).sum()); tot["scored"] += int(mask.sum())
        for b in range(0, end - start, BIN):
            mb = mask[:, b:b + BIN]
            covered = mb.sum(1) >= BIN // 5            # fibers covering >= 1/5 of the bin
            if covered.sum() < 5:
                rows.append((chrom, start + b, np.nan, np.nan, int(covered.sum())))
                continue
            denom = mb[covered].sum(1)
            f_msp = ((msp[covered, b:b + BIN] * mb[covered]).sum(1) / denom).mean()
            f_fire = ((fire[covered, b:b + BIN] * mb[covered]).sum(1) / denom).mean()
            rows.append((chrom, start + b, f_msp, f_fire, int(covered.sum())))
    lib = {"sample": sample, "windows": len(wins), "windows_no_fibers": int(sum(v == 0 for v in n_fib)),
           "fibers_per_window": float(np.mean(n_fib)), "frac_windows_at_cap": float(np.mean([v == MAX_FIBERS for v in n_fib])),
           "scored_span_bp": float(np.mean(span)) if span else np.nan,
           "m6a_rate": tot["m6a_pos"] / max(tot["m6a_all"], 1), "cpg_meth_rate": tot["cpg_pos"] / max(tot["cpg_all"], 1),
           "frac_msp": tot["msp"] / max(tot["scored"], 1), "frac_fire": tot["fire"] / max(tot["scored"], 1),
           "frac_nuc": tot["nuc"] / max(tot["scored"], 1)}
    bins = pd.DataFrame(rows, columns=["chrom", "bin_start", "f_msp", "f_fire", "n_fibers"])
    bins["sample"] = sample
    return lib, bins


def track_bins(bins, line_dirs, assay):
    """asinh(mean FC) per bin for every cell line that has the assay."""
    import json
    import pyBigWig
    out = {}
    keys = bins[["chrom", "bin_start"]].drop_duplicates()
    for line, d in line_dirs.items():
        assays = json.load(open(os.path.join(d, "manifest.json")))["assays"]
        if assay not in assays:
            continue
        bw = pyBigWig.open(os.path.join(d, assays[assay]["file"]))
        vals = [bw.stats(c, int(s), int(s) + BIN, type="mean", exact=True)[0] for c, s in keys.itertuples(index=False)]
        bw.close()
        out[line] = pd.Series(np.arcsinh(np.nan_to_num(np.array(vals, dtype=float))),
                              index=pd.MultiIndex.from_frame(keys))
    return pd.DataFrame(out)


def agreement(bins, tracks, col):
    """Pearson r between each sample's fiber accessibility (col) and each cell line's track, over shared bins."""
    rows = {}
    for sample, g in bins.groupby("sample", sort=False):
        g = g.dropna(subset=[col]).set_index(["chrom", "bin_start"])
        t = tracks.loc[g.index]
        rows[sample] = {line: float(np.corrcoef(g[col], t[line])[0, 1]) for line in tracks.columns}
    return pd.DataFrame(rows).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--samples", nargs="+", default=None,
                   help="Fiber-seq samples to check (default: every sample with ENCODE data in metapaths.yaml)")
    p.add_argument("--out", default=None, help="Output folder (default: DATA_FIBER/qc next to the region files)")
    a = p.parse_args()
    os.chdir(REPO_ROOT)
    from utils import load_metapaths
    m = load_metapaths("configs/metapaths.yaml")
    samples = [s for s in m["fiber_paths"] if s in (m.get("cell_types") or {})]
    if a.samples:
        unknown = sorted(set(a.samples) - set(samples))
        if unknown:
            sys.exit(f"Unknown samples (or without ENCODE data): {unknown}")
        samples = [s for s in samples if s in a.samples]
    out = a.out or os.path.join(os.path.dirname(m["regions_root"].rstrip("/")), "qc")
    line_of = {s: os.path.basename(m["cell_types"][s].rstrip("/")) for s in samples}
    line_dirs = {line_of[s]: m["cell_types"][s] for s in samples}
    wins = windows(a.windows)
    print(f"{len(samples)} samples, {len(wins)} windows of test_chr21_mini2000, {a.workers} workers", flush=True)

    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("forkserver")) as ex:
        results = list(ex.map(sample_stats, [(s, wins) for s in samples]))
    lib = pd.DataFrame([r[0] for r in results]).set_index("sample")
    lib.insert(0, "line", [line_of[s] for s in lib.index])
    bins = pd.concat([r[1] for r in results], ignore_index=True)

    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 3)
    lib.to_csv(f"{out}/library_stats.tsv", sep="\t")
    print("\n=== 1. Library stats (same windows for every sample) ===\n", lib, flush=True)

    for assay in ("atac", "h3k4me3"):
        tracks = track_bins(bins, line_dirs, assay)
        for col in ("f_fire", "f_msp"):
            ag = agreement(bins, tracks, col)
            ag.insert(0, "own_line", [line_of[s] for s in ag.index])
            own = [ag.loc[s, line_of[s]] if line_of[s] in ag.columns else np.nan for s in ag.index]
            best = ag.drop(columns="own_line").idxmax(axis=1)
            ag["r_own"], ag["best_line"] = own, best
            ag.to_csv(f"{out}/agreement_{assay}_{col}.tsv", sep="\t")
            print(f"\n=== 2. Fiber accessibility ({col}) vs ENCODE {assay}, Pearson r per 1 kb bin ===\n", ag, flush=True)
    print(f"\nWrote {out}/library_stats.tsv and {out}/agreement_*.tsv")


if __name__ == "__main__":
    main()
