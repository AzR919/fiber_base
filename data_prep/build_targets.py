"""
Target BEDs for the single-molecule tests (reports/TODO.md section 2), read by molecule_predict.py. Built from
annotations only: reference, GENCODE, UCSC CpG islands, ENCODE CTCF peaks, the JASPAR CTCF motif and (poised tiles)
the ENCODE H3K4me3 / H3K27me3 fold-change tracks (no fibers, no model). Not part of setup_data.py; run it when the spec changes.

Reads data_prep/targets_spec.yaml (rules in its comments) and writes into <regions_root>/targets/:
  ctcf_sites_<chroms>.bed   CTCF motif sites (T5e)     chrom start end name score strand set motif_p
                            name = ctcf_<set>_<i>,        GM12878 K562 HepG2 H1
                            score = motif log-odds (bits), set = bound / unbound,
                            <cell> = that line's peak signalValue at the site (0 = no peak)
  tss_<chroms>.bed          protein-coding TSSs (T5b,  chrom start end name score strand gene class cgi
                            T5c), TSS +- tss_half       class = par / escape / xist / subject (chrX) or autosomal
  poised_tiles_<chroms>.bed tiles for poised vs        chrom start end name score strand cgi tss
                            Polycomb (T5g), classed     [class_<cell> k4_<cell> k27_<cell>] per cell line
                            per cell line from bulk     class = poised (H3K4me3 and H3K27me3 marked), polycomb
                            H3K4me3 / H3K27me3          (H3K27me3 only), active (H3K4me3 only), neither, mid;
                                                        k4 / k27 = mean FC over the tile; cgi / tss = overlaps a
                                                        CpG island / within tss_reach of a protein-coding TSS
  poised_fractions_<chroms>.tsv  genome fractions: tiles per cell, chromosome and class (all fraction_chroms)
  manifest.json             spec, input md5s, motif threshold, counts per file and class

Each BED's first line is '#chrom start end name score strand <extra columns>' (molecule_predict.py names the extra
columns from it).

Usage (from the repo root, on nibi; a few minutes on a login node for chr21 + chrX):
    python data_prep/build_targets.py [--spec data_prep/targets_spec.yaml]
"""

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pysam
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_regions import repo_path, md5_file, _now

CELLS_ORDER = ["GM12878", "K562", "HepG2", "H1"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="data_prep/targets_spec.yaml")
    p.add_argument("--sets", nargs="+", default=["ctcf", "tss", "poised"], choices=["ctcf", "tss", "poised"],
                   help="Which target sets to (re)build; the others keep their files and manifest entries")
    return p.parse_args()


def load_yaml(path):
    with open(repo_path(path)) as f:
        return yaml.safe_load(f)


def write_targets(path, df):
    """BED with a '#col1\\tcol2...' first line naming every column."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("#" + "\t".join(df.columns) + "\n")
        df.to_csv(f, sep="\t", header=False, index=False)
    os.replace(tmp, path)


def chrom_tag(chroms):
    return "_".join(chroms)


#--------------------------------------------------------------------------------------------------
# Motif: JASPAR counts -> log-odds PWM, exact p-value threshold, genome scan

def read_jaspar(path):
    """(L, 4) counts in A, C, G, T order from a JASPAR-format file."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            base, vals = line[0], re.sub(r"[\[\]]", " ", line[1:]).split()
            rows[base] = [float(v) for v in vals]
    return np.array([rows[b] for b in "ACGT"]).T


def log_odds(counts, pseudocount):
    """log2(p / 0.25) per position and base, with pseudocount mass spread evenly (uniform background)."""
    probs = (counts + pseudocount / 4) / (counts.sum(1, keepdims=True) + pseudocount)
    return np.log2(probs / 0.25)


SCALE = 100   # score grid for exact p-values: 1/SCALE bits


def score_tail(pwm):
    """(tail, offset): tail[k] = P(score >= (offset + k) / SCALE) for a uniform random sequence, exact on the grid
    (the score distribution is the convolution of the per-position distributions)."""
    w = np.round(pwm * SCALE).astype(np.int64)
    dist, offset = np.ones(1), 0
    for col in w:
        cmin = int(col.min())
        new = np.zeros(len(dist) + int(col.max()) - cmin)
        for v in col:
            new[int(v) - cmin:int(v) - cmin + len(dist)] += dist * 0.25
        dist, offset = new, offset + cmin
    return np.cumsum(dist[::-1])[::-1], offset


def score_threshold(tail, offset, pvalue):
    """Smallest grid score s with P(score >= s) <= pvalue, and that probability."""
    k = int(np.argmax(tail <= pvalue))
    return (offset + k) / SCALE, float(tail[k])


def score_pvalue(tail, offset, s):
    """P(score >= s) on the grid."""
    return float(tail[int(np.clip(np.floor(s * SCALE + 1e-6) - offset, 0, len(tail) - 1))])


_INDEX = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    _INDEX[_b] = _INDEX[_b + 32] = _i


def scan(seq, pwm):
    """Best-strand motif score at every start position of seq: (scores, strands) with strand +1 / -1.
    Positions whose motif window contains N get -inf."""
    idx = _INDEX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    L = len(pwm)
    n = len(idx) - L + 1
    if n <= 0:
        return np.zeros(0), np.zeros(0, np.int8)
    fw = np.vstack([pwm.T, np.full((1, L), -np.inf)])          # (5, L): row = base, N -> -inf
    rc_pwm = pwm[::-1, ::-1]                                     # reverse complement (A<->T, C<->G, reversed)
    rc = np.vstack([rc_pwm.T, np.full((1, L), -np.inf)])
    f = np.zeros(n, np.float32)
    r = np.zeros(n, np.float32)
    for j in range(L):
        f += fw[idx[j:j + n], j]
        r += rc[idx[j:j + n], j]
    strand = np.where(f >= r, 1, -1).astype(np.int8)
    return np.maximum(f, r), strand


#--------------------------------------------------------------------------------------------------
# CTCF sites

def read_peaks(path, chroms):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 6], names=["chrom", "start", "end", "signal"])
    return df[df.chrom.isin(chroms)].reset_index(drop=True)


def build_ctcf(spec, fasta):
    c = spec["ctcf"]
    chroms = c["chroms"]
    ann = spec["annotations"]
    pwm = log_odds(read_jaspar(os.path.join(ann, spec["ctcf_motif"])), c["pseudocount"])
    L = len(pwm)
    tail, offset = score_tail(pwm)
    thr, p_at_thr = score_threshold(tail, offset, c["motif_pvalue"])
    print(f"CTCF motif: {L} bp, max {pwm.max(1).sum():.1f} bits, threshold {thr:.2f} bits (p = {p_at_thr:.2e})",
          flush=True)
    peaks = {cell: read_peaks(os.path.join(ann, spec["ctcf_peaks"][cell]), chroms) for cell in spec["ctcf_peaks"]}
    cells = [x for x in CELLS_ORDER if x in peaks] + [x for x in peaks if x not in CELLS_ORDER]

    # bound: best hit per peak, merged across cells by motif position
    sites = {}
    for cell in cells:
        for p in peaks[cell].itertuples(index=False):
            s, st = scan(fasta.fetch(p.chrom, p.start, p.end), pwm)
            if len(s) == 0 or s.max() < thr:
                continue
            k = int(np.argmax(s))
            key = (p.chrom, p.start + k)
            site = sites.setdefault(key, {"score": float(s[k]), "strand": "+" if st[k] > 0 else "-",
                                          **{x: 0.0 for x in cells}})
            site[cell] = max(site[cell], float(p.signal))
    bound = pd.DataFrame([{"chrom": k[0], "start": k[1], **v} for k, v in sites.items()])
    print(f"  bound sites: {len(bound)} (per cell: " +
          ", ".join(f"{x} {int((bound[x] > 0).sum())}/{len(peaks[x])} peaks" for x in cells) + ")", flush=True)

    # unbound: genome hits on the same chromosomes outside every peak (+- margin), score-matched to bound
    rng = np.random.default_rng(c["seed"])
    cand = []
    for chrom in chroms:
        seq = fasta.fetch(chrom)
        s, st = scan(seq, pwm)
        hit = np.flatnonzero(s >= thr)
        excl = np.zeros(len(seq) + 1, np.int32)
        for cell in cells:
            pk = peaks[cell][peaks[cell].chrom == chrom]
            np.add.at(excl, np.clip(pk.start.values - c["unbound_margin"], 0, len(seq)), 1)
            np.add.at(excl, np.clip(pk.end.values + c["unbound_margin"], 0, len(seq)), -1)
        covered = np.cumsum(excl)[:len(seq)] > 0
        hit = hit[~covered[hit] & ~covered[np.minimum(hit + L - 1, len(seq) - 1)]]
        cand.append(pd.DataFrame({"chrom": chrom, "start": hit, "score": s[hit].astype(float),
                                  "strand": np.where(st[hit] > 0, "+", "-")}))
    cand = pd.concat(cand, ignore_index=True)
    b_bin = np.floor(bound.score / c["score_bin"]).astype(int)
    c_bin = np.floor(cand.score / c["score_bin"]).astype(int)
    picks, short = [], 0
    for b, n_b in b_bin.value_counts().items():
        pool = np.flatnonzero(c_bin.values == b)
        want = n_b * c["unbound_per_bound"]
        short += max(0, want - len(pool))
        picks.extend(rng.choice(pool, size=min(want, len(pool)), replace=False))
    unbound = cand.iloc[sorted(picks)].assign(**{x: 0.0 for x in cells})
    print(f"  unbound candidates: {len(cand)}; sampled {len(unbound)} score-matched"
          + (f" ({short} short in high-score bins)" if short else ""), flush=True)

    df = pd.concat([bound.assign(set="bound"), unbound.assign(set="unbound")], ignore_index=True)
    df = df.sort_values(["chrom", "start"], kind="stable").reset_index(drop=True)
    df["end"] = df.start + L
    df["name"] = [f"ctcf_{s}_{i}" for i, s in enumerate(df["set"])]
    df["score"] = df.score.round(2)
    # motif p-value of each site's score, from the same exact distribution (for later stratification)
    df["motif_p"] = [f"{score_pvalue(tail, offset, s):.1e}" for s in df.score]
    for x in cells:
        df[x] = df[x].round(1)
    out = df[["chrom", "start", "end", "name", "score", "strand", "set", "motif_p"] + cells]
    info = {"motif_length": L, "threshold_bits": thr, "threshold_p": p_at_thr,
            "counts": {"bound": int((out["set"] == "bound").sum()), "unbound": int((out["set"] == "unbound").sum()),
                       **{f"bound_{x}": int((out[x] > 0).sum()) for x in cells}},
            "unbound_candidates": int(len(cand)), "unbound_short": int(short)}
    return out, info


#--------------------------------------------------------------------------------------------------
# TSSs

def read_genes(gtf, chroms):
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene" or p[0] not in chroms:
                continue
            attr = p[8]
            gtype = re.search(r'gene_type "([^"]+)"', attr).group(1)
            name = re.search(r'gene_name "([^"]+)"', attr).group(1)
            if gtype != "protein_coding" and name != "XIST":
                continue
            start, end, strand = int(p[3]) - 1, int(p[4]), p[6]
            rows.append({"chrom": p[0], "tss": start if strand == "+" else end - 1, "strand": strand, "gene": name})
    return pd.DataFrame(rows)


def read_cgi(path, chroms):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3], names=["chrom", "start", "end"])
    return df[df.chrom.isin(chroms)]


def build_tss(spec, fasta):
    t = spec["tss"]
    chroms, half = t["chroms"], t["tss_half"]
    ann = spec["annotations"]
    genes = read_genes(os.path.join(ann, spec["gtf"]), chroms)
    genes = genes.drop_duplicates("gene").reset_index(drop=True)
    escape = set(t["escape_genes"])
    missing = sorted(escape - set(genes.gene))
    if missing:
        print(f"  escape genes not found as protein-coding genes: {missing}", flush=True)

    def gene_class(g):
        if g.chrom != "chrX":
            return "autosomal"
        if any(g.chrom == c and s <= g.tss < e for c, s, e in t["par"]):
            return "par"
        if g.gene == "XIST":
            return "xist"
        return "escape" if g.gene in escape else "subject"

    genes["class"] = [gene_class(g) for g in genes.itertuples(index=False)]
    genes["start"] = genes.tss - half
    genes["end"] = genes.tss + half
    cgi = read_cgi(os.path.join(ann, spec["cpg_islands"]), chroms)
    genes["cgi"] = 0
    for chrom, g in genes.groupby("chrom"):
        ci = cgi[cgi.chrom == chrom]
        cs, ce = np.sort(ci.start.values), ci.end.values[np.argsort(ci.start.values)]
        k = np.searchsorted(cs, g.end.values)                # islands starting before the target end
        # an island overlaps if any island with start < target end has end > target start
        prefix_max = np.maximum.accumulate(ce) if len(ce) else ce
        ov = np.array([kk > 0 and prefix_max[kk - 1] > s for kk, s in zip(k, g.start.values)])
        genes.loc[g.index, "cgi"] = ov.astype(int)
    genes["name"] = [f"tss_{g}" for g in genes.gene]
    genes["score"] = 0
    order = {c: i for i, c in enumerate(chroms)}
    genes = genes.assign(_o=genes.chrom.map(order)).sort_values(["_o", "start"], kind="stable")
    out = genes[["chrom", "start", "end", "name", "score", "strand", "gene", "class", "cgi"]]
    counts = {f"{c}_{k}": int(v) for (c, k), v in out.groupby(["chrom", "class"]).size().items()}
    counts.update({f"{c}_cgi": int(v) for c, v in out.groupby("chrom").cgi.sum().items()})
    print("  TSS counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    return out, {"counts": counts, "escape_missing": missing}


#--------------------------------------------------------------------------------------------------
# Poised vs Polycomb (T5g): 1 kb tiles classed per cell line from bulk H3K4me3 / H3K27me3

POISED_CLASSES = ["poised", "polycomb", "active", "neither"]


def overlaps(starts, ends, iv_starts, iv_ends):
    """(n,) bool: does each [start, end) overlap any interval (intervals may overlap each other)."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    out = np.zeros(len(starts), dtype=bool)
    if len(iv_starts) == 0:
        return out
    order = np.argsort(iv_starts)
    s, e = np.asarray(iv_starts)[order], np.maximum.accumulate(np.asarray(iv_ends)[order])
    k = np.searchsorted(s, ends)                      # intervals starting before the tile end
    ok = k > 0
    out[ok] = e[k[ok] - 1] > starts[ok]
    return out


def tile_means(path, chrom, n, tile):
    """(n,) mean fold change per tile from the chromosome start (no data -> 0), from the base-level values in chunks
    (bw.stats(exact=True) is far too slow on whole chromosomes)."""
    import pyBigWig
    bw = pyBigWig.open(path)
    out = np.zeros(n, dtype=np.float64)
    step = 10000                                     # tiles per chunk
    try:
        for i in range(0, n, step):
            m = min(step, n - i)
            v = np.asarray(bw.values(chrom, i * tile, (i + m) * tile, numpy=True), dtype=np.float64)
            out[i:i + m] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).reshape(m, tile).mean(axis=1)
    finally:
        bw.close()
    return out


def poised_class(k4, k27, marked, unmarked):
    """Per tile: poised (both marked), polycomb (K27 only), active (K4 only), neither, or mid (in between)."""
    cls = np.full(len(k4), "mid", dtype=object)
    hi4, lo4, hi27, lo27 = k4 >= marked, k4 < unmarked, k27 >= marked, k27 < unmarked
    cls[hi4 & hi27], cls[lo4 & hi27], cls[hi4 & lo27], cls[lo4 & lo27] = POISED_CLASSES
    return cls


def build_poised(spec, rspec, fasta):
    """Tiles of `tile` bp on every fraction chromosome (excluding blacklist / N runs, <regions_root>/exclude.bed),
    classed per cell line from the mean FC of its bulk H3K4me3 and H3K27me3. Returns the target table (sampled
    tiles on the target chromosomes), the genome-fraction table and info."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_loci import cell_tracks
    p = spec["poised"]
    tile, marked, unmarked = p["tile"], p["marked"], p["unmarked"]
    encode_root = load_yaml(p["encode_spec"])["encode_root"]
    tracks = cell_tracks(encode_root, p["cells"], ["h3k4me3", "h3k27me3"])
    ann = spec["annotations"]
    frac_chroms = p["fraction_chroms"] or rspec["chroms"]
    excl = pd.read_csv(os.path.join(rspec["regions_root"], "exclude.bed"), sep="\t", comment="#", header=None,
                       usecols=[0, 1, 2], names=["chrom", "start", "end"])
    cgi = read_cgi(os.path.join(ann, spec["cpg_islands"]), frac_chroms)
    genes = read_genes(os.path.join(ann, spec["gtf"]), p["chroms"])
    frac_rows, tiles = [], []
    for chrom in frac_chroms:
        n = fasta.get_reference_length(chrom) // tile
        st = np.arange(n, dtype=np.int64) * tile
        en = st + tile
        ex = excl[excl.chrom == chrom]
        valid = ~overlaps(st, en, ex.start.values, ex.end.values)
        ci = cgi[cgi.chrom == chrom]
        in_cgi = overlaps(st, en, ci.start.values, ci.end.values)
        df = pd.DataFrame({"chrom": chrom, "start": st, "end": en, "cgi": in_cgi.astype(int)})
        for cell in p["cells"]:
            k4 = tile_means(tracks[cell]["h3k4me3"], chrom, n, tile)
            k27 = tile_means(tracks[cell]["h3k27me3"], chrom, n, tile)
            df[f"k4_{cell}"], df[f"k27_{cell}"] = np.round(k4, 3), np.round(k27, 3)
            df[f"class_{cell}"] = poised_class(k4, k27, marked, unmarked)
            for c in POISED_CLASSES + ["mid"]:
                sel = valid & (df[f"class_{cell}"].values == c)
                frac_rows.append({"cell": cell, "chrom": chrom, "class": c, "n_tiles": int(sel.sum()),
                                  "n_cgi": int((sel & in_cgi).sum()), "n_valid": int(valid.sum()),
                                  "n_valid_cgi": int((valid & in_cgi).sum())})
        print(f"  {chrom}: {int(valid.sum())} valid tiles", flush=True)
        if chrom in p["chroms"]:
            g = genes[genes.chrom == chrom]
            df["tss"] = overlaps(st, en, g.tss.values - p["tss_reach"], g.tss.values + p["tss_reach"] + 1).astype(int)
            tiles.append(df[valid])
    tiles = pd.concat(tiles, ignore_index=True)
    rng = np.random.default_rng(p["seed"])
    pick = np.zeros(len(tiles), dtype=bool)
    for cell in p["cells"]:
        for c in POISED_CLASSES:
            idx = np.flatnonzero(tiles[f"class_{cell}"].values == c)
            if len(idx) > p["per_class"]:
                idx = rng.choice(idx, p["per_class"], replace=False)
            pick[idx] = True
    out = tiles[pick].copy()
    out["name"] = [f"tile_{c}_{s}" for c, s in zip(out.chrom, out.start)]
    out["score"], out["strand"] = 0, "."
    cols = ["chrom", "start", "end", "name", "score", "strand", "cgi", "tss"]
    for cell in p["cells"]:
        cols += [f"class_{cell}", f"k4_{cell}", f"k27_{cell}"]
    out = out[cols]
    fr = pd.DataFrame(frac_rows)
    counts = {f"{cell}_{c}": int((out[f"class_{cell}"] == c).sum()) for cell in p["cells"] for c in POISED_CLASSES}
    tot = fr.groupby(["cell", "class"])[["n_tiles", "n_valid"]].sum()
    for cell in p["cells"]:
        print(f"  {cell} genome fractions: " + ", ".join(
            f"{c} {tot.loc[(cell, c), 'n_tiles'] / tot.loc[(cell, c), 'n_valid']:.4f}" for c in POISED_CLASSES + ["mid"]),
            flush=True)
    print(f"  {len(out)} target tiles; per cell and class: {counts}", flush=True)
    return out, fr, {"counts": counts, "n_targets": int(len(out)), "tracks": tracks}


#--------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    spec = load_yaml(args.spec)
    rspec = load_yaml(spec["regions_spec"])
    out_dir = os.path.join(rspec["regions_root"], "targets")
    os.makedirs(out_dir, exist_ok=True)
    fasta = pysam.FastaFile(rspec["fasta_path"])
    ann = spec["annotations"]
    mpath = os.path.join(out_dir, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest = {"built": _now(), "spec": spec, "script_md5": md5_file(os.path.abspath(__file__)),
                "inputs": {os.path.basename(p): md5_file(p) for p in
                           [os.path.join(ann, spec[k]) for k in ("gtf", "cpg_islands", "ctcf_motif")] +
                           [os.path.join(ann, v) for v in spec["ctcf_peaks"].values()]},
                "files": dict(old.get("files", {}))}   # sets not rebuilt keep their entries

    if "ctcf" in args.sets:
        ctcf, info = build_ctcf(spec, fasta)
        name = f"ctcf_sites_{chrom_tag(spec['ctcf']['chroms'])}.bed"
        write_targets(os.path.join(out_dir, name), ctcf)
        manifest["files"][name] = info

    if "tss" in args.sets:
        tss, info = build_tss(spec, fasta)
        name = f"tss_{chrom_tag(spec['tss']['chroms'])}.bed"
        write_targets(os.path.join(out_dir, name), tss)
        manifest["files"][name] = info

    if "poised" in args.sets:
        tiles, fractions, info = build_poised(spec, rspec, fasta)
        tag = chrom_tag(spec["poised"]["chroms"])
        name = f"poised_tiles_{tag}.bed"
        write_targets(os.path.join(out_dir, name), tiles)
        fname = f"poised_fractions_{tag}.tsv"
        fractions.to_csv(os.path.join(out_dir, fname), sep="\t", index=False)
        manifest["files"][name] = info
        manifest["files"][fname] = {"rows": int(len(fractions))}

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {', '.join(manifest['files'])} and manifest.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
