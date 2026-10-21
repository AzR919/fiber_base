"""
Data setup step 3: cell-specific loci, i.e. tiles where one cell line's ENCODE track departs strongly from the
other cell lines' (leave-one-out), and pairwise differential loci between two cell lines (for mixture tests, T3).
Normally run through data_prep/setup_data.py, which runs every step.

Reads data_prep/loci_spec.yaml (heuristic in its comments) and writes into <regions_root>/loci/:
  <split>/table.tsv     every tile x cell line x assay with signal   chrom start end region_id cell assay d
                                                                      bin_start cell_asinh others_asinh n_others
  <split>/<cell>.bed    the cell line's cell-specific tiles           chrom start end region_id assay d direction
                        (|d| >= threshold), one row per tile and assay;  bin_start
                        direction = gain / loss, bin_start = the 1 kb bin d was measured in
  pairs/<A>__<B>.tsv    every tile x assay with signal, d = A - B     chrom start end region_id assay d bin_start
                                                                      a_asinh b_asinh
  pairs/<A>__<B>.bed    differential tiles (|d| >= threshold)         chrom start end region_id assay d direction
                                                                      bin_start (direction gain = higher in A)
  manifest.json         how everything was built, counts per split x cell x assay x direction (written last)

table.tsv holds every tile where some bin passes min_signal, so the threshold can be changed without a rebuild.
Needs the ENCODE tracks (step 1) and the region files (step 2). Re-running is cheap: if manifest.json matches
the spec, this script, the region files and the ENCODE manifests, nothing is rebuilt.

Usage (from the repo root):
    python data_prep/build_loci.py --dry-run   # report whether a rebuild is needed, write nothing
    python data_prep/build_loci.py             # build (skips if up to date)
    python data_prep/build_loci.py --force     # rebuild regardless
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyBigWig
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_regions import repo_path, md5_file, write_bed, _now


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="data_prep/loci_spec.yaml")
    p.add_argument("--dry-run", action="store_true", help="report whether a rebuild is needed; write nothing")
    p.add_argument("--force", action="store_true", help="rebuild even if the manifest says everything is up to date")
    return p.parse_args()


def load_yaml(path):
    with open(repo_path(path)) as f:
        return yaml.safe_load(f)


#--------------------------------------------------------------------------------------------------
# Inputs

def read_tiles(regions_root, name):
    path = os.path.join(regions_root, "windows", f"{name}.bed")
    if not os.path.exists(path):
        sys.exit(f"Window set {path} not found; build the region files first (data_prep/build_regions.py).")
    return pd.read_csv(path, sep="\t", comment="#", header=None, names=["chrom", "start", "end", "region_id", "class"])


def cell_tracks(encode_root, cells, assays):
    """{cell: {assay: bigWig path}} from each ENCODE folder's manifest.json (assays the line lacks are absent)."""
    tracks = {}
    for cell in cells:
        manifest_path = os.path.join(encode_root, cell, "manifest.json")
        if not os.path.exists(manifest_path):
            sys.exit(f"{manifest_path} not found; run the ENCODE step first (data_prep/setup_encode.py).")
        with open(manifest_path) as f:
            available = json.load(f)["assays"]
        tracks[cell] = {a: os.path.join(encode_root, cell, available[a]["file"]) for a in assays if a in available}
    return tracks


def bin_means(path, chrom, n_bins, bin_size):
    """(n_bins,) mean fold change per bin from the chromosome start (NaN = no data -> 0, as at runtime)."""
    bw = pyBigWig.open(path)
    try:
        vals = np.asarray(bw.values(chrom, 0, n_bins * bin_size, numpy=True), dtype=np.float64)
    finally:
        bw.close()
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).reshape(n_bins, bin_size).mean(axis=1)


#--------------------------------------------------------------------------------------------------
# Heuristic

def score_tiles(cell_fc, others_fc, min_signal):
    """Per tile, from (T, B) bin means of the cell and of the leave-one-out average (FC): the d of the counting
    bin with the largest |d|. Returns (has_signal (T,), d, bin index, cell asinh, others asinh) for that bin."""
    a, b = np.arcsinh(cell_fc), np.arcsinh(others_fc)
    d = a - b
    counts = np.maximum(a, b) >= min_signal
    pick = np.argmax(np.where(counts, np.abs(d), -1.0), axis=1)
    rows = np.arange(len(d))
    return counts.any(axis=1), d[rows, pick], pick, a[rows, pick], b[rows, pick]


def tile_bins(t, bin_size):
    """(T, B) indices of each tile's bins in the chromosome's bin-mean array."""
    n_sub = int((t["end"] - t["start"]).iloc[0]) // bin_size                         # bins per tile
    return (t["start"].to_numpy(np.int64) // bin_size)[:, None] + np.arange(n_sub)


def build_pair(a, b, tiles, tracks, fc_cache, spec):
    """pairs/<a>__<b>.tsv rows: d = asinh(a) - asinh(b) per tile and assay (same bins and signal rule)."""
    out = []
    for chrom, t in tiles.groupby("chrom", sort=False):
        idx = tile_bins(t, spec["bin_size"])
        for assay in spec["assays"]:
            if assay not in tracks[a] or assay not in tracks[b]:
                continue
            ok, d, pick, x, y = score_tiles(fc_cache[(a, assay, chrom)][idx], fc_cache[(b, assay, chrom)][idx],
                                            spec["min_signal"])
            sel = t[ok]
            out.append(pd.DataFrame({
                "chrom": chrom, "start": sel["start"].to_numpy(), "end": sel["end"].to_numpy(),
                "region_id": sel["region_id"].to_numpy(), "assay": assay, "d": d[ok].round(4),
                "bin_start": sel["start"].to_numpy() + pick[ok] * spec["bin_size"],
                "a_asinh": x[ok].round(4), "b_asinh": y[ok].round(4),
            }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def selected(table, threshold):
    """Rows with |d| >= threshold, with direction (gain / loss)."""
    sel = table[table["d"].abs() >= threshold].copy()
    sel["direction"] = np.where(sel["d"] > 0, "gain", "loss")
    return sel


def build_split(cells, tiles, tracks, fc_cache, spec):
    """table.tsv rows for one split: each cell line against the mean of the split's other lines with the assay."""
    bin_size = spec["bin_size"]
    out = []
    for chrom, t in tiles.groupby("chrom", sort=False):
        idx = tile_bins(t, bin_size)
        for assay in spec["assays"]:
            have = [c for c in cells if assay in tracks[c]]
            for cell in have:
                others = [c for c in have if c != cell]
                if not others:
                    continue
                cell_fc = fc_cache[(cell, assay, chrom)][idx]
                others_fc = np.mean([fc_cache[(o, assay, chrom)][idx] for o in others], axis=0)
                ok, d, pick, a, b = score_tiles(cell_fc, others_fc, spec["min_signal"])
                sel = t[ok]
                out.append(pd.DataFrame({
                    "chrom": chrom, "start": sel["start"].to_numpy(), "end": sel["end"].to_numpy(),
                    "region_id": sel["region_id"].to_numpy(), "cell": cell, "assay": assay,
                    "d": d[ok].round(4), "bin_start": sel["start"].to_numpy() + pick[ok] * bin_size,
                    "cell_asinh": a[ok].round(4), "others_asinh": b[ok].round(4), "n_others": len(others),
                }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


#--------------------------------------------------------------------------------------------------
# Manifest / rebuild check

def fingerprint(spec, regions_root, tracks):
    """Hash of everything the outputs depend on: the spec, this script, the region files and the ENCODE tracks."""
    regions_manifest = os.path.join(regions_root, "manifest.json")
    inputs = {
        "spec": spec,
        "script": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "regions": json.load(open(regions_manifest))["fingerprint"] if os.path.exists(regions_manifest) else None,
        "encode": {c: md5_file(os.path.join(os.path.dirname(next(iter(p.values()))), "manifest.json"))
                   for c, p in sorted(tracks.items()) if p},
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def pair_name(pair):
    return f"{pair[0]}__{pair[1]}"


def output_paths(spec):
    return [f"{split}/table.tsv" for split in spec["splits"]] + \
           [f"{split}/{cell}.bed" for split, cells in spec["splits"].items() for cell in cells] + \
           [f"pairs/{pair_name(p)}.{ext}" for p in spec.get("pairs", []) for ext in ("tsv", "bed")]


def rebuild_reason(spec, root, fp, force):
    manifest_path = os.path.join(root, "manifest.json")
    if force:
        return "--force"
    if not os.path.exists(manifest_path):
        return "no manifest.json yet"
    with open(manifest_path) as f:
        if json.load(f).get("fingerprint") != fp:
            return "spec, build_loci.py, the region files or the ENCODE tracks changed"
    missing = [p for p in output_paths(spec) if not os.path.exists(os.path.join(root, p))]
    return f"missing outputs: {', '.join(missing)}" if missing else None


def print_summary(manifest):
    print(f"\nCell-specific loci ({manifest['built']}, tiles {manifest['tiles']}, bin {manifest['bin_size']} bp, "
          f"min_signal {manifest['min_signal']}, |d| >= {manifest['threshold']}):")
    for split, info in manifest["splits"].items():
        print(f"  {split}: {info['table_rows']:,} table rows; cell-specific tiles per cell line (gain / loss):")
        assays = manifest["assays"]
        print("    " + f"{'':8s}" + "".join(f"{a:>12s}" for a in assays))
        for cell, per_assay in info["counts"].items():
            cells = "".join(f"{(str(per_assay[a]['gain']) + '/' + str(per_assay[a]['loss'])) if a in per_assay else '-':>12s}"
                            for a in assays)
            print(f"    {cell:8s}{cells}")
    for name, info in manifest.get("pairs", {}).items():
        print(f"  pair {name}: {info['table_rows']:,} table rows; differential tiles (higher in first / second):")
        print("    " + "".join(f"{a:>12s}" for a in manifest["assays"]))
        print("    " + "".join(f"{(str(info['counts'][a]['gain']) + '/' + str(info['counts'][a]['loss'])) if a in info['counts'] else '-':>12s}"
                               for a in manifest["assays"]))


#--------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    spec = load_yaml(args.spec)
    regions_root = load_yaml(spec["regions_spec"])["regions_root"]
    encode_root = load_yaml(spec["encode_spec"])["encode_root"]
    root = os.path.join(regions_root, "loci")
    manifest_path = os.path.join(root, "manifest.json")
    all_cells = list(dict.fromkeys([c for cells in spec["splits"].values() for c in cells]
                                   + [c for pair in spec.get("pairs", []) for c in pair]))
    print(f"Spec: {args.spec} | loci root: {root} | tiles: {spec['tiles']} | cell lines: {', '.join(all_cells)}",
          flush=True)

    tiles = read_tiles(regions_root, spec["tiles"])
    if any((tiles["end"] - tiles["start"]) % spec["bin_size"]):
        sys.exit(f"Tile length is not a multiple of bin_size {spec['bin_size']}.")
    tracks = cell_tracks(encode_root, all_cells, spec["assays"])
    fp = fingerprint(spec, regions_root, tracks)

    reason = rebuild_reason(spec, root, fp, args.force)
    if reason is None:
        print("Cell-specific loci are up to date (manifest matches spec, script and inputs); nothing to do.")
        print_summary(json.load(open(manifest_path)))
        return 0
    print(f"Rebuild needed: {reason}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    # Bin means of every track on the tiles' chromosomes (read once, shared by the splits)
    fc_cache = {}
    for chrom in tiles["chrom"].unique():
        n_bins = int(tiles.loc[tiles["chrom"] == chrom, "end"].max()) // spec["bin_size"]
        for cell in all_cells:
            for assay, path in tracks[cell].items():
                fc_cache[(cell, assay, chrom)] = bin_means(path, chrom, n_bins, spec["bin_size"])
            print(f"  read {chrom} {cell}: {len(tracks[cell])} assays", flush=True)

    built = _now()
    splits_info = {}
    for split, cells in spec["splits"].items():
        table = build_split(cells, tiles, tracks, fc_cache, spec)
        write_bed(os.path.join(root, split, "table.tsv"), table,
                  [f"Cell-specific loci table, split {split} (each of {', '.join(cells)} against the mean of the "
                   f"split's other lines). Built {built} by data_prep/build_loci.py from {args.spec}.",
                   f"Rows: tiles of {spec['tiles']} with a {spec['bin_size']} bp bin where max(cell, others) >= "
                   f"{spec['min_signal']} (asinh); d = cell - others at the counting bin with the largest |d|.",
                   "columns: " + " ".join(table.columns)])
        counts = {}
        for cell in cells:
            sel = selected(table[table["cell"] == cell], spec["threshold"])
            write_bed(os.path.join(root, split, f"{cell}.bed"),
                      sel[["chrom", "start", "end", "region_id", "assay", "d", "direction", "bin_start"]],
                      [f"{cell}-specific tiles, split {split}: |d| >= {spec['threshold']} against the mean of "
                       f"{', '.join(c for c in cells if c != cell)}. Built {built} by data_prep/build_loci.py.",
                       "One row per tile and assay (a tile can appear for several assays).",
                       "columns: chrom start end region_id assay d direction bin_start (0-based, half-open)"])
            counts[cell] = {a: {"gain": int((g["d"] > 0).sum()), "loss": int((g["d"] < 0).sum())}
                            for a, g in sel.groupby("assay")}
            for a in tracks[cell]:
                counts[cell].setdefault(a, {"gain": 0, "loss": 0})
        splits_info[split] = {"cells": cells, "table_rows": len(table), "counts": counts}

    pairs_info = {}
    for pair in spec.get("pairs", []):
        a, b = pair
        name = pair_name(pair)
        table = build_pair(a, b, tiles, tracks, fc_cache, spec)
        write_bed(os.path.join(root, "pairs", f"{name}.tsv"), table,
                  [f"Pairwise loci table {a} vs {b}. Built {built} by data_prep/build_loci.py from {args.spec}.",
                   f"Rows: tiles of {spec['tiles']} with a {spec['bin_size']} bp bin where max(a, b) >= "
                   f"{spec['min_signal']} (asinh); d = asinh(a) - asinh(b) at the counting bin with the largest |d|.",
                   "columns: " + " ".join(table.columns)])
        sel = selected(table, spec["threshold"])
        write_bed(os.path.join(root, "pairs", f"{name}.bed"),
                  sel[["chrom", "start", "end", "region_id", "assay", "d", "direction", "bin_start"]],
                  [f"Differential tiles {a} vs {b}: |d| >= {spec['threshold']}, d = asinh({a}) - asinh({b}); "
                   f"gain = higher in {a}. Built {built} by data_prep/build_loci.py.",
                   "One row per tile and assay (a tile can appear for several assays).",
                   "columns: chrom start end region_id assay d direction bin_start (0-based, half-open)"])
        pairs_info[name] = {"cells": [a, b], "table_rows": len(table),
                            "counts": {a_: {"gain": int((g["d"] > 0).sum()), "loss": int((g["d"] < 0).sum())}
                                       for a_, g in sel.groupby("assay")}}

    manifest = {
        "built": built, "builder": "data_prep/build_loci.py", "spec": args.spec, "fingerprint": fp,
        "tiles": spec["tiles"], "n_tiles": len(tiles), "assays": spec["assays"], "bin_size": spec["bin_size"],
        "min_signal": spec["min_signal"], "threshold": spec["threshold"],
        "encode_root": encode_root, "splits": splits_info, "pairs": pairs_info,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    print_summary(manifest)
    print(f"\nWrote cell-specific loci and manifest.json under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
