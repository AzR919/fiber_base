"""
Data setup step 2: build the region files that say where we train, validate and test.
Normally run through data_prep/setup_data.py, which runs every step.

Reads data_prep/regions_spec.yaml and writes into <regions_root>/:
  sources/hg38-blacklist.v2.bed.gz   downloaded once (md5 recorded in the manifest)
  exclude.bed                        blacklist + N runs                       chrom start end reason
  allowed_centres.bed                where a region centre may lie            chrom start end
  strata/ccre_classes.bed            every cCRE with its class (bp scoring)   chrom start end id class
  pools/ccre_centres.bed             allowed cCRE centres (training pool)     chrom centre centre+1 id class
  windows/<set>.bed                  fixed val/test region sets               chrom start end region_id class
  manifest.json                      how everything was built (written last)

Region rule: a region is R = region_length bp, [c - R/2, c + R/2) for centre c. c is allowed when
  - the region does not overlap the blacklist, and
  - an input window of max_window bp centred on c lies inside the chromosome and contains no N.
Val/test regions are tiles of a fixed grid (start = k * R from each chromosome start) whose centre is allowed,
so they never overlap and a sampled set is always a subset of the full tiling.

Window class (for stratified sampling only): the first of PLS > ELS > CTCF > other with a cCRE overlapping the
region, else background. Per-bp scoring by class uses strata/ccre_classes.bed at runtime.

All coordinates are 0-based, half-open (BED). Outputs are deterministic: same spec + inputs -> same files.
Re-running is cheap: if manifest.json matches the spec, this script and the input files, nothing is rebuilt.

Usage (from the repo root):
    python data_prep/build_regions.py --dry-run   # report whether a rebuild is needed, write nothing
    python data_prep/build_regions.py             # build (skips if up to date)
    python data_prep/build_regions.py --force     # rebuild regardless
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pysam
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# cCRE registry V3 label (column 11) -> class we score by
CCRE_CLASS = {"PLS": "PLS", "pELS": "ELS", "dELS": "ELS", "CTCF-only": "CTCF", "DNase-H3K4me3": "other"}
CLASS_PRIORITY = ["PLS", "ELS", "CTCF", "other"]   # window class = first class with a cCRE in the region
BACKGROUND = "background"
ALL_CLASSES = CLASS_PRIORITY + [BACKGROUND]
METHODS = {"tiles", "stratified", "ccre_centred"}
BLACKLIST_FILE = "sources/hg38-blacklist.v2.bed.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="data_prep/regions_spec.yaml")
    p.add_argument("--dry-run", action="store_true", help="report whether a rebuild is needed; write nothing")
    p.add_argument("--force", action="store_true", help="rebuild even if the manifest says everything is up to date")
    return p.parse_args()


def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def md5_file(path):
    return download.md5_of(path).hexdigest()


#--------------------------------------------------------------------------------------------------
# Interval helpers. An interval set is (starts, ends) int64 arrays for one chromosome, half-open.

def merge(starts, ends):
    """Sorted, non-overlapping union (touching intervals are joined)."""
    if len(starts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(starts, kind="stable")
    out_s, out_e = [int(starts[order[0]])], [int(ends[order[0]])]
    for a, b in zip(starts[order[1:]], ends[order[1:]]):
        if a <= out_e[-1]:
            out_e[-1] = max(out_e[-1], int(b))
        else:
            out_s.append(int(a))
            out_e.append(int(b))
    return np.array(out_s, np.int64), np.array(out_e, np.int64)


def subtract(a, b):
    """a minus b; both merged."""
    out_s, out_e = [], []
    j = 0
    for s, e in zip(*a):
        cur = int(s)
        while j < len(b[0]) and b[1][j] <= cur:
            j += 1
        k = j
        while k < len(b[0]) and b[0][k] < e:
            if b[0][k] > cur:
                out_s.append(cur)
                out_e.append(int(b[0][k]))
            cur = max(cur, int(b[1][k]))
            k += 1
        if cur < e:
            out_s.append(cur)
            out_e.append(int(e))
    return np.array(out_s, np.int64), np.array(out_e, np.int64)


def contains(intervals, points):
    """Boolean mask: which points lie inside the (merged) interval set."""
    s, e = intervals
    if len(s) == 0:
        return np.zeros(len(points), bool)
    idx = np.searchsorted(s, points, side="right") - 1
    return (idx >= 0) & (points < e[np.clip(idx, 0, None)])


def overlaps_any(starts, ends, qs, qe):
    """Boolean mask: which query intervals [qs, qe) overlap at least one of the given intervals."""
    if len(starts) == 0:
        return np.zeros(len(qs), bool)
    n_started = np.searchsorted(np.sort(starts), qe, side="left")   # intervals with start < qe
    n_ended = np.searchsorted(np.sort(ends), qs, side="right")      # intervals with end <= qs
    return n_started - n_ended > 0


#--------------------------------------------------------------------------------------------------
# Inputs

def chrom_sizes(fasta_path):
    with open(fasta_path + ".fai") as f:
        return {line.split("\t")[0]: int(line.split("\t")[1]) for line in f}


def n_runs(fasta_path, chroms):
    """{chrom: (starts, ends)} of runs of N/n in the reference."""
    runs = {}
    with pysam.FastaFile(fasta_path) as fa:
        for chrom in chroms:
            seq = np.frombuffer(fa.fetch(chrom).encode("ascii"), dtype=np.uint8)
            is_n = ((seq == ord("N")) | (seq == ord("n"))).astype(np.int8)
            edges = np.diff(is_n, prepend=0, append=0)
            runs[chrom] = (np.flatnonzero(edges == 1).astype(np.int64), np.flatnonzero(edges == -1).astype(np.int64))
            print(f"  {chrom:6s} {len(runs[chrom][0]):4d} N runs, {int((runs[chrom][1] - runs[chrom][0]).sum()):>11,} bp", flush=True)
    return runs


def read_blacklist(path, chroms):
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "reason"])
    df["reason"] = "blacklist:" + df["reason"].str.replace(" ", "_")
    return df[df["chrom"].isin(chroms)].reset_index(drop=True)


def read_ccres(path, chroms):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3, 10], names=["chrom", "start", "end", "id", "label"])
    unknown = sorted(set(df["label"]) - set(CCRE_CLASS))
    if unknown:
        sys.exit(f"Unknown cCRE labels in {path}: {unknown}. Add them to CCRE_CLASS in build_regions.py.")
    df["class"] = df["label"].map(CCRE_CLASS)
    df = df[df["chrom"].isin(chroms)].drop(columns="label")
    return df.sort_values(["chrom", "start"], kind="stable").reset_index(drop=True)


#--------------------------------------------------------------------------------------------------
# Region rule and tiles

def allowed_centres(size, gaps, blacklist, R, L_max):
    """Centres c whose region [c - R/2, c + R/2) avoids the blacklist and whose L_max window
    [c - L_max/2, c + L_max/2) is inside the chromosome and N-free."""
    H, r = L_max // 2, R // 2
    base = (np.array([H], np.int64), np.array([size - H + 1], np.int64))
    forbid_s = np.concatenate([gaps[0] - H + 1, blacklist[0] + r - R + 1])
    forbid_e = np.concatenate([gaps[1] + H, blacklist[1] + r])
    return subtract(base, merge(forbid_s, forbid_e))


def window_class(ccres_chrom, qs, qe):
    """Class of each region [qs, qe): first class in CLASS_PRIORITY with an overlapping cCRE, else background."""
    cls = np.full(len(qs), BACKGROUND, dtype=object)
    unset = np.ones(len(qs), bool)
    for c in CLASS_PRIORITY:
        sub = ccres_chrom[ccres_chrom["class"] == c]
        hit = overlaps_any(sub["start"].values, sub["end"].values, qs, qe) & unset
        cls[hit] = c
        unset &= ~hit
    return cls


def build_tiles(chrom, size, allowed, ccres_chrom, R):
    starts = np.arange(0, size - R + 1, R, dtype=np.int64)
    starts = starts[contains(allowed, starts + R // 2)]
    ends = starts + R
    return pd.DataFrame({"chrom": chrom, "start": starts, "end": ends,
                         "region_id": [f"{chrom}:{s}-{e}" for s, e in zip(starts, ends)],
                         "class": window_class(ccres_chrom, starts, ends)})


def stratified_sample(tiles, n, seed):
    """n tiles, allocated to classes in proportion to their share (largest remainder), sampled with a seeded RNG."""
    counts = tiles["class"].value_counts()
    classes = [c for c in ALL_CLASSES if counts.get(c, 0) > 0]
    total = int(counts.sum())
    if n > total:
        sys.exit(f"Stratified set asks for {n} regions but only {total} tiles are available.")
    exact = np.array([n * counts[c] / total for c in classes])
    quota = np.floor(exact).astype(int)
    quota[np.argsort(-(exact - quota), kind="stable")[: n - quota.sum()]] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for c, q in zip(classes, quota):
        group = tiles[tiles["class"] == c]
        picked.append(group.iloc[np.sort(rng.choice(len(group), size=q, replace=False))])
    return pd.concat(picked)


def ccre_centred(chrom, size, ccres_chrom, R, L_max):
    """Legacy protocol: R bp centred on every cCRE (same centring as the old dataset code), overlaps kept.
    Only requirement: the L_max window fits in the chromosome."""
    c = ((ccres_chrom["start"].values + ccres_chrom["end"].values) // 2).astype(np.int64)
    keep = (c - L_max // 2 >= 0) & (c + L_max // 2 <= size)
    starts = c[keep] - R // 2
    return pd.DataFrame({"chrom": chrom, "start": starts, "end": starts + R,
                         "region_id": ccres_chrom["id"].values[keep], "class": ccres_chrom["class"].values[keep]})


#--------------------------------------------------------------------------------------------------
# Output

def write_bed(path, df, header_lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, sep="\t", header=False, index=False)
    os.replace(tmp, path)


def sort_bed(df, chroms):
    order = {c: i for i, c in enumerate(chroms)}
    return df.assign(_o=df["chrom"].map(order)).sort_values(["_o", "start"], kind="stable").drop(columns="_o")


def class_counts(df):
    counts = df["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in ALL_CLASSES if counts.get(c, 0) > 0}


#--------------------------------------------------------------------------------------------------

def check_spec(spec):
    R, L_max = spec["region_length"], spec["max_window"]
    if R % 2 or L_max % 2 or L_max < R:
        sys.exit(f"region_length ({R}) and max_window ({L_max}) must be even, with max_window >= region_length.")
    for name, ws in spec["window_sets"].items():
        if ws.get("method") not in METHODS:
            sys.exit(f"window_sets.{name}: method must be one of {sorted(METHODS)}.")
        if ws["method"] == "stratified" and not ("n" in ws and "seed" in ws):
            sys.exit(f"window_sets.{name}: stratified sets need n and seed.")
        bad = [c for c in ws["chroms"] if c not in spec["chroms"]]
        if bad:
            sys.exit(f"window_sets.{name}: chromosomes {bad} are not in the spec's chroms list.")


def output_paths(spec):
    return (["exclude.bed", "allowed_centres.bed", "strata/ccre_classes.bed", "pools/ccre_centres.bed"]
            + [f"windows/{name}.bed" for name in spec["window_sets"]])


def fingerprint(spec, root):
    """Hash of everything the outputs depend on: the spec, this script, and the input files."""
    bl = os.path.join(root, BLACKLIST_FILE)
    fasta = spec["fasta_path"]
    inputs = {
        "spec": spec,
        "script": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "fasta": [os.path.getsize(fasta), md5_file(fasta + ".fai")],
        "ccre": md5_file(spec["ccre_path"]),
        "blacklist": md5_file(bl) if os.path.exists(bl) else None,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(), inputs


def rebuild_reason(spec, root, force):
    manifest_path = os.path.join(root, "manifest.json")
    if force:
        return "--force"
    if not os.path.exists(manifest_path):
        return "no manifest.json yet"
    if not os.path.exists(os.path.join(root, BLACKLIST_FILE)):
        return "blacklist not downloaded"
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("fingerprint") != fingerprint(spec, root)[0]:
        return "spec, build_regions.py or an input file changed"
    missing = [p for p in output_paths(spec) if not os.path.exists(os.path.join(root, p))]
    if missing:
        return f"missing outputs: {', '.join(missing)}"
    return None


def print_summary(manifest):
    files = manifest["files"]
    print(f"\nRegions ({manifest['built']}, R = {manifest['region_length']}, L_max = {manifest['max_window']}):")
    print(f"  exclude.bed              {files['exclude.bed']['rows']:>8,} intervals ({files['exclude.bed']['bp']:,} bp)")
    print(f"  allowed_centres.bed      {files['allowed_centres.bed']['rows']:>8,} intervals ({files['allowed_centres.bed']['bp']:,} bp of centres)")
    print(f"  strata/ccre_classes.bed  {files['strata/ccre_classes.bed']['rows']:>8,} cCREs")
    print(f"  pools/ccre_centres.bed   {files['pools/ccre_centres.bed']['rows']:>8,} cCRE centres "
          f"({files['pools/ccre_centres.bed']['dropped']:,} dropped by the region rule)")
    for path, info in files.items():
        if path.startswith("windows/"):
            counts = "  ".join(f"{c} {k:,}" for c, k in info["class_counts"].items())
            print(f"  {path:34s} {info['rows']:>7,}  [{info['method']}, {','.join(info['chroms'])}]  {counts}")


def main():
    args = parse_args()
    spec = yaml.safe_load(open(repo_path(args.spec)))
    check_spec(spec)
    root = spec["regions_root"]
    R, L_max, chroms = spec["region_length"], spec["max_window"], spec["chroms"]
    manifest_path = os.path.join(root, "manifest.json")
    print(f"Spec: {args.spec} | regions_root: {root} | R = {R}, L_max = {L_max} | {len(chroms)} chromosomes", flush=True)

    reason = rebuild_reason(spec, root, args.force)
    if reason is None:
        print("Regions are up to date (manifest matches spec, script and inputs); nothing to do.")
        print_summary(json.load(open(manifest_path)))
        return 0
    print(f"Rebuild needed: {reason}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    # Inputs
    os.makedirs(root, exist_ok=True)
    bl_path = os.path.join(root, BLACKLIST_FILE)
    if not os.path.exists(bl_path):
        os.makedirs(os.path.dirname(bl_path), exist_ok=True)
        print(f"Downloading blacklist: {spec['blacklist_url']}", flush=True)
        download.fetch_url(spec["blacklist_url"], bl_path)
    if not os.path.exists(spec["fasta_path"] + ".fai"):
        pysam.faidx(spec["fasta_path"])
    fp, inputs = fingerprint(spec, root)

    sizes = chrom_sizes(spec["fasta_path"])
    missing = [c for c in chroms if c not in sizes]
    if missing:
        sys.exit(f"Chromosomes {missing} are not in {spec['fasta_path']}.fai")
    blacklist = read_blacklist(bl_path, chroms)
    ccres = read_ccres(spec["ccre_path"], chroms)
    print("Scanning the reference for N runs...", flush=True)
    gaps = n_runs(spec["fasta_path"], chroms)

    # Exclusions and allowed centres
    exclude = pd.concat([blacklist] + [pd.DataFrame({"chrom": c, "start": s, "end": e, "reason": "N_gap"})
                                       for c, (s, e) in gaps.items()])
    exclude = sort_bed(exclude, chroms)
    allowed, allowed_rows = {}, []
    for c in chroms:
        bl_c = blacklist[blacklist["chrom"] == c]
        allowed[c] = allowed_centres(sizes[c], gaps[c], merge(bl_c["start"].values, bl_c["end"].values), R, L_max)
        allowed_rows.append(pd.DataFrame({"chrom": c, "start": allowed[c][0], "end": allowed[c][1]}))
    allowed_df = pd.concat(allowed_rows)

    # Training pool: allowed cCRE centres
    centres = ((ccres["start"] + ccres["end"]) // 2).astype(np.int64)
    in_allowed = np.zeros(len(ccres), bool)
    for c in chroms:
        m = (ccres["chrom"] == c).values
        in_allowed[m] = contains(allowed[c], centres.values[m])
    pool = pd.DataFrame({"chrom": ccres["chrom"], "start": centres, "end": centres + 1,
                         "id": ccres["id"], "class": ccres["class"]})[in_allowed]

    # Window sets
    ccres_by_chrom = {c: g for c, g in ccres.groupby("chrom")}
    empty_ccres = ccres.iloc[:0]
    tiles_cache = {}

    def tiles_for(cs):
        for c in cs:
            if c not in tiles_cache:
                tiles_cache[c] = build_tiles(c, sizes[c], allowed[c], ccres_by_chrom.get(c, empty_ccres), R)
        return pd.concat([tiles_cache[c] for c in cs])

    sets = {}
    for name, ws in spec["window_sets"].items():
        if ws["method"] == "tiles":
            df = tiles_for(ws["chroms"])
        elif ws["method"] == "stratified":
            df = stratified_sample(tiles_for(ws["chroms"]), ws["n"], ws["seed"])
        else:
            df = pd.concat([ccre_centred(c, sizes[c], ccres_by_chrom.get(c, empty_ccres), R, L_max) for c in ws["chroms"]])
        sets[name] = sort_bed(df, chroms)

    # Write (manifest last: its presence means a complete build)
    built = _now()
    cols5 = "columns: chrom start end region_id class (0-based, half-open)"
    write_bed(os.path.join(root, "exclude.bed"), exclude,
              [f"Excluded intervals: ENCODE blacklist v2 + N runs. Built {built} by data_prep/build_regions.py.",
               "columns: chrom start end reason"])
    write_bed(os.path.join(root, "allowed_centres.bed"), allowed_df,
              [f"Allowed region centres (R = {R}, L_max = {L_max}): region avoids the blacklist, L_max window is "
               f"inside the chromosome and N-free. Built {built}.", "columns: chrom start end"])
    write_bed(os.path.join(root, "strata/ccre_classes.bed"), ccres,
              [f"cCRE registry V3 with scoring class (PLS / ELS / CTCF / other). Built {built}.",
               "columns: chrom start end id class"])
    write_bed(os.path.join(root, "pools/ccre_centres.bed"), pool,
              [f"Training pool: cCRE centres that satisfy the region rule (R = {R}, L_max = {L_max}). Built {built}.",
               "columns: chrom centre centre+1 id class"])
    for name, df in sets.items():
        ws = spec["window_sets"][name]
        params = ", ".join(f"{k}: {v}" for k, v in ws.items())
        write_bed(os.path.join(root, f"windows/{name}.bed"), df,
                  [f"Window set {name} ({params}). Built {built} by data_prep/build_regions.py.",
                   f"Each row is a scored region of {R} bp; the runtime centres the input window on it.", cols5])

    stale = sorted(set(os.listdir(os.path.join(root, "windows"))) - {f"{n}.bed" for n in sets})
    for f in stale:
        print(f"[WARN] windows/{f} is not in the spec any more (left in place; delete it if no config uses it)")

    files = {
        "exclude.bed": {"rows": len(exclude), "bp": int((exclude["end"] - exclude["start"]).sum())},
        "allowed_centres.bed": {"rows": len(allowed_df), "bp": int((allowed_df["end"] - allowed_df["start"]).sum())},
        "strata/ccre_classes.bed": {"rows": len(ccres), "class_counts": class_counts(ccres)},
        "pools/ccre_centres.bed": {"rows": len(pool), "dropped": int((~in_allowed).sum()), "class_counts": class_counts(pool)},
    }
    for name, df in sets.items():
        ws = spec["window_sets"][name]
        info = {"rows": len(df), **ws, "class_counts": class_counts(df)}
        if ws["method"] == "stratified":
            info["tile_class_counts"] = class_counts(tiles_for(ws["chroms"]))
        files[f"windows/{name}.bed"] = info
    manifest = {
        "built": built,
        "builder": "data_prep/build_regions.py",
        "spec": args.spec,
        "fingerprint": fp,
        "region_length": R,
        "max_window": L_max,
        "chroms": chroms,
        "sources": {
            "fasta": {"path": spec["fasta_path"], "size": inputs["fasta"][0], "fai_md5": inputs["fasta"][1]},
            "ccre": {"path": spec["ccre_path"], "md5": inputs["ccre"]},
            "blacklist": {"url": spec["blacklist_url"], "path": bl_path, "md5": inputs["blacklist"]},
        },
        "files": files,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    print_summary(manifest)
    print(f"\nWrote region files and manifest.json under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
