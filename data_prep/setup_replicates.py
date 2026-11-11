"""
Replicate-level ENCODE tracks for the noise ceiling (reports/TODO_extended.md I11).

For every (cell line, assay) in the main lock file (data_prep/encode_lock.yaml), this finds the per-replicate
bigWigs of the same ENCODE experiment and output type as the pooled training target ("fold change over control",
GRCh38): released files from exactly one biological replicate, the newest file per replicate, and the two
lowest-numbered replicates. Experiments with fewer than two are recorded as such (no ceiling for that track);
nothing is substituted from another experiment.

Steps:
  1. Select (or reuse from data_prep/encode_replicates_lock.yaml) two replicate files per track.
  2. Download each one, verify size + md5, keep only the chromosomes in --chroms (default chr21, the test
     chromosome) and delete the full file, into <encode_root>/<cell>/replicates/.
  3. Write the replicate lock and <encode_root>/<cell>/replicates/manifest.json.

Usage (from the repo root):
    python data_prep/setup_replicates.py --dry-run   # select and report sizes, download nothing
    python data_prep/setup_replicates.py             # download + crop + write outputs
Re-running is safe: cropped files already listed in the manifest are skipped; partial downloads resume.
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download
import encode_api

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACE_MARGIN = 10 * 1024**3
FIELDS = encode_api.FILE_FIELDS + ["technical_replicates"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="data_prep/encode_spec.yaml")
    p.add_argument("--lock", default="data_prep/encode_lock.yaml", help="main lock: the pooled training tracks")
    p.add_argument("--rep-lock", default="data_prep/encode_replicates_lock.yaml")
    p.add_argument("--chroms", nargs="+", default=["chr21"], help="chromosomes kept in the cropped files")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reselect", action="store_true", help="ignore the replicate lock and select again")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def cropped_name(assay, rep, accession, chroms):
    return f"{assay}_rep{rep}_{accession}_fc_{'_'.join(chroms)}.bigWig"


def select_replicates(experiment, pooled_file, spec):
    """(entry, note). entry has the pooled file's replicates and up to two per-replicate files."""
    files = encode_api.search({
        "type": "File", "dataset": f"/experiments/{experiment}/", "status": "released", "file_format": "bigWig",
        "assembly": spec["assembly"], "output_type": spec["output_type"], "field": FIELDS,
    })
    pooled = next((f for f in files if f["accession"] == pooled_file), None)
    per_rep = {}
    for f in files:
        reps = f.get("biological_replicates") or []
        if len(reps) != 1:
            continue
        r = int(reps[0])
        if r not in per_rep or (f.get("date_created", ""), f["accession"]) > (per_rep[r].get("date_created", ""),
                                                                             per_rep[r]["accession"]):
            per_rep[r] = f
    chosen = sorted(per_rep)[:2]
    entry = {
        "experiment": experiment,
        "pooled_file": pooled_file,
        "pooled_replicates": sorted(int(r) for r in (pooled or {}).get("biological_replicates") or []),
        "available_replicates": sorted(per_rep),
        "replicates": [{"rep": r, "file": per_rep[r]["accession"], "md5sum": per_rep[r]["md5sum"],
                        "file_size": per_rep[r]["file_size"]} for r in chosen],
    }
    note = "ok" if len(chosen) == 2 else ("single replicate" if chosen else "no per-replicate file")
    entry["status"] = note
    return entry, note


def crop(src_path, dst_path, chroms):
    """Write a bigWig with only `chroms` (same intervals and values) via dst.part, then rename."""
    import numpy as np
    import pyBigWig
    src = pyBigWig.open(src_path)
    have = src.chroms()
    keep = [c for c in chroms if c in have]
    if not keep:
        raise ValueError(f"{src_path}: none of {chroms} present")
    part = dst_path + ".part"
    out = pyBigWig.open(part, "w")
    out.addHeader([(c, have[c]) for c in keep], maxZooms=10)
    for c in keep:
        iv = src.intervals(c) or []
        for i in range(0, len(iv), 1_000_000):
            block = np.asarray(iv[i:i + 1_000_000], dtype=np.float64)
            out.addEntries([c] * len(block), block[:, 0].astype(np.int64).tolist(),
                           ends=block[:, 1].astype(np.int64).tolist(), values=block[:, 2].tolist())
    out.close()
    src.close()
    os.replace(part, dst_path)


def fetch_and_crop(url, full_path, dst_path, md5sum, size, chroms):
    download.fetch(url, full_path, md5sum, size)
    crop(full_path, dst_path, chroms)
    os.remove(full_path)
    return dst_path


def main():
    args = parse_args()
    spec = load_yaml(repo_path(args.spec))
    lock = load_yaml(repo_path(args.lock))
    rep_lock_path = repo_path(args.rep_lock)
    rep_lock = {} if args.reselect else load_yaml(rep_lock_path)
    encode_root = spec["encode_root"]

    print(f"Selecting replicate files for {sum(len(v) for v in lock.values())} locked tracks...", flush=True)
    plan = {}
    for cell, tracks in lock.items():
        plan[cell] = {}
        for assay, sel in tracks.items():
            prev = (rep_lock.get(cell) or {}).get(assay)
            if prev and prev.get("pooled_file") == sel["file"]:
                entry, src = prev, "lock"
            else:
                entry, _ = select_replicates(sel["experiment"], sel["file"], spec)
                src = "new"
            plan[cell][assay] = entry
            reps = " ".join(f"rep{r['rep']}={r['file']}({r['file_size'] / 1e9:.2f}G)" for r in entry["replicates"])
            print(f"  {cell:7s} {assay:9s} {entry['experiment']} pooled {entry['pooled_file']} "
                  f"reps{entry['pooled_replicates']} | available {entry['available_replicates']} | "
                  f"{entry['status']:20s} {reps} [{src}]", flush=True)

    todo = []
    for cell, tracks in plan.items():
        rep_dir = os.path.join(encode_root, cell, "replicates")
        for assay, entry in tracks.items():
            for r in entry["replicates"]:
                dst = os.path.join(rep_dir, cropped_name(assay, r["rep"], r["file"], args.chroms))
                r["filename"] = os.path.basename(dst)
                if not os.path.exists(dst):
                    todo.append((cell, assay, r, dst))

    n_ok = sum(e["status"] == "ok" for t in plan.values() for e in t.values())
    n_all = sum(len(t) for t in plan.values())
    total = sum(r["file_size"] for _, _, r, _ in todo)
    biggest = sorted((r["file_size"] for _, _, r, _ in todo), reverse=True)[:args.workers]
    free = shutil.disk_usage(encode_root if os.path.exists(encode_root) else REPO_ROOT).free
    print(f"\n{n_ok} / {n_all} tracks have two replicates. To fetch: {len(todo)} files, {total / 1e9:.1f} GB in "
          f"total; peak extra disk ~{sum(biggest) / 1e9:.1f} GB ({args.workers} full files at once, each cropped to "
          f"{', '.join(args.chroms)} and deleted). Free on the target filesystem: {free / 1e9:.0f} GB.", flush=True)
    if args.dry_run:
        return
    if sum(biggest) + SPACE_MARGIN > free:
        sys.exit("Not enough free space for the parallel downloads; lower --workers.")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for cell, assay, r, dst in todo:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            full = os.path.join(os.path.dirname(dst), f"{r['file']}.full.bigWig")
            url = f"{encode_api.BASE_URL}/files/{r['file']}/@@download/{r['file']}.bigWig"
            futs[pool.submit(fetch_and_crop, url, full, dst, r["md5sum"], r["file_size"], args.chroms)] = (cell, assay, r)
        for fut in as_completed(futs):
            cell, assay, r = futs[fut]
            try:
                fut.result()
                print(f"  done {cell} {assay} rep{r['rep']} {r['file']}", flush=True)
            except Exception as e:  # keep going; report at the end
                failed.append((cell, assay, r["file"], str(e)))
                print(f"  FAILED {cell} {assay} rep{r['rep']} {r['file']}: {e}", flush=True)

    with open(rep_lock_path, "w") as f:
        f.write("# GENERATED by data_prep/setup_replicates.py: the per-replicate ENCODE files (same experiment and\n"
                "# output type as the pooled training track in encode_lock.yaml) used for the noise ceiling.\n")
        yaml.safe_dump(plan, f, sort_keys=False)
    for cell, tracks in plan.items():
        rep_dir = os.path.join(encode_root, cell, "replicates")
        if os.path.isdir(rep_dir):
            with open(os.path.join(rep_dir, "manifest.json"), "w") as f:
                json.dump({"chroms": args.chroms, "output_type": spec["output_type"], "assembly": spec["assembly"],
                           "assays": tracks}, f, indent=1)
    print(f"\nWrote {args.rep_lock} and replicates/manifest.json per cell line.", flush=True)
    if failed:
        sys.exit(f"{len(failed)} downloads failed: {failed}")


if __name__ == "__main__":
    main()
