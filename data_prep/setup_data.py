"""
Download ENCODE bulk tracks for the cell lines in a spec and regenerate configs/metapaths.yaml.

Steps:
  1. Pick one ENCODE file per (cell line, assay): reuse the choice in the lock file if present, otherwise
     select it with the rule in encode_api.py.
  2. Download missing/invalid files into <encode_root>/<cell>/ (resumable, md5-verified, parallel).
  3. Write <encode_root>/<cell>/manifest.json, the lock file, and the metapaths file.

Usage (from the repo root):
    python data_prep/setup_data.py --dry-run     # show what would be selected/downloaded, write nothing
    python data_prep/setup_data.py               # download + write outputs
    sbatch data_prep/slurm_setup_data.sh         # same, as a CPU job

Re-running is cheap and safe: files already present with the right md5 are skipped.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_outputs
import download
import encode_api

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACE_MARGIN = 10 * 1024**3  # keep at least this much free on the target filesystem


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default="data_prep/encode_spec.yaml")
    p.add_argument("--lock", default="data_prep/encode_lock.yaml")
    p.add_argument("--metapaths-out", default="configs/metapaths.yaml")
    p.add_argument("--dry-run", action="store_true", help="select and report only; download/write nothing")
    p.add_argument("--reselect", action="store_true", help="ignore the lock file and select every file again")
    p.add_argument("--verify", action="store_true", help="re-hash existing files even if their manifest md5 matches")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    return p.parse_args()


def repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def check_spec(spec):
    known = set(spec["fiber_samples"])
    for cell, cell_spec in spec["encode_cells"].items():
        for sample in cell_spec["fiber_samples"]:
            if sample not in known:
                sys.exit(f"Spec error: encode_cells.{cell} lists fiber sample '{sample}', which is not in fiber_samples.")
    for sample, cram in spec["fiber_samples"].items():
        path = os.path.join(spec["fiber_dir"], cram)
        if not os.path.exists(path):
            print(f"[WARN] fiber sample {sample}: CRAM not found at {path}")
        elif not os.path.exists(path + ".crai"):
            print(f"[WARN] fiber sample {sample}: no .crai index next to {path}")


def free_bytes(path):
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


def select_all(spec, lock, reselect):
    """{cell: {assay: selection}} plus a list of (cell, assay) with no ENCODE candidate."""
    plan, missing = {}, []
    for cell, cell_spec in spec["encode_cells"].items():
        plan[cell] = {}
        for assay, assay_spec in spec["assays"].items():
            locked = (lock.get(cell) or {}).get(assay)
            if locked and not reselect:
                sel = encode_api.fetch_locked(locked, spec["assembly"], spec["output_type"])
                sel["source"], sel["n_candidates"] = "lock", None
            else:
                sel, n = encode_api.select_track(cell_spec["term_name"], assay_spec, spec["assembly"], spec["output_type"])
                if sel is None:
                    missing.append((cell, assay))
                    print(f"  {cell:8s} {assay:9s} -> no ENCODE candidate", flush=True)
                    continue
                sel["source"], sel["n_candidates"] = "new", n
            sel["filename"] = build_outputs.track_filename(assay, sel["file"], spec["output_type"])
            plan[cell][assay] = sel
            print(f"  {cell:8s} {assay:9s} -> {sel['file']} ({sel['source']})", flush=True)
    return plan, missing


def print_plan(plan, missing, todo):
    print("\n" + "=" * 110)
    print(f"{'cell':8s} {'assay':9s} {'file':12s} {'experiment':12s} {'reps':8s} {'audit E/NC/W':13s} "
          f"{'size':>8s}  {'cands':>5s}  {'source':6s} {'action'}")
    print("-" * 110)
    for cell, tracks in plan.items():
        for assay, sel in tracks.items():
            a = sel["audit"]
            reps = ",".join(str(r) for r in sel["biological_replicates"]) or "-"
            cands = "-" if sel["n_candidates"] is None else str(sel["n_candidates"])
            action = "download" if (cell, assay) in todo else "present"
            print(f"{cell:8s} {assay:9s} {sel['file']:12s} {sel['experiment']:12s} {reps:8s} "
                  f"{a['ERROR']}/{a['NOT_COMPLIANT']}/{a['WARNING']:<9d} {sel['file_size'] / 1e9:7.2f}G  "
                  f"{cands:>5s}  {sel['source']:6s} {action}")
    for cell, assay in missing:
        print(f"{cell:8s} {assay:9s} {'-- none --':12s}")
    print("=" * 110)


def main():
    args = parse_args()
    spec_path = args.spec
    spec = load_yaml(repo_path(spec_path))
    lock_path = repo_path(args.lock)
    lock = {} if args.reselect or not os.path.exists(lock_path) else load_yaml(lock_path)
    encode_root = spec["encode_root"]

    print(f"Spec: {spec_path} | output type: {spec['output_type']} | {len(spec['encode_cells'])} cell lines "
          f"x {len(spec['assays'])} assays", flush=True)
    check_spec(spec)

    print("\nSelecting ENCODE files...", flush=True)
    plan, missing = select_all(spec, lock, args.reselect)

    # Which files still need downloading (present + md5 recorded in the previous manifest = skip without re-hashing)
    todo = {}
    for cell, tracks in plan.items():
        cell_dir = os.path.join(encode_root, cell)
        prev = build_outputs.read_manifest(cell_dir) or {}
        prev_md5 = {info["file"]: info["md5sum"] for info in prev.get("assays", {}).values()}
        for assay, sel in tracks.items():
            dest = os.path.join(cell_dir, sel["filename"])
            if not download.is_complete(dest, sel["md5sum"], sel["file_size"], prev_md5.get(sel["filename"]), args.verify):
                todo[(cell, assay)] = dest

    print_plan(plan, missing, todo)
    need = sum(plan[c][a]["file_size"] for c, a in todo)
    free = free_bytes(encode_root)
    print(f"To download: {len(todo)} files, {need / 1e9:.1f} GB | free on target filesystem: {free / 1e9:.1f} GB")

    if args.dry_run:
        print("\nDry run: nothing downloaded or written.")
        return 0
    if need > free - SPACE_MARGIN:
        sys.exit(f"Not enough space: need {need / 1e9:.1f} GB + {SPACE_MARGIN / 1e9:.0f} GB margin, "
                 f"only {free / 1e9:.1f} GB free. Free up space or drop assays/cells from the spec.")

    failures = {}
    if todo:
        print(f"\nDownloading with {args.workers} workers...", flush=True)
        for cell in plan:
            os.makedirs(os.path.join(encode_root, cell), exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download.fetch, plan[c][a]["url"], dest, plan[c][a]["md5sum"],
                                   plan[c][a]["file_size"]): (c, a)
                       for (c, a), dest in todo.items()}
            for i, fut in enumerate(as_completed(futures), 1):
                cell, assay = futures[fut]
                try:
                    fut.result()
                    print(f"  [{i}/{len(futures)}] ok    {cell} {assay} ({plan[cell][assay]['file_size'] / 1e9:.2f} GB)", flush=True)
                except Exception as e:
                    failures[(cell, assay)] = str(e)
                    print(f"  [{i}/{len(futures)}] FAIL  {cell} {assay}: {e}", flush=True)

    # Outputs reflect what is actually on disk
    cell_assays = {}
    for cell, tracks in plan.items():
        present = {a: s for a, s in tracks.items() if (cell, a) not in failures}
        cell_assays[cell] = list(present)
        if present:
            cell_dir = os.path.join(encode_root, cell)
            os.makedirs(cell_dir, exist_ok=True)
            build_outputs.write_manifest(cell_dir, cell, spec["encode_cells"][cell], spec, present)
    build_outputs.write_lock(lock_path, plan)
    build_outputs.write_metapaths(repo_path(args.metapaths_out), spec, spec_path, cell_assays)

    print(f"\nWrote manifests under {encode_root}, {args.lock}, {args.metapaths_out}")
    print("\nAssay coverage:")
    for cell in spec["encode_cells"]:
        have = cell_assays.get(cell, [])
        lacking = [a for a in spec["assays"] if a not in have]
        print(f"  {cell:8s} {len(have)}/{len(spec['assays'])}" + (f"  missing: {', '.join(lacking)}" if lacking else ""))

    if failures:
        print(f"\n{len(failures)} download(s) failed (re-run to retry; completed files are kept):")
        for (cell, assay), err in failures.items():
            print(f"  {cell} {assay}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
