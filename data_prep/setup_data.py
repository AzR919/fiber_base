"""
Set up all data with one command. Runs each step in order; every step skips work that is already done,
so re-running is cheap and safe.

  1. setup_encode.py   ENCODE bulk tracks (from encode_spec.yaml) + configs/metapaths.yaml
  2. build_regions.py  region files for training pools and val/test window sets (from regions_spec.yaml)
  3. build_loci.py     cell-specific loci from the ENCODE tracks on the test tiles (from loci_spec.yaml)

Usage (from the repo root):
    python data_prep/setup_data.py --dry-run   # report what each step would do, write nothing
    python data_prep/setup_data.py             # run every step
    sbatch data_prep/slurm_setup_data.sh       # same, as a CPU job (needs internet: ENCODE + GitHub)

If a step fails, the later ones still run (step 3 needs steps 1 and 2), and the exit code is non-zero.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="report what each step would do; write nothing")
    p.add_argument("--workers", type=int, default=4, help="parallel ENCODE downloads")
    p.add_argument("--verify", action="store_true", help="re-hash existing ENCODE files")
    p.add_argument("--force-regions", action="store_true", help="rebuild the region files even if up to date")
    p.add_argument("--force-loci", action="store_true", help="rebuild the cell-specific loci even if up to date")
    return p.parse_args()


def main():
    args = parse_args()
    dry = ["--dry-run"] if args.dry_run else []
    steps = [
        ("ENCODE tracks + metapaths", ["setup_encode.py", "--spec", "data_prep/encode_spec.yaml",
                                       "--workers", str(args.workers)] + (["--verify"] if args.verify else []) + dry),
        ("Regions", ["build_regions.py", "--spec", "data_prep/regions_spec.yaml"]
                    + (["--force"] if args.force_regions else []) + dry),
        ("Cell-specific loci", ["build_loci.py", "--spec", "data_prep/loci_spec.yaml"]
                               + (["--force"] if args.force_loci else []) + dry),
    ]

    results = []
    for i, (title, cmd) in enumerate(steps, 1):
        print(f"\n{'=' * 80}\n Step {i}/{len(steps)}: {title}  ({' '.join(cmd)})\n{'=' * 80}", flush=True)
        t0 = time.time()
        # -u: unbuffered, so slurm logs show each step's progress as it happens
        code = subprocess.call([sys.executable, "-u", os.path.join(HERE, cmd[0])] + cmd[1:], cwd=REPO_ROOT)
        results.append((title, code, time.time() - t0))

    print(f"\n{'=' * 80}\n Data setup summary\n{'=' * 80}")
    for title, code, secs in results:
        print(f"  {'ok    ' if code == 0 else 'FAILED'}  {title}  ({secs / 60:.1f} min)")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
