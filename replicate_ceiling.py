"""
Replicate noise ceiling for the bulk metrics (reports/TODO_extended.md I11).

Each ENCODE target track is pooled from 2-3 biological replicates, and noisy. This scores replicate A against
replicate B of the same experiment (data_prep/setup_replicates.py), on exactly the positions eval.py scores: the
same eval configs and chr21 tiles, positions covered by >= min_coverage fibers, asinh fold change, and the same
cross-cell-average baseline (the training cells of --train_data_config) for the cell-type r.

Per cell type and assay it reports:
  rep r        Pearson r of replicate A vs B (= the reliability of one replicate)
  rep ct r     partial r of A vs B given the baseline (reliability of the cell-specific part)
  rep MSE      MSE of A vs B
  ceiling r    sqrt(Spearman-Brown reliability of the pooled track, k = replicates in the pooled file):
               the r a prediction equal to the true signal would reach against the pooled target. Same for ct r.
  MSE floor    rep MSE / (2k): the pooled target's noise variance, i.e. the MSE of a perfect prediction. Assumes
               independent, additive replicate noise in asinh space (approximate).
  A/B vs pooled  r of each replicate against the pooled target (inflated: the pooled track contains it).
A model scored against the pooled target can be read as a fraction of the ceiling.

Usage (CPU is enough; fibers are read only to find the scored positions):
    python replicate_ceiling.py --train_data_config configs/data/data24_dev_loo_Panc1_200U.yaml \
        --eval_configs configs/evals/eval21_GM12878.yaml [more...] --num_workers 4
Writes results/<timestamp>_replicates_<train data config>/replicate_summary.json.
"""

import argparse
import datetime
import json
import os

import numpy as np
import pyBigWig
import torch
import yaml
from torch.utils.data import DataLoader

from data_utils import make_fiber_dataset, loader_context
from evaluator import resolve_baseline_cells
from metrics import PearsonMeter, score_weights, partial_correlation, MIN_COVERAGE
from utils import load_config_file, load_metapaths, resolve_config_with_metapaths, seed_worker

ENCODE_SPEC = "data_prep/encode_spec.yaml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_configs", nargs="+", required=True)
    p.add_argument("--train_data_config", required=True, help="its cell_types define the cross-cell baseline")
    p.add_argument("--output_assays", nargs="+", default=None, help=f"default: every assay in {ENCODE_SPEC}")
    p.add_argument("--metapaths", default="configs/metapaths.yaml")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--min_coverage", type=int, default=MIN_COVERAGE)
    p.add_argument("--output_dir", default="./results")
    return p.parse_args()


def open_replicates(encode_dir, output_assays):
    """Per assay: (bigWig A, bigWig B, k replicates in the pooled file, file names) or None."""
    path = os.path.join(encode_dir, "replicates", "manifest.json")
    if not os.path.exists(path):
        print(f"  [WARN] no replicate manifest at {path}", flush=True)
        return [None] * len(output_assays)
    manifest = json.load(open(path))
    out = []
    for a in output_assays:
        e = manifest["assays"].get(a)
        if not e or e.get("status") != "ok" or len(e["replicates"]) < 2:
            out.append(None)
            continue
        files = [os.path.join(encode_dir, "replicates", r["filename"]) for r in e["replicates"][:2]]
        if not all(os.path.exists(f) for f in files):
            print(f"  [WARN] {a}: replicate file missing in {encode_dir}/replicates", flush=True)
            out.append(None)
            continue
        out.append((pyBigWig.open(files[0]), pyBigWig.open(files[1]), max(len(e["pooled_replicates"]), 2),
                    [r["file"] for r in e["replicates"][:2]]))
    return out


def read(bw, chrom, start, end):
    raw = np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float32)
    return np.arcsinh(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))


class CellMeters:
    def __init__(self, K):
        self.ab, self.at, self.bt = PearsonMeter(K), PearsonMeter(K), PearsonMeter(K)   # on assay x coverage
        self.ab_b, self.bb, self.ab_ = PearsonMeter(K), PearsonMeter(K), PearsonMeter(K)  # + baseline mask
        self.n_tiles = 0


def spearman_brown(rho, k):
    return k * rho / (1 + (k - 1) * rho)


def summarize(cm, reps, output_assays):
    r_ab, r_at, r_bt = cm.ab.value, cm.at.value, cm.bt.value
    ct = partial_correlation(cm.ab_b.value, cm.bb.value, cm.ab_.value)
    mse = cm.ab.mse
    out = {}
    for k, a in enumerate(output_assays):
        if reps[k] is None or cm.ab.n[k] == 0:
            out[a] = None
            continue
        npool = reps[k][2]
        rho, rho_ct = float(r_ab[k]), float(ct[k])
        out[a] = {
            "files": reps[k][3], "k_pooled": npool, "n_positions": int(cm.ab.n[k]),
            "rep_r": rho, "rep_ct_r": rho_ct, "rep_mse": float(mse[k]),
            "ceiling_r": float(np.sqrt(spearman_brown(rho, npool))) if rho > 0 else None,
            "ceiling_ct_r": float(np.sqrt(spearman_brown(rho_ct, npool))) if rho_ct > 0 else None,
            "mse_floor": float(mse[k]) / (2 * npool),
            "a_vs_pooled_r": float(r_at[k]), "b_vs_pooled_r": float(r_bt[k]),
        }
    return out


def main():
    args = parse_args()
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    output_assays = args.output_assays or list(yaml.safe_load(open(ENCODE_SPEC))["assays"])
    baseline_cells = list(load_config_file(args.train_data_config)["cell_types"])
    metapaths = load_metapaths(args.metapaths)
    baseline_meta = resolve_baseline_cells(baseline_cells, metapaths, output_assays)
    K = len(output_assays)
    name = os.path.splitext(os.path.basename(args.train_data_config))[0]
    out_dir = os.path.join(args.output_dir, f"{now}_replicates_{name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Replicate ceiling | baseline cells {baseline_cells} | assays {output_assays} | out {out_dir}", flush=True)

    results = {}
    for cfg_path in args.eval_configs:
        cfg = load_config_file(cfg_path)
        meta = resolve_config_with_metapaths(cfg, metapaths, output_assays)
        ds = make_fiber_dataset(
            cfg.get("dataset_type", "single"), mode="eval", metadata=meta,
            fibers_per_entry=cfg.get("fibers_per_entry", 100), context_length=cfg.get("context_length", 5000),
            windows=cfg["windows"], max_regions=cfg.get("max_regions"), input_flags=[1, 1, 1, 1, 1],
            dna_type="none", output_assays=output_assays, seed=cfg.get("seed", 919),
            baseline_cells=baseline_meta, baseline_include_own=False)
        reps = {c: open_replicates(meta["cell_types"][c]["encode_dir"], output_assays) for c in cfg["cell_types"]}
        meters = {c: CellMeters(K) for c in cfg["cell_types"]}
        loader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, worker_init_fn=seed_worker,
                            multiprocessing_context=loader_context(args.num_workers))
        print(f"\n{cfg_path}: {list(cfg['cell_types'])}", flush=True)
        for i, batch in enumerate(loader):
            cell = batch["cell_type"][0]
            chrom, start, end = batch["locus"][0][0], int(batch["locus"][1][0]), int(batch["locus"][2][0])
            target = batch["target_bulk"].float()
            w = score_weights(batch["assay_mask"], batch["fiber_coverage"], args.min_coverage)   # (1, K, L)
            L = target.shape[-1]
            ra, rb, rmask = np.zeros((K, L), np.float32), np.zeros((K, L), np.float32), np.zeros(K, np.float32)
            for k, rep in enumerate(reps[cell]):
                if rep is not None:
                    ra[k], rb[k], rmask[k] = read(rep[0], chrom, start, end), read(rep[1], chrom, start, end), 1
            ra, rb = torch.from_numpy(ra)[None], torch.from_numpy(rb)[None]
            w = w * torch.from_numpy(rmask)[None, :, None]
            cm = meters[cell]
            cm.n_tiles += 1
            cm.ab.update(ra, rb, w)
            cm.at.update(ra, target, w)
            cm.bt.update(rb, target, w)
            b_w = w * batch["baseline_mask"].float()[:, :, None]
            base = batch["baseline_bulk"].float()
            cm.ab_b.update(ra, rb, b_w)
            cm.bb.update(base, rb, b_w)
            cm.ab_.update(ra, base, b_w)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} samples", flush=True)
        for c, cm in meters.items():
            res = summarize(cm, reps[c], output_assays)
            results[c] = {"eval_config": cfg_path, "n_tiles": cm.n_tiles, "per_assay": res}
            print(f"\n{c} ({cm.n_tiles} tiles): rep r / rep ct r / rep MSE | ceiling r / ceiling ct r / MSE floor "
                  f"| A, B vs pooled r (k = replicates pooled)", flush=True)
            for a, v in res.items():
                if v is None:
                    print(f"  {a:9s} -", flush=True)
                    continue
                f = lambda x: "  -  " if x is None else f"{x:.3f}"
                print(f"  {a:9s} {f(v['rep_r'])} / {f(v['rep_ct_r'])} / {f(v['rep_mse'])} | {f(v['ceiling_r'])} / "
                      f"{f(v['ceiling_ct_r'])} / {f(v['mse_floor'])} | {f(v['a_vs_pooled_r'])}, "
                      f"{f(v['b_vs_pooled_r'])} (k={v['k_pooled']})", flush=True)
        with open(os.path.join(out_dir, "replicate_summary.json"), "w") as fh:   # after each config
            json.dump({"train_data_config": args.train_data_config, "baseline_cells": baseline_cells,
                       "min_coverage": args.min_coverage, "results": results}, fh, indent=1)
    print(f"\nwrote {out_dir}/replicate_summary.json", flush=True)


if __name__ == "__main__":
    main()
