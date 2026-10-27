"""
One-off script for the Oct 30 presentation: single-molecule figures of the mixture test (T3, reports/Oct_23_T3.md).

The mixture model (trained only on GM12878 + HepG2 mixtures) is run on each cell type's fibers alone at chosen
chr21 tiles, and each figure shows a pair of assays (the flagged assay and its partner, PARTNER) with the paired
renderer of the eval (vis_utils._render_paired_assay): both cell types' tracks, the 50:50 mixture (what "bulk on
every molecule" gives both), each cell type's prediction from its own fibers, and every fiber's prediction
(GM12878 fibers above the cyan line, HepG2 below).

Tile choice, from the mixture eval's per-bin fiber-origin AUC table (<eval dir>/<PAIR>_paired.npz, no rerun of the
whole eval): rows are (tile, assay) differential bins of the pairwise loci; auc_oriented = chance that a fiber of
the cell type with the higher track scores above a fiber of the other. Bins with >= MIN_FIBERS fibers on each side.
Per direction (GM12878 higher / HepG2 higher):
  best:   per assay, the bin with the highest auc_oriented (so every assay, CTCF included, appears)
  random: per group (sharp / broad), RANDOM_PER_GROUP bins drawn from the rest
File names start with best_ / random_ so the showcase picks can be checked against unpicked ones.

Writes results/<YY-MM-DD_THH-MM-SS>_t3_single_molecule/: selected.tsv and
<sharp|broad>/<best|random>_<GM12878|HepG2>-high_<NN>_<chrom>_<start>_<flagged>-<partner>.png
(results/ is not synced; copy the good ones to reports/figures/oct30/).
Edit the constants in the EDIT ME block (no CLI by design).
Usage: python plots_code/26_oct_30/t3_single_molecule.py
"""

import csv
import datetime
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# ---- EDIT ME ----
CHECKPOINT = ("results/26-10-21_T00-55-17_T3_mix_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train02/"
              "Model_epoch_25.pt")
EVAL_DIR = "results/26-10-21_T02-06-54_T3_mix_data27_mix_GM12878_HepG2_200U_model15_uct_mid_8A_train02_eval"
PAIR = "eval24_pair_GM12878_HepG2_200U"
CELLS = ["GM12878", "HepG2_200U"]                 # A, B (order of the pair config)
PAIR_LOCI = "pairs/GM12878__HepG2"
WINDOWS = "test_chr21_tiles"
GROUPS = {"sharp": ["atac", "h3k4me3", "h3k27ac", "ctcf"],
          "broad": ["h3k4me1", "h3k27me3", "h3k36me3", "h3k9me3"]}
PARTNER = {"h3k27ac": "h3k4me3", "h3k4me3": "h3k27ac", "atac": "ctcf", "ctcf": "atac",
           "h3k27me3": "h3k9me3", "h3k9me3": "h3k27me3", "h3k4me1": "h3k36me3", "h3k36me3": "h3k4me1"}
MIN_FIBERS = 10                                   # fibers covering the bin, per cell type
RANDOM_PER_GROUP = 2                              # per group and direction
MIN_COVERAGE = 5
NUM_WORKERS = 4
SEED = 919
RUN_NAME = "t3_single_molecule"
# -----------------

DPI = 110


def cell_label(cell):
    return cell.replace("_200U", "")


def select(npz):
    """Rows of the per-bin AUC table to plot: [(row dict, selected, NN)]."""
    assays = [str(a) for a in npz["assays"]]
    cols = [str(c) for c in npz["auc_columns"]]
    auc = npz["auc"]
    rows = []
    for r in auc:
        r = dict(zip(cols, r))
        k = int(r["assay"])
        if min(r["n_a"], r["n_b"]) < MIN_FIBERS:
            continue
        t = int(r["tile"])
        rows.append({"chrom": str(npz["tile_chrom"][t]), "start": int(npz["tile_start"][t]), "assay": assays[k],
                     "group": next(g for g, a in GROUPS.items() if assays[k] in a), "d": float(r["d"]),
                     "auc": float(r["auc"]), "auc_oriented": float(r["auc_oriented"]),
                     "n_a": int(r["n_a"]), "n_b": int(r["n_b"]),
                     "direction": f"{cell_label(CELLS[0])}-high" if r["d"] > 0 else f"{cell_label(CELLS[1])}-high"})
    rng = random.Random(SEED)
    out = []
    for direction in sorted({r["direction"] for r in rows}):
        for group, g_assays in GROUPS.items():
            pool = [r for r in rows if r["direction"] == direction and r["group"] == group]
            best = []
            for a in g_assays:
                cands = sorted([r for r in pool if r["assay"] == a], key=lambda r: -r["auc_oriented"])
                if cands:
                    best.append(cands[0])
            rest = [r for r in pool if r not in best]
            rand = rng.sample(rest, min(RANDOM_PER_GROUP, len(rest)))
            out += [(r, "best", j) for j, r in enumerate(best)] + [(r, "random", j) for j, r in enumerate(rand)]
    print(f"[select] {len(rows)} differential bins with >= {MIN_FIBERS} fibers per cell type; plotting {len(out)}",
          flush=True)
    return out


def run_cell(model, cell, regions, mp, cfg, device):
    """{window: dict(pred, target, score, fibers, fiber_mask)} for one cell type's fibers alone at the regions."""
    import torch
    from data_utils import make_fiber_dataset, loader_context
    from utils import resolve_config_with_metapaths, unpack_batch
    from metrics.metrics import score_weights
    output_assays = model.init_args["output_assays"]
    meta = resolve_config_with_metapaths({"cell_types": {cell: None}}, mp, output_assays)
    ds = make_fiber_dataset("single", mode="eval", metadata=meta, fibers_per_entry=cfg["fibers_per_entry"],
                            context_length=cfg["context_length"], windows=WINDOWS,
                            input_flags=model.init_args["input_flags"], dna_type=model.init_args["dna_type"],
                            output_assays=output_assays, seed=SEED)
    ds.region_list = sorted(set(regions))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS,
                                         multiprocessing_context=loader_context(NUM_WORKERS))
    out = {}
    with torch.no_grad():
        for batch in loader:
            window = (batch["locus"][0][0], int(batch["locus"][1][0]), int(batch["locus"][2][0]))
            fibers_t, target_t, kwargs = unpack_batch(batch, device)
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                    enabled=device == "cuda"):
                pred, per_fiber = model(fibers_t, **kwargs)
            w = score_weights(batch["assay_mask"], batch["fiber_coverage"], MIN_COVERAGE)
            n = int(batch["n_fibers"][0])
            out[window] = {"pred": pred[0].float().cpu(), "target": target_t[0].float().cpu(), "score": w[0] > 0,
                           "fibers": per_fiber[0][:, :, :n].float().cpu(), "fiber_mask": batch["fiber_mask"][0][:, :n]}
    return out


def plot_pair(rec, ks, assays, row):
    """One row, one column per assay (flagged, partner), with the eval's paired renderer."""
    from vis_utils import _render_paired_assay, FS_SUPTITLE, FS_LEGEND
    height = 8.6
    fig = plt.figure(figsize=(22, height))
    grid = gridspec.GridSpec(1, len(ks), figure=fig, wspace=0.32, top=1 - 1.9 / height, bottom=0.08, left=0.06,
                             right=0.95)
    cells = [cell_label(c) for c in rec["cells"]]
    first = None
    for j, k in enumerate(ks):
        ax = _render_paired_assay(fig, grid[0, j], rec, k, assays[k], cells, rec["bin_size"])
        first = first or ax
    chrom, start, end = rec["locus"]
    hi = cells[0] if row["d"] > 0 else cells[1]
    fig.suptitle(f"{chrom}:{start:,}-{end:,}  |  mixture model, each cell type's fibers alone  |  "
                 f"{row['assay']}: {hi} higher, fiber-origin AUC {row['auc_oriented']:.2f} "
                 f"({row['n_a']} vs {row['n_b']} fibers)", fontsize=FS_SUPTITLE - 2, fontweight="bold",
                 y=1 - 0.25 / height)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1 - 0.8 / height))
    return fig


def main():
    import torch
    from models import BaseModel
    from utils import load_metapaths
    from regions import Loci

    os.chdir(REPO_ROOT)
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{now}_{RUN_NAME}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_cfg = BaseModel.load_model(os.path.join(REPO_ROOT, CHECKPOINT), map_location=device)
    model = model.to(device).eval()
    assays = model.init_args["output_assays"]
    cfg = {"context_length": int(ckpt_cfg.get("context_length", 5000)),
           "fibers_per_entry": int(ckpt_cfg.get("fibers_per_entry", 100))}
    print(f"[model] {CHECKPOINT}\n  aggregation {model.fiber_aggregation}, context {cfg['context_length']}, "
          f"fibers {cfg['fibers_per_entry']}", flush=True)

    npz = np.load(os.path.join(REPO_ROOT, EVAL_DIR, f"{PAIR}_paired.npz"))
    chosen = select(npz)
    L = cfg["context_length"]
    regions = sorted({(r["chrom"], r["start"], r["start"] + L) for r, _, _ in chosen})

    mp = load_metapaths(os.path.join(REPO_ROOT, "configs", "metapaths.yaml"))
    loci = Loci(mp["regions_root"], PAIR_LOCI)
    per_cell = {}
    for cell in CELLS:
        print(f"\n[run] {cell}: {len(regions)} tiles", flush=True)
        per_cell[cell] = run_cell(model, cell, regions, mp, cfg, device)

    table = []
    for r, sel, j in chosen:
        window = (r["chrom"], r["start"], r["start"] + L)
        if not all(window in per_cell[c] for c in CELLS):
            print(f"  {window}: missing fibers in one cell type; skipped", flush=True)
            continue
        A, B = (per_cell[c][window] for c in CELLS)
        d, bins = loci.lookup(r["chrom"], r["start"], assays)
        rec = {"locus": window, "cells": CELLS, "d": d, "rel_bins": np.where(bins >= 0, bins - r["start"], -1),
               "bin_size": loci.bin_size, "pred": [A["pred"], B["pred"]], "target": [A["target"], B["target"]],
               "score": [A["score"], B["score"]], "fibers": [A["fibers"], B["fibers"]],
               "fiber_mask": [A["fiber_mask"], B["fiber_mask"]]}
        ks = [assays.index(r["assay"]), assays.index(PARTNER[r["assay"]])]
        fig = plot_pair(rec, ks, assays, r)
        sub = os.path.join(out_dir, r["group"])
        os.makedirs(sub, exist_ok=True)
        name = f"{sel}_{r['direction']}_{j:02d}_{r['chrom']}_{r['start']}_{r['assay']}-{PARTNER[r['assay']]}.png"
        fig.savefig(os.path.join(sub, name), dpi=DPI)
        plt.close(fig)
        table.append({**r, "partner": PARTNER[r["assay"]], "selected": sel, "figure": f"{r['group']}/{name}"})
        print(f"  saved {r['group']}/{name}  (AUC {r['auc_oriented']:.2f}, d {r['d']:+.2f})", flush=True)

    with open(os.path.join(out_dir, "selected.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "direction", "selected", "assay", "partner", "chrom", "start", "d",
                                          "auc_oriented", "auc", "n_a", "n_b", "figure"], delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    print(f"\nDone. {len(table)} figures and selected.tsv in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
