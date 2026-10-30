"""
One-off script for the Oct 30 presentation: single-molecule figures of the mixture test (T3) with the Fiber-seq
inputs. One assay per figure (successor of t3_single_molecule.py, which showed the flagged assay and a partner).

The mixture model (trained only on GM12878 + HepG2 mixtures) is run on each cell type's fibers alone, in a window
centred on a differential 1 kb bin of the pairwise loci. Each figure, top to bottom:
  1. bulk: both ENCODE tracks, the 50:50 mixture (what "bulk on every molecule", H0, gives both cell types) and each
     cell type's prediction from its own fibers; the differential bin shaded
  2. every fiber's prediction (GM12878 above the cyan line, HepG2 below), sorted by the bin within each cell type
  3. the same fibers in the same order, drawn from their inputs: nucleosome / MSP (accessible) / FIRE
  4. NUM_EXAMPLES example fibers per cell type (spread over its ranking, preferring fibers that cover the window on
     both sides of the bin): all input channels and the fiber's prediction against the mixture (dashed)
The AUC in the title is recomputed from the fibers shown (each fiber's mean prediction over the bin, fibers covering
>= half of it): the chance a fiber of the cell type with the higher track scores above one of the other.

Bins are the ones t3_single_molecule.py picks (its select(): per direction, best per assay by the eval's
fiber-origin AUC, plus random ones). Writes results/<YY-MM-DD_THH-MM-SS>_t3_single_molecule_fibers/:
selected.tsv and <narrow|broad>/<best|random>_<GM12878|HepG2>-high_<NN>_<chrom>_<start>-<end>_<assay>.png
(results/ is not synced; copy the good ones to plots/26_oct_30/t3/). Runs on CPU or GPU.
Usage: python plots_code/26_oct_30/t3_single_molecule_fibers.py
"""

import csv
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
import t3_single_molecule as t3                  # select(), run constants
import single_molecule_dev5 as sm                # fiber rendering, fiber choice, bin means

# ---- EDIT ME ----
NUM_EXAMPLES = 3                                 # example fibers per cell type
RANK_WEIGHT = 3.0                                # as in single_molecule_dev5: rank vs centring trade-off
RUN_NAME = "t3_single_molecule_fibers"
# -----------------

CELLS = t3.CELLS
GROUP_OF = {a: ("narrow" if g == "sharp" else "broad") for g, assays in t3.GROUPS.items() for a in assays}
CELL_COLORS = ["tab:blue", "tab:red"]            # tracks, labels (as the eval's paired dashboards)
PRED_COLORS = ["navy", "darkred"]                # predictions
BIN_COLOR = "#f2c94c"
# input heatmap categories: 0 covered (linker / unlabelled), 1 nucleosome, 2 MSP, 3 FIRE; uncovered = blank
INPUT_CMAP = ListedColormap(["#d9d9d9", "#7fbf7f", "#7b7bff", "#ff5a5a"])
INPUT_CMAP.set_bad("white")
FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_FIBER = 24, 18, 15, 12, 14, 13
DPI = 100


def cell_label(cell):
    return cell.replace("_200U", "")


def run_cell(model, cell, regions, mp, cfg, device):
    """{window: out} for one cell type's fibers alone; out as single_molecule_dev5.run_model (per_fiber, fibers_in,
    mask, target, bulk), without the baseline."""
    import torch
    from data_utils import make_fiber_dataset, loader_context
    from utils import resolve_config_with_metapaths, unpack_batch
    output_assays = model.init_args["output_assays"]
    meta = resolve_config_with_metapaths({"cell_types": {cell: None}}, mp, output_assays)
    ds = make_fiber_dataset("single", mode="eval", metadata=meta, fibers_per_entry=cfg["fibers_per_entry"],
                            context_length=cfg["context_length"], windows=t3.WINDOWS,
                            input_flags=model.init_args["input_flags"], dna_type=model.init_args["dna_type"],
                            output_assays=output_assays, seed=t3.SEED)
    ds.region_list = sorted(set(regions))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=t3.NUM_WORKERS,
                                         multiprocessing_context=loader_context(t3.NUM_WORKERS))
    out = {}
    with torch.no_grad():
        for batch in loader:
            window = (batch["locus"][0][0], int(batch["locus"][1][0]), int(batch["locus"][2][0]))
            fibers_t, target_t, kwargs = unpack_batch(batch, device)
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                    enabled=device == "cuda"):
                bulk, per_fiber = model(fibers_t, **kwargs)
            n = int(batch["n_fibers"][0])
            out[window] = {"target": target_t[0].float().cpu().numpy(),                  # (K, L)
                           "bulk": bulk[0].float().cpu().numpy(),                        # (K, L)
                           "per_fiber": per_fiber[0][:, :, :n].float().cpu().numpy(),    # (K, L, n)
                           "fibers_in": fibers_t[0][:, :, :n].float().cpu().numpy(),     # (5, L, n)
                           "mask": batch["fiber_mask"][0][:, :n].numpy().astype(bool)}   # (L, n)
    return out


def input_categories(fibers_in, mask):
    """(n, L) category per fiber and position from the input channels (see INPUT_CMAP); NaN where not covered."""
    names = sm.FEATURE_NAMES
    msp, nuc, fire = (fibers_in[names.index(c)] > 0.5 for c in ("msp", "nuc", "fire_msp"))   # (L, n)
    cat = np.zeros(mask.shape, np.float32)
    cat[nuc] = 1
    cat[msp] = 2
    cat[fire] = 3
    return np.where(mask, cat, np.nan).T


def oriented_auc(scores_hi, scores_lo):
    """Chance a fiber of the cell type with the higher track scores above one of the other (ties count half)."""
    a, b = np.asarray(scores_hi)[:, None], np.asarray(scores_lo)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def plot_bin(outs, k, b, row):
    """outs: [GM12878 out, HepG2 out] in the centred window; k assay index, b bin index in the window."""
    L = outs[0]["mask"].shape[0]
    x = np.arange(L)
    b0, b1 = b * sm.BIN, (b + 1) * sm.BIN
    sm.NUM_FIBERS, sm.RANK_WEIGHT = NUM_EXAMPLES, RANK_WEIGHT
    choice = [sm.choose_fibers(o, k, b) for o in outs]          # (order, picks, n_covering) per cell type
    n_rows = NUM_EXAMPLES
    height = 4.2 + 3.6 + 3.6 + 2.5 * n_rows + 1.6
    fig = plt.figure(figsize=(24, height))
    gs = gridspec.GridSpec(3 + n_rows, 2, figure=fig, height_ratios=[1.5, 1.6, 1.6] + [1.15] * n_rows,
                           hspace=0.5, wspace=0.14, top=1 - 2.9 / height, bottom=0.04, left=0.07, right=0.95)

    # 1. bulk
    tA, tB = outs[0]["target"][k], outs[1]["target"][k]
    mix = np.arcsinh(0.5 * np.sinh(tA.astype(np.float64)) + 0.5 * np.sinh(tB.astype(np.float64)))
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(b0, b1, color=BIN_COLOR, alpha=0.25, lw=0)
    for t, c in zip((tA, tB), CELL_COLORS):
        ax.fill_between(x, 0, t, color=c, alpha=0.18, lw=0)
        ax.plot(x, t, color=c, lw=0.7, alpha=0.6)
    ax.plot(x, mix, color="black", lw=1.4, ls="--")
    for o, c in zip(outs, PRED_COLORS):
        ax.plot(x, o["bulk"][k], color=c, lw=2.6)
    ax.set_xlim(0, L)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("asinh FC", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(f"{sm.ASSAY_LABELS[row['assay']]}: tracks and each cell type's prediction from its own fibers",
                 fontsize=FS_TITLE, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FS_TICK, labelbottom=False)

    # 2-3. prediction and input heatmaps, one row per fiber, same order
    orders = [c[0] for c in choice]
    pred = np.concatenate([np.where(o["mask"], o["per_fiber"][k], np.nan)[:, order].T for o, order in zip(outs, orders)])
    cats = np.concatenate([input_categories(o["fibers_in"], o["mask"])[order] for o, order in zip(outs, orders)])
    n_a = len(orders[0])
    rank = [{f: i for i, f in enumerate(order)} for order in orders]
    ticks = [rank[0][f] + 0.5 for f in choice[0][1]] + [n_a + rank[1][f] + 0.5 for f in choice[1][1]]
    tick_labels = [f"#{f}" for f in choice[0][1]] + [f"#{f}" for f in choice[1][1]]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    for r_, (mat, kw, title) in enumerate((
            (pred, dict(cmap=cmap, vmin=0, vmax=max(float(np.nanpercentile(pred, 99)), 0.5)),
             "Every fiber's prediction"),
            (cats, dict(cmap=INPUT_CMAP, vmin=-0.5, vmax=3.5), "The same fibers' Fiber-seq inputs"))):
        axh = fig.add_subplot(gs[1 + r_, :], sharex=ax)
        im = axh.imshow(np.ma.masked_invalid(mat), aspect="auto", interpolation="nearest",
                        extent=(0, L, mat.shape[0], 0), **kw)
        axh.axhline(n_a, color="cyan", lw=2.5)
        for xv in (b0, b1):
            axh.axvline(xv, color=BIN_COLOR, lw=1.3, ls="--")
        axh.set_yticks(ticks)
        axh.set_yticklabels(tick_labels, fontsize=FS_TICK - 2)
        for lab, c in zip(axh.get_yticklabels(), [CELL_COLORS[0]] * len(choice[0][1]) + [CELL_COLORS[1]] * len(choice[1][1])):
            lab.set_color(c)
        n_all = mat.shape[0]
        for lo, n, c, cell in ((0, n_a, CELL_COLORS[0], CELLS[0]), (n_a, n_all - n_a, CELL_COLORS[1], CELLS[1])):
            axh.text(-0.025, 1 - (lo + n / 2) / n_all, f"{cell_label(cell)}\n({n} fibers)", transform=axh.transAxes,
                     ha="right", va="center", fontsize=FS_LABEL - 3, fontweight="bold", color=c)
        axh.set_title(title + " (sorted by the differential bin within each cell type)", fontsize=FS_LABEL,
                      fontweight="bold", loc="left")
        axh.tick_params(labelsize=FS_TICK, labelbottom=False)
        if r_ == 0:
            cax = axh.inset_axes([1.01, 0, 0.012, 1])
            fig.colorbar(im, cax=cax).ax.tick_params(labelsize=FS_TICK - 2)

    # 4. example fibers, one column per cell type
    ymax = max(1.0, max(float(np.max(o["per_fiber"][k][:, f])) for o, c in zip(outs, choice) for f in c[1]),
               float(np.max(mix))) * 1.08
    for col, (o, (order, picks, n_cov)) in enumerate(zip(outs, choice)):
        for i, f in enumerate(picks):
            sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3 + i, col], hspace=0.0, height_ratios=[1, 1.3])
            ax_in = fig.add_subplot(sub[0, 0])
            sm._render_single_fiber_features(ax_in, o["fibers_in"][:, :, f])
            ax_in.set_xlim(0, L)
            ax_in.set_title(f"{cell_label(CELLS[col])} fiber #{f}  (rank {rank[col][f] + 1} of {n_cov} covering the bin)",
                            fontsize=FS_FIBER, fontweight="bold", loc="left", pad=3, color=CELL_COLORS[col])
            ax_in.tick_params(labelbottom=False)
            ax_p = fig.add_subplot(sub[1, 0], sharex=ax_in)
            ax_p.axvspan(b0, b1, color=BIN_COLOR, alpha=0.2, lw=0)
            y = np.where(o["mask"][:, f], o["per_fiber"][k][:, f], np.nan)
            ax_p.plot(x, mix, color="black", ls="--", lw=1.2)
            ax_p.fill_between(x, 0, y, color=PRED_COLORS[col], alpha=0.2, lw=0)
            ax_p.plot(x, y, color=PRED_COLORS[col], lw=1.8)
            ax_p.set_ylim(0, ymax)
            ax_p.set_ylabel("Imputed", fontsize=FS_LABEL - 3, fontweight="bold")
            ax_p.tick_params(labelsize=FS_TICK - 1)
            if i < len(picks) - 1:
                ax_p.tick_params(labelbottom=False)
            else:
                ax_p.set_xlabel("Position in window (bp)", fontsize=FS_LABEL, fontweight="bold")

    # title and legend
    hi_col = 0 if row["d"] > 0 else 1
    fm = [sm.fiber_bin_means(o, k, b) for o in outs]
    fm = [v[~np.isnan(v)] for v in fm]
    auc = oriented_auc(fm[hi_col], fm[1 - hi_col]) if min(len(v) for v in fm) > 0 else float("nan")
    chrom, s, e = row["window"]
    fig.suptitle(f"Mixture model, each cell type's fibers alone: {chrom}:{s:,}-{e:,}\n"
                 f"{sm.ASSAY_LABELS[row['assay']]} higher in {cell_label(CELLS[hi_col])}: fiber-origin AUC {auc:.2f} "
                 f"({len(fm[0])} vs {len(fm[1])} fibers covering the bin)", fontsize=FS_SUPTITLE - 2,
                 fontweight="bold", y=1 - 0.2 / height)
    handles = [Patch(color=CELL_COLORS[0], alpha=0.35, label=f"{cell_label(CELLS[0])} track"),
               Patch(color=CELL_COLORS[1], alpha=0.35, label=f"{cell_label(CELLS[1])} track"),
               Line2D([], [], color="black", ls="--", lw=1.4, label="50:50 mixture (H0: every molecule)"),
               Line2D([], [], color=PRED_COLORS[0], lw=2.6, label=f"{cell_label(CELLS[0])} prediction / fiber"),
               Line2D([], [], color=PRED_COLORS[1], lw=2.6, label=f"{cell_label(CELLS[1])} prediction / fiber"),
               Patch(color=BIN_COLOR, alpha=0.4, label="Differential 1 kb bin"),
               Patch(color="#d9d9d9", label="covered (linker)"), Patch(color="#7fbf7f", label="Nucleosome"),
               Patch(color="#7b7bff", label="MSP (accessible)"), Patch(color="#ff5a5a", label="FIRE"),
               Patch(color="black", alpha=0.7, label="m6A"), Patch(color="purple", alpha=0.7, label="CpG (5mC)")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1 - 1.3 / height))
    return fig, auc


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
    model, ckpt_cfg = BaseModel.load_model(os.path.join(REPO_ROOT, t3.CHECKPOINT), map_location=device)
    model = model.to(device).eval()
    assays = model.init_args["output_assays"]
    cfg = {"context_length": int(ckpt_cfg.get("context_length", 5000)),
           "fibers_per_entry": int(ckpt_cfg.get("fibers_per_entry", 100))}
    L = cfg["context_length"]
    print(f"[model] {t3.CHECKPOINT} on {device}", flush=True)

    npz = np.load(os.path.join(REPO_ROOT, t3.EVAL_DIR, f"{t3.PAIR}_paired.npz"))
    chosen = t3.select(npz)
    mp = load_metapaths(os.path.join(REPO_ROOT, "configs", "metapaths.yaml"))
    loci = Loci(mp["regions_root"], t3.PAIR_LOCI)

    # centre each pick on its differential bin
    jobs = []
    for r, sel, j in chosen:
        _, bins = loci.lookup(r["chrom"], r["start"], [r["assay"]])
        c = sm.centre_on_bin({"region": (r["chrom"], r["start"], r["start"] + L),
                              "bin": (int(bins[0]) - r["start"]) // sm.BIN}, L)
        jobs.append((r, sel, j, c["region"], c["bin"]))
    regions = sorted({reg for *_, reg, _ in jobs})

    per_cell = {}
    for cell in CELLS:
        print(f"\n[run] {cell}: {len(regions)} windows", flush=True)
        outs = run_cell(model, cell, regions, mp, cfg, device)
        # key by region (the dataset reports the window it read)
        per_cell[cell] = {reg: outs.get((reg[0], (reg[1] + reg[2]) // 2 - L // 2, (reg[1] + reg[2]) // 2 - L // 2 + L))
                          for reg in regions}

    table = []
    for r, sel, j, reg, b in jobs:
        outs = [per_cell[c][reg] for c in CELLS]
        if any(o is None for o in outs):
            print(f"  {reg} {r['assay']}: no fibers in one cell type; skipped", flush=True)
            continue
        centre = (reg[1] + reg[2]) // 2
        r = dict(r, window=(reg[0], centre - L // 2, centre - L // 2 + L))
        fig, auc = plot_bin(outs, assays.index(r["assay"]), b, r)
        group = GROUP_OF[r["assay"]]
        os.makedirs(os.path.join(out_dir, group), exist_ok=True)
        chrom, s, e = r["window"]
        name = f"{sel}_{r['direction']}_{j:02d}_{chrom}_{s}-{e}_{r['assay']}.png"
        fig.savefig(os.path.join(out_dir, group, name), dpi=DPI)
        plt.close(fig)
        table.append({**r, "group": group, "selected": sel, "auc_shown": round(auc, 3), "figure": f"{group}/{name}"})
        print(f"  saved {group}/{name}  (eval AUC {r['auc_oriented']:.2f}, shown {auc:.2f}, d {r['d']:+.2f})",
              flush=True)

    with open(os.path.join(out_dir, "selected.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "direction", "selected", "assay", "chrom", "start", "d",
                                          "auc_oriented", "auc_shown", "n_a", "n_b", "figure"], delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    print(f"\nDone. {len(table)} figures and selected.tsv in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
