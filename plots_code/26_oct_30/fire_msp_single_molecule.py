"""
One-off script for the Oct 30 presentation: single-molecule view of the fire_msp ablation (reports/Oct_16.md).

Two models, identical except for the fire_msp input channel (both trained on GM12878 + K562_200U + HepG2_200U,
old 4-assay pipeline, chr20 cCREs):
  with fire_msp:    model11 (inputs m6a, cpg, msp, nuc, fire_msp)
  without fire_msp: model12 (inputs m6a, cpg, msp, nuc)
Both run on the same windows and the same fibers; the without model simply does not get the fire_msp channel.

For each cell type in CELL_TYPES: from the chr21 cCRE-centred windows (the set the Oct 16 table was scored on),
keep windows where every assay in ASSAYS peaks above MIN_SIGNAL (asinh FC) in that cell type, and draw
NUM_LOCI of them (seeded). Per window, one figure with one column per assay:
  top row:   ENCODE track (GT) and both models' bulk predictions
  per fiber: the fiber's input channels (fire_msp included), then both models' per-fiber predictions overlaid
NUM_FIBERS fibers per figure, the same fibers in both columns: fibers spanning >= MIN_SPAN of the window with
summed m6a > M6A_THRESHOLD, preferring fibers with FIRE calls (up to half), seeded.

Writes results/26_oct_30_fire_msp/<cell>/<NN>_<chrom>_<start>-<end>_<assays>.png (results/ is not synced; pick
the good ones and copy them to reports/figures/oct30/).
Edit the constants in the EDIT ME block (no CLI by design).
Usage: python plots_code/26_oct_30/fire_msp_single_molecule.py
"""

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
CHECKPOINTS = {
    "with fire_msp": "results/26-09-22_T11-37-41_All_A_GM_K5_Hep_run_data17_GM12878_K562_200U_HepG2_200U_"
                     "model11_uct_mid_all_A_train01/Model_epoch_25.pt",
    "without fire_msp": "results/26-10-14_T16-57-00_All_A_All_T_no_fire_data17_GM12878_K562_200U_HepG2_200U_"
                        "model12_uct_mid_all_A_no_fire_msp_train01/Model_epoch_25.pt",
}
CELL_TYPES = ["GM12878", "K562_200U"]
ASSAYS = ["h3k27ac", "h3k4me3"]       # columns of each figure
WINDOWS = "test_chr21_ccre_centred"   # the windows of the Oct 16 table
NUM_LOCI = 10                         # per cell type
MIN_SIGNAL = 1.5                      # window max of asinh(FC) every assay in ASSAYS must exceed
NUM_FIBERS = 7
MIN_SPAN = 0.8                        # fraction of the window a fiber must cover
M6A_THRESHOLD = 200                   # summed m6a calls a fiber must exceed
SEED = 919
OUT_DIR = os.path.join(REPO_ROOT, "results", "26_oct_30_fire_msp")
# -----------------

FS_SUPTITLE = 38
FS_TITLE = 34
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 28
FS_FIBER_TITLE = 24
FS_FIBER_YLABEL = 20
DPI = 120
SM_COL_WIDTH = 16
SM_HEIGHT = 32

plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
                     "axes.labelsize": FS_LABEL})

FEATURE_NAMES = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
FEATURE_COLORS = ["black", "purple", "blue", "green", "red"]
UNMETHYLATED_COLOR = "lightslategrey"
MODEL_COLORS = {"with fire_msp": "crimson", "without fire_msp": "royalblue"}
ASSAY_LABELS = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3"}


#--------------------------------------------------------------------------------------------------
# Plotting (fiber-feature renderer copied from plots_code/ceehrc_26/single_molecule_plotter.py)

def _render_single_fiber_features(ax, fiber_inp):
    """All five input channels of one fiber on one Axes, one row per channel. fiber_inp: (5, L)."""
    for k, (name, color) in enumerate(zip(FEATURE_NAMES, FEATURE_COLORS)):
        feat = fiber_inp[k]
        if name in ("m6a", "cpg"):
            off = np.where(feat < -0.5)[0]
            if len(off):
                ax.scatter(off, np.full_like(off, -k), marker='|', color=UNMETHYLATED_COLOR, s=15, alpha=0.35,
                           linewidths=0.6)
            on = np.where(feat > 0.5)[0]
            if len(on):
                ax.scatter(on, np.full_like(on, -k), marker='|', color=color, s=25, alpha=0.85, linewidths=1.0)
        else:
            diff = np.diff((feat > 0.5).astype(float), prepend=0.0, append=0.0)
            for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
                ax.axhspan(-k - 0.35, -k + 0.35, xmin=s / len(feat), xmax=e / len(feat), color=color, alpha=0.5,
                           lw=0)
    ax.set_ylim(-len(FEATURE_NAMES) + 0.5, 0.5)
    ax.set_yticks([])


def plot_fire_comparison(target, preds, fibers_in, fiber_preds, fibers, locus, cell_label):
    """
    target: {assay: (L,)}; preds: {model: {assay: (L,)}}; fibers_in: (5, L, N);
    fiber_preds: {model: {assay: (L, N)}}; fibers: selected fiber indices.
    One column per assay: bulk row, then per fiber [input channels, per-fiber predictions of both models].
    """
    n, L = len(fibers), fibers_in.shape[1]
    x = np.arange(L)
    chrom, s, e = locus
    fig = plt.figure(figsize=(SM_COL_WIDTH * len(ASSAYS), SM_HEIGHT))
    fig.subplots_adjust(top=0.91, bottom=0.05, left=0.07, right=0.985)
    fig.suptitle(f"{cell_label}: {chrom}:{s:,}-{e:,}", fontsize=FS_SUPTITLE, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(1 + n, len(ASSAYS), figure=fig, hspace=0.6, wspace=0.18, height_ratios=[1.3] + [1] * n)

    for col, assay in enumerate(ASSAYS):
        ax_bulk = fig.add_subplot(gs[0, col])
        ax_bulk.plot(x, target[assay], color='dimgray', lw=3, label='ENCODE')
        for m, p in preds.items():
            ax_bulk.plot(x, p[assay], color=MODEL_COLORS[m], lw=2.5, alpha=0.9, label=m)
        ax_bulk.set_xlim(0, L)
        ax_bulk.set_ylabel("asinh FC", fontsize=FS_LABEL, fontweight='bold')
        ax_bulk.set_title(ASSAY_LABELS.get(assay, assay), fontsize=FS_TITLE, fontweight='bold', pad=10)
        ax_bulk.set_xticklabels([])
        if col == 0:
            ax_bulk.legend(loc='upper left', frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.55))

        ymax = max(1.0, max(float(fp[assay][:, fibers].max()) for fp in fiber_preds.values()))
        for i, f in enumerate(fibers):
            sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i + 1, col], hspace=0.0,
                                                   height_ratios=[1, 1.2])
            ax_in = fig.add_subplot(sub[0, 0], sharex=ax_bulk)
            _render_single_fiber_features(ax_in, fibers_in[:, :, f])
            fire_on = (fibers_in[4, :, f] > 0.5).astype(int)
            n_fire = int(np.count_nonzero(np.diff(fire_on, prepend=0) == 1))   # FIRE elements = runs of calls
            ax_in.set_title(f"Fiber #{f}  ({n_fire} FIRE element{'s' if n_fire != 1 else ''})",
                            fontsize=FS_FIBER_TITLE, fontweight='bold', loc='left', pad=4)
            ax_in.tick_params(labelbottom=False)
            ax_p = fig.add_subplot(sub[1, 0], sharex=ax_bulk)
            for m, fp in fiber_preds.items():
                ax_p.plot(x, fp[assay][:, f], color=MODEL_COLORS[m], lw=2.0)
                ax_p.fill_between(x, 0, fp[assay][:, f], color=MODEL_COLORS[m], alpha=0.15)
            ax_p.set_ylim(0, ymax * 1.05)
            ax_p.set_ylabel("Imputed", fontsize=FS_FIBER_YLABEL, fontweight='bold')
            if i < n - 1:
                ax_p.tick_params(labelbottom=False)
            else:
                ax_p.set_xlabel("Position in window (bp)", fontsize=FS_LABEL, fontweight='bold')
    return fig


#--------------------------------------------------------------------------------------------------

def select_fibers(fibers_in, fiber_mask, rng):
    """Up to NUM_FIBERS fibers spanning >= MIN_SPAN of the window with summed m6a > M6A_THRESHOLD; up to half
    of them with FIRE calls (so the channel the models differ in is visible). Returned in fiber order."""
    L = fibers_in.shape[1]
    span = fiber_mask.sum(0) / L
    m6a = (fibers_in[0] > 0.5).sum(0)
    ok = np.where((span >= MIN_SPAN) & (m6a > M6A_THRESHOLD))[0].tolist()
    fire = [f for f in ok if (fibers_in[4, :, f] > 0.5).any()]
    rest = [f for f in ok if f not in fire]
    pick = rng.sample(fire, min(len(fire), NUM_FIBERS // 2 + NUM_FIBERS % 2))
    pick += rng.sample(rest, min(len(rest), NUM_FIBERS - len(pick)))
    if len(pick) < NUM_FIBERS:
        pick += rng.sample([f for f in fire if f not in pick], min(NUM_FIBERS - len(pick), len(fire) - len(pick)))
    return sorted(pick)


def main():
    import torch
    from models import BaseModel
    from utils import load_metapaths, resolve_config_with_metapaths, unpack_batch
    from data_utils import make_fiber_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, cfgs = {}, {}
    for label, path in CHECKPOINTS.items():
        model, cfg = BaseModel.load_model(os.path.join(REPO_ROOT, path), map_location=device)
        models[label], cfgs[label] = model.to(device).eval(), cfg
        print(f"[{label}] {path}\n  inputs {model.init_args['input_flags']}, assays {model.init_args['output_assays']}, "
              f"aggregation {model.fiber_aggregation}, context {cfg.get('context_length')}, "
              f"fibers {cfg.get('fibers_per_entry')}", flush=True)
    ref = cfgs["with fire_msp"]
    output_assays = models["with fire_msp"].init_args["output_assays"]
    assay_idx = {a: output_assays.index(a) for a in ASSAYS}
    context, n_fibers = int(ref.get("context_length", 5000)), int(ref.get("fibers_per_entry", 100))

    mp = load_metapaths(os.path.join(REPO_ROOT, "configs", "metapaths.yaml"))
    for cell in CELL_TYPES:
        cell_label = cell.replace("_200U", "")
        out_dir = os.path.join(OUT_DIR, cell_label)
        os.makedirs(out_dir, exist_ok=True)
        meta = resolve_config_with_metapaths({"cell_types": {cell: None}}, mp, output_assays)
        # All five channels are loaded; the without model is given the first four (fire_msp is the last one)
        ds = make_fiber_dataset("single", mode="eval", metadata=meta, fibers_per_entry=n_fibers,
                                context_length=context, windows=WINDOWS, input_flags=[1, 1, 1, 1, 1],
                                dna_type=models["with fire_msp"].init_args["dna_type"], output_assays=output_assays,
                                seed=SEED)
        ds.init_worker_resources()

        # Windows with a clear peak in every assay of the figure (cheap: bigWig lookups only)
        strong = []
        for region in ds.region_list:
            sig = ds.get_other_bw_data(0, *ds.window_for_region(region))
            if all(float(sig[assay_idx[a]].max()) > MIN_SIGNAL for a in ASSAYS):
                strong.append(region)
        rng = random.Random(SEED)
        chosen = sorted(rng.sample(strong, min(NUM_LOCI, len(strong))), key=lambda r: r[1])
        print(f"\n[{cell}] {len(strong)} of {len(ds.region_list)} {WINDOWS} windows have "
              f"{' and '.join(ASSAYS)} > {MIN_SIGNAL}; plotting {len(chosen)}", flush=True)
        ds.region_list = chosen

        with torch.no_grad():
            for j, batch in enumerate(torch.utils.data.DataLoader(ds, batch_size=1)):
                locus = (batch["locus"][0][0], int(batch["locus"][1][0]), int(batch["locus"][2][0]))
                fibers_t, target_t, kwargs = unpack_batch(batch, device)
                fibers_in = fibers_t[0].float().cpu().numpy()                    # (5, L, N)
                fiber_mask = batch["fiber_mask"][0].numpy()                      # (L, N)
                fibers = select_fibers(fibers_in, fiber_mask, random.Random(f"{SEED}:{locus}"))
                if not fibers:
                    print(f"  {locus}: no fiber passes the span/m6a filter; skipped", flush=True)
                    continue
                preds, fiber_preds = {}, {}
                for label, model in models.items():
                    flags = model.init_args["input_flags"]
                    x_in = fibers_t[:, [c for c in range(5) if flags[c]]]
                    with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                            enabled=device == "cuda"):
                        bulk, per_fiber = model(x_in, **kwargs)
                    bulk, per_fiber = bulk[0].float().cpu().numpy(), per_fiber[0].float().cpu().numpy()
                    preds[label] = {a: bulk[assay_idx[a]] for a in ASSAYS}
                    fiber_preds[label] = {a: per_fiber[assay_idx[a]] for a in ASSAYS}
                target = {a: target_t[0, assay_idx[a]].float().cpu().numpy() for a in ASSAYS}
                fig = plot_fire_comparison(target, preds, fibers_in, fiber_preds, fibers, locus, cell_label)
                name = f"{j:02d}_{locus[0]}_{locus[1]}-{locus[2]}_{'-'.join(ASSAYS)}.png"
                fig.savefig(os.path.join(out_dir, name), dpi=DPI)
                plt.close(fig)
                print(f"  saved {cell_label}/{name}  (fibers {fibers})", flush=True)

    print(f"\nDone. Figures in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
