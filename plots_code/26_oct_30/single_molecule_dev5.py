"""
One-off script for the Oct 30 presentation: single-molecule figures of the DEV5 long model (100 epochs, the model
behind the bulk-recovery numbers), on chr21 test tiles, for the 5 dev and the 3 held-out cell types.

Every figure shows one tile and a pair of assays (the flagged assay and its partner, PARTNER), with the ENCODE
track, the model's average over fibers and the cross-cell-average baseline on top, a heatmap of every fiber's
prediction, and NUM_FIBERS example fibers (input channels + prediction, with the baseline as a reference line: what
a bulk baseline would give every fiber).

Three passes in one job:
  0. Screen (ENCODE tracks only, no fibers, no model): per cell type, tile and assay, cut the tile into BIN bp bins;
     cell = asinh(mean FC of the cell), base = asinh(mean over the dev cell lines except the cell's own of their
     mean FC), i.e. the baseline plotted (the loci rule of data_prep/loci_spec.yaml, against this baseline).
     Bins count when max(cell, base) >= MIN_SIGNAL; d = cell - base at the counting bin with the largest |d|.
       gain / loss: |d| >= DIFF_THRESHOLD (the cell departs from the bulk average)
       agree:       |d| <  AGREE_MAX_D and the cell has a peak >= MIN_SIGNAL (cell and average line up)
     Per cell type x group (sharp / broad) x category, keep CANDIDATES tiles: ranked by |d| (agree: peak height),
     round robin over the group's assays so every assay (CTCF included) is represented.
  1. Rank: run the model on every candidate; write ranking.tsv (bin values of target / model / baseline, fraction
     of the departure from the baseline the model recovers, tile Pearson of model and baseline, fiber spread).
  2. Plot: per cell type x group x category, the PLOT_BEST best by score and PLOT_RANDOM drawn at random from the
     rest (file names start with best_ / random_), so the showcase picks can be checked against unpicked ones.

Writes results/<YY-MM-DD_THH-MM-SS>_single_molecule_dev5/: ranking.tsv and
<dev|test>/<cell>/<sharp|broad>/<best|random>_<category>_<NN>_<chrom>_<start>-<end>_<flagged>-<partner>.png
(results/ is not synced; copy the good ones to reports/figures/oct30/).
Edit the constants in the EDIT ME block (no CLI by design).
Usage: python plots_code/26_oct_30/single_molecule_dev5.py
"""

import csv
import datetime
import os
import random
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# ---- EDIT ME ----
CHECKPOINT = ("results/26-10-18_T10-14-23_DEV5_long_data20_dev5_200U_model15_uct_mid_8A_train03/"
              "Model_epoch_100.pt")
DEV_CELLS = ["GM12878", "K562_200U", "HepG2_200U", "Panc1_200U", "H1_200U"]   # training cells = baseline cells
HELD_OUT = ["H9_200U", "HCT116_200U", "MCF_7_200U"]
WINDOWS = "test_chr21_tiles"
GROUPS = {"sharp": ["atac", "h3k4me3", "h3k27ac", "ctcf"],
          "broad": ["h3k4me1", "h3k27me3", "h3k36me3", "h3k9me3"]}
PARTNER = {"h3k27ac": "h3k4me3", "h3k4me3": "h3k27ac", "atac": "ctcf", "ctcf": "atac",
           "h3k27me3": "h3k9me3", "h3k9me3": "h3k27me3", "h3k4me1": "h3k36me3", "h3k36me3": "h3k4me1"}
PARTNER_FALLBACK = {"ctcf": "h3k27ac"}   # cells without the partner's track (H1 / H9 have no ATAC)
BIN = 1000
MIN_SIGNAL = 1.5
DIFF_THRESHOLD = 0.5
AGREE_MAX_D = 0.25
CANDIDATES = {"gain": 25, "loss": 10, "agree": 15}   # per cell type x group, scored by the model
PLOT_BEST = {"gain": 4, "loss": 2, "agree": 3}
PLOT_RANDOM = {"gain": 2, "loss": 1, "agree": 2}
MIN_FIBERS = 20                                      # tiles with fewer fibers are ranked but not plotted
MIN_COVERAGE = 5                                     # scored positions (as in the eval)
NUM_FIBERS = 5                                       # example fibers per figure
MIN_SPAN = 0.6                                       # fraction of the window an example fiber should cover
NUM_WORKERS = 4
SEED = 919
RUN_NAME = "single_molecule_dev5"
# A previous run dir (env var, set by replot_single_molecule_cpu.sh): skip screen + ranking, re-plot only the
# figures listed in its ranking.tsv
REPLOT_FROM = os.environ.get("REPLOT_FROM")
# -----------------

FS_SUPTITLE = 30
FS_TITLE = 24
FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 17
FS_FIBER_TITLE = 16
DPI = 100
COL_WIDTH = 13
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
                     "axes.labelsize": FS_LABEL})

FEATURE_NAMES = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
FEATURE_LABELS = ["m6A", "CpG (5mC)", "MSP (accessible)", "Nucleosome", "FIRE"]
FEATURE_COLORS = ["black", "purple", "blue", "green", "red"]
UNMETHYLATED_COLOR = "lightslategrey"
MODEL_COLOR = "#2a78d6"
TARGET_COLOR = "0.55"
BIN_COLOR = "#f2c94c"
ASSAY_LABELS = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac", "h3k27me3": "H3K27me3",
                "h3k4me1": "H3K4me1", "h3k36me3": "H3K36me3", "h3k9me3": "H3K9me3", "ctcf": "CTCF"}
TSV_FIELDS = ["split", "cell", "group", "category", "flagged", "partner", "chrom", "start", "end", "bin", "d_screen",
              "n_fibers", "target_bin", "model_bin", "baseline_bin", "frac_recovered", "r_model", "r_baseline",
              "r_model_partner", "r_baseline_partner", "mse_model", "mse_baseline", "mse_model_partner",
              "mse_baseline_partner", "fiber_spread", "score", "selected", "figure"]


def cell_label(cell):
    return cell.replace("_200U", "").replace("MCF_7", "MCF-7")


def group_of(assay):
    return next(g for g, assays in GROUPS.items() if assay in assays)


#--------------------------------------------------------------------------------------------------
# Pass 0: screen

def screen(meta, output_assays, region_list, context):
    """{(cell, group, category): [candidate dict]} from the ENCODE tracks of every cell type (bigWig reads only),
    on the input window of each region (context bp centred on it, as the dataset does)."""
    import pyBigWig
    from data_utils import SingleCellFiberDataset
    cells = list(meta["cell_types"])
    bulk_bases = meta.get("bulk_base_paths", {})
    bws = {c: [None if meta["cell_types"][c]["bulk"].get(a) is None
               else pyBigWig.open(os.path.join(bulk_bases.get(a, ""), meta["cell_types"][c]["bulk"][a]))
               for a in output_assays] for c in cells}
    encode = {c: meta["cell_types"][c]["encode_dir"] for c in cells}
    found = {}
    for n, region in enumerate(region_list):
        chrom = region[0]
        start = (region[1] + region[2]) // 2 - context // 2
        end = start + context
        if context % BIN:
            raise ValueError(f"context {context} is not a multiple of BIN={BIN}")
        nb = context // BIN
        fc = {c: np.stack([np.full(nb, np.nan) if bw is None else
                           SingleCellFiberDataset._read_track(bw, chrom, start, end).reshape(nb, BIN).mean(1)
                           for bw in bws[c]]) for c in cells}                       # (K, nb) mean FC per bin
        for c in cells:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base = np.nanmean(np.stack([fc[o] for o in DEV_CELLS if encode[o] != encode[c]]), 0)
            cell_a, base_a = np.arcsinh(fc[c]), np.arcsinh(base)
            for k, a in enumerate(output_assays):
                if np.isnan(cell_a[k]).any() or np.isnan(base_a[k]).any():
                    continue
                counting = np.maximum(cell_a[k], base_a[k]) >= MIN_SIGNAL
                if not counting.any():
                    continue
                d = np.where(counting, cell_a[k] - base_a[k], 0.0)
                b = int(np.argmax(np.abs(d)))
                if abs(d[b]) >= DIFF_THRESHOLD:
                    cat, score = ("gain" if d[b] > 0 else "loss"), abs(float(d[b]))
                elif abs(d[b]) < AGREE_MAX_D and cell_a[k].max() >= MIN_SIGNAL:
                    b = int(np.argmax(cell_a[k]))
                    cat, score = "agree", float(cell_a[k][b])
                else:
                    continue
                partner = PARTNER[a]
                if np.isnan(fc[c][output_assays.index(partner)]).any():
                    partner = PARTNER_FALLBACK[a]
                found.setdefault((c, group_of(a), cat), []).append(
                    {"cell": c, "region": tuple(region), "flagged": a, "partner": partner, "bin": b,
                     "d_screen": float(d[b]), "screen_score": score, "category": cat, "group": group_of(a)})
        if (n + 1) % 1000 == 0:
            print(f"  screened {n + 1} / {len(region_list)} tiles", flush=True)
    return found


def pick_candidates(found):
    """Per (cell, group, category): CANDIDATES tiles by screen score, round robin over the assays, one per tile."""
    out = {}
    for key, cands in found.items():
        by_assay = {}
        for c in sorted(cands, key=lambda c: -c["screen_score"]):
            by_assay.setdefault(c["flagged"], []).append(c)
        queues, chosen, seen = list(by_assay.values()), [], set()
        while queues and len(chosen) < CANDIDATES[key[2]]:
            for q in list(queues):
                while q and q[0]["region"] in seen:
                    q.pop(0)
                if not q:
                    queues.remove(q)
                    continue
                c = q.pop(0)
                seen.add(c["region"])
                chosen.append(c)
                if len(chosen) == CANDIDATES[key[2]]:
                    break
        out[key] = chosen
    return out


#--------------------------------------------------------------------------------------------------
# Model passes

def run_model(model, cell, regions, mp, baseline_meta, cfg, device):
    """Yields one dict per region that has fibers: the model's bulk and per-fiber predictions with the inputs."""
    import torch
    from data_utils import make_fiber_dataset, loader_context
    from utils import resolve_config_with_metapaths, unpack_batch
    output_assays = model.init_args["output_assays"]
    meta = resolve_config_with_metapaths({"cell_types": {cell: None}}, mp, output_assays)
    ds = make_fiber_dataset("single", mode="eval", metadata=meta, fibers_per_entry=cfg["fibers_per_entry"],
                            context_length=cfg["context_length"], windows=WINDOWS,
                            input_flags=model.init_args["input_flags"], dna_type=model.init_args["dna_type"],
                            output_assays=output_assays, seed=SEED, baseline_cells=baseline_meta)
    ds.region_list = sorted(set(regions))
    by_window = {ds.window_for_region(r): r for r in ds.region_list}
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=NUM_WORKERS,
                                         multiprocessing_context=loader_context(NUM_WORKERS))
    with torch.no_grad():
        for batch in loader:
            window = (batch["locus"][0][0], int(batch["locus"][1][0]), int(batch["locus"][2][0]))
            fibers_t, target_t, kwargs = unpack_batch(batch, device)
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                    enabled=device == "cuda"):
                bulk, per_fiber = model(fibers_t, **kwargs)
            yield {"region": by_window[window], "window": window, "n_fibers": int(batch["n_fibers"][0]),
                   "target": target_t[0].float().cpu().numpy(),                    # (K, L)
                   "baseline": batch["baseline_bulk"][0].numpy(),                   # (K, L)
                   "bulk": bulk[0].float().cpu().numpy(),                           # (K, L)
                   "per_fiber": per_fiber[0].float().cpu().numpy(),                 # (K, L, N)
                   "fibers_in": fibers_t[0].float().cpu().numpy(),                  # (5, L, N)
                   "mask": batch["fiber_mask"][0].numpy().astype(bool),             # (L, N)
                   "coverage": batch["fiber_coverage"][0].numpy()}                  # (L,)


def _pearson(a, b):
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mse(a, b):
    return float(np.mean((a - b) ** 2)) if len(a) else float("nan")


def _bin_value(x, sl):
    """asinh of the mean FC over a bin (tracks are asinh FC)."""
    return float(np.arcsinh(np.sinh(x[sl].astype(np.float64)).mean()))


def fiber_bin_means(out, k, b):
    """Per fiber: mean prediction of assay k over the bin where the fiber covers it; NaN if it covers < half."""
    sl = slice(b * BIN, (b + 1) * BIN)
    m = out["mask"][sl]
    covered = m.sum(0)
    means = (out["per_fiber"][k][sl] * m).sum(0) / np.maximum(covered, 1)
    return np.where(covered >= BIN // 2, means, np.nan)


def score_candidate(cand, out, output_assays):
    k, kp = output_assays.index(cand["flagged"]), output_assays.index(cand["partner"])
    w = out["coverage"] >= MIN_COVERAGE
    sl = slice(cand["bin"] * BIN, (cand["bin"] + 1) * BIN)
    t_b, p_b, b_b = (_bin_value(out[key][k], sl) for key in ("target", "bulk", "baseline"))
    frac = (p_b - b_b) / (t_b - b_b) if abs(t_b - b_b) > 1e-6 else float("nan")
    r_m, r_b = _pearson(out["bulk"][k][w], out["target"][k][w]), _pearson(out["baseline"][k][w], out["target"][k][w])
    fm = fiber_bin_means(out, k, cand["bin"])
    fm = fm[~np.isnan(fm)]
    row = dict(cand, n_fibers=out["n_fibers"], target_bin=t_b, model_bin=p_b, baseline_bin=b_b, frac_recovered=frac,
               r_model=r_m, r_baseline=r_b,
               r_model_partner=_pearson(out["bulk"][kp][w], out["target"][kp][w]),
               r_baseline_partner=_pearson(out["baseline"][kp][w], out["target"][kp][w]),
               mse_model=_mse(out["bulk"][k][w], out["target"][k][w]),
               mse_baseline=_mse(out["baseline"][k][w], out["target"][k][w]),
               mse_model_partner=_mse(out["bulk"][kp][w], out["target"][kp][w]),
               mse_baseline_partner=_mse(out["baseline"][kp][w], out["target"][kp][w]),
               fiber_spread=float(fm.std()) if len(fm) >= 3 else float("nan"))
    r0 = 0.0 if np.isnan(r_m) else r_m
    if cand["category"] == "agree":
        row["score"] = r0 - abs(p_b - t_b)
    else:   # how much of the departure from the average the model recovers (capped at all of it), then shape
        row["score"] = (0.0 if np.isnan(frac) else min(max(frac, 0.0), 1.0)) + 0.5 * r0
    return row


#--------------------------------------------------------------------------------------------------
# Plotting

def _render_single_fiber_features(ax, fiber_inp):
    """All five input channels of one fiber on one Axes, one row per channel. fiber_inp: (5, L)."""
    for k, (name, color) in enumerate(zip(FEATURE_NAMES, FEATURE_COLORS)):
        feat = fiber_inp[k]
        if name in ("m6a", "cpg"):
            off = np.where(feat < -0.5)[0]
            if len(off):
                ax.scatter(off, np.full_like(off, -k), marker='|', color=UNMETHYLATED_COLOR, s=12, alpha=0.35,
                           linewidths=0.6)
            on = np.where(feat > 0.5)[0]
            if len(on):
                ax.scatter(on, np.full_like(on, -k), marker='|', color=color, s=20, alpha=0.85, linewidths=1.0)
        else:
            diff = np.diff((feat > 0.5).astype(float), prepend=0.0, append=0.0)
            for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
                ax.axhspan(-k - 0.35, -k + 0.35, xmin=s / len(feat), xmax=e / len(feat), color=color, alpha=0.5,
                           lw=0)
    ax.set_ylim(-len(FEATURE_NAMES) + 0.5, 0.5)
    ax.set_yticks([])


def choose_fibers(out, k, b):
    """Fibers sorted by their prediction in the flagged bin (covering fibers first, the rest by their own mean),
    and NUM_FIBERS examples spread over that ranking (top to bottom), preferring fibers spanning >= MIN_SPAN."""
    L = out["mask"].shape[0]
    present = np.where(out["mask"].sum(0) > 0)[0]
    fm = fiber_bin_means(out, k, b)
    own = (out["per_fiber"][k] * out["mask"]).sum(0) / np.maximum(out["mask"].sum(0), 1)
    covering = sorted([f for f in present if not np.isnan(fm[f])], key=lambda f: -fm[f])
    order = covering + sorted([f for f in present if np.isnan(fm[f])], key=lambda f: -own[f])
    span = out["mask"].sum(0) / L
    pool = [f for f in covering if span[f] >= MIN_SPAN]
    if len(pool) < NUM_FIBERS:
        pool = covering
    idx = np.unique(np.linspace(0, len(pool) - 1, min(NUM_FIBERS, len(pool))).round().astype(int))
    return order, [pool[i] for i in idx], len(covering)


def plot_locus(out, row, output_assays, split):
    k_flag = output_assays.index(row["flagged"])
    assays = [row["flagged"], row["partner"]]
    order, picks, n_cov = choose_fibers(out, k_flag, row["bin"])
    if not picks:
        return None
    rank = {f: i for i, f in enumerate(order)}
    L = out["mask"].shape[0]
    x = np.arange(L)
    b0, b1 = row["bin"] * BIN, (row["bin"] + 1) * BIN
    n = len(picks)
    chrom, s, e = out["window"]

    fig = plt.figure(figsize=(COL_WIDTH * 2, 7.5 + 2.6 * n))
    top = 1 - 2.3 / (7.5 + 2.6 * n)
    fig.subplots_adjust(top=top, bottom=0.04, left=0.06, right=0.95)
    cat_text = {"gain": "higher than the cross-cell average", "loss": "lower than the cross-cell average",
                "agree": "in line with the cross-cell average"}[row["category"]]
    fig.suptitle(f"{cell_label(row['cell'])} ({'dev' if split == 'dev' else 'held out'}): {chrom}:{s:,}-{e:,}   "
                 f"{ASSAY_LABELS[row['flagged']]} {cat_text}", fontsize=FS_SUPTITLE, fontweight="bold", y=0.997)
    gs = gridspec.GridSpec(2 + n, 2, figure=fig, hspace=0.55, wspace=0.16, height_ratios=[1.5, 2.0] + [1.1] * n)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.93")

    for col, assay in enumerate(assays):
        k = output_assays.index(assay)
        pf, mask = out["per_fiber"][k], out["mask"]
        ax_b = fig.add_subplot(gs[0, col])
        ax_b.axvspan(b0, b1, color=BIN_COLOR, alpha=0.25, lw=0)
        ax_b.fill_between(x, 0, out["target"][k], color=TARGET_COLOR, alpha=0.45, lw=0)
        ax_b.plot(x, out["baseline"][k], color="black", ls="--", lw=1.8)
        ax_b.plot(x, out["bulk"][k], color=MODEL_COLOR, lw=2.8)
        ax_b.set_xlim(0, L)
        ax_b.set_ylim(bottom=0)
        ax_b.set_ylabel("asinh FC", fontweight="bold")
        title = ASSAY_LABELS[assay]
        if col == 0:
            title += f"  (flagged: d = {row['d_screen']:+.2f})"
        m_m = row["mse_model"] if col == 0 else row["mse_model_partner"]
        m_b = row["mse_baseline"] if col == 0 else row["mse_baseline_partner"]
        ax_b.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left", pad=8)
        ax_b.text(1.0, 1.02, f"tile MSE: model {m_m:.3f}, baseline {m_b:.3f}", transform=ax_b.transAxes,
                  ha="right", va="bottom", fontsize=FS_TICK, color="0.3")
        ax_b.tick_params(labelbottom=False)

        ax_h = fig.add_subplot(gs[1, col], sharex=ax_b)
        heat = np.where(mask, pf, np.nan)[:, order].T                               # (fibers, L)
        vmax = np.nanpercentile(heat, 99) if np.isfinite(heat).any() else 1.0
        im = ax_h.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=max(vmax, 0.5), interpolation="nearest",
                         extent=(0, L, len(order), 0))
        ax_h.axvline(b0, color=BIN_COLOR, lw=1.2)
        ax_h.axvline(b1, color=BIN_COLOR, lw=1.2)
        ax_h.set_yticks([rank[f] + 0.5 for f in picks])
        ax_h.set_yticklabels([f"#{f}" for f in picks], fontsize=FS_TICK - 2)
        ax_h.set_ylabel(f"{len(order)} fibers\n(sorted by flagged bin)", fontweight="bold")
        cax = ax_h.inset_axes([1.01, 0, 0.015, 1])
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=FS_TICK - 2)
        ax_h.tick_params(labelbottom=False)

        ymax = max(1.0, float(np.nanmax(out["baseline"][k])), max(float(np.max(pf[:, f])) for f in picks)) * 1.08
        for i, f in enumerate(picks):
            sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2 + i, col], hspace=0.0,
                                                   height_ratios=[1, 1.3])
            ax_in = fig.add_subplot(sub[0, 0], sharex=ax_b)
            _render_single_fiber_features(ax_in, out["fibers_in"][:, :, f])
            ax_in.set_title(f"Fiber #{f}  (rank {rank[f] + 1} of {n_cov} covering the flagged bin)",
                            fontsize=FS_FIBER_TITLE, fontweight="bold", loc="left", pad=3)
            ax_in.tick_params(labelbottom=False)
            ax_p = fig.add_subplot(sub[1, 0], sharex=ax_b)
            ax_p.axvspan(b0, b1, color=BIN_COLOR, alpha=0.2, lw=0)
            y = np.where(mask[:, f], pf[:, f], np.nan)
            ax_p.plot(x, out["baseline"][k], color="black", ls="--", lw=1.4)   # what a bulk baseline gives every fiber
            ax_p.fill_between(x, 0, y, color=MODEL_COLOR, alpha=0.2, lw=0)
            ax_p.plot(x, y, color=MODEL_COLOR, lw=1.8)
            ax_p.set_ylim(0, ymax)
            ax_p.set_ylabel("Imputed", fontsize=FS_LABEL - 3, fontweight="bold")
            if i < n - 1:
                ax_p.tick_params(labelbottom=False)
            else:
                ax_p.set_xlabel("Position in window (bp)", fontweight="bold")

    handles = [Patch(color=TARGET_COLOR, alpha=0.45, label="ENCODE (this cell type)"),
               Line2D([], [], color=MODEL_COLOR, lw=2.8, label="Model: average over fibers"),
               Line2D([], [], color="black", ls="--", lw=1.8, label="Baseline: cross-cell average (also on each fiber)"),
               Line2D([], [], color=MODEL_COLOR, lw=1.8, label="Model: this fiber"),
               Patch(color=BIN_COLOR, alpha=0.4, label="Flagged 1 kb bin")]
    handles += [Patch(color=c, alpha=0.7, label=l) for c, l in zip(FEATURE_COLORS, FEATURE_LABELS)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1 - 0.75 / (7.5 + 2.6 * n)))
    return fig


#--------------------------------------------------------------------------------------------------

def main():
    import torch
    from models import BaseModel
    from utils import load_metapaths, resolve_config_with_metapaths
    from evaluator import resolve_baseline_cells

    os.chdir(REPO_ROOT)
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{now}_{RUN_NAME}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_cfg = BaseModel.load_model(os.path.join(REPO_ROOT, CHECKPOINT), map_location=device)
    model = model.to(device).eval()
    output_assays = model.init_args["output_assays"]
    cfg = {"context_length": int(ckpt_cfg.get("context_length", 5000)),
           "fibers_per_entry": int(ckpt_cfg.get("fibers_per_entry", 100))}
    print(f"[model] {CHECKPOINT}\n  inputs {model.init_args['input_flags']}, assays {output_assays}, aggregation "
          f"{model.fiber_aggregation}, context {cfg['context_length']}, fibers {cfg['fibers_per_entry']}", flush=True)

    mp = load_metapaths(os.path.join(REPO_ROOT, "configs", "metapaths.yaml"))
    cells = DEV_CELLS + HELD_OUT
    meta_all = resolve_config_with_metapaths({"cell_types": {c: None for c in cells}}, mp, output_assays)
    baseline_meta = resolve_baseline_cells(DEV_CELLS, mp, output_assays)
    if REPLOT_FROM:
        return replot(model, output_assays, mp, baseline_meta, cfg, device, out_dir)
    from regions import Regions
    region_list = Regions(meta_all["regions_root"]).window_set(WINDOWS)

    # Pass 0
    print(f"\n[screen] {len(region_list)} {WINDOWS} tiles x {len(cells)} cell types", flush=True)
    found = screen(meta_all, output_assays, region_list, cfg["context_length"])
    chosen = pick_candidates(found)
    for key in sorted(chosen):
        assays = {}
        for c in chosen[key]:
            assays[c["flagged"]] = assays.get(c["flagged"], 0) + 1
        print(f"  {cell_label(key[0]):8s} {key[1]:5s} {key[2]:5s}: {len(found[key]):5d} found, "
              f"{len(chosen[key])} kept {assays}", flush=True)

    # Per cell type: rank every candidate (pass 1), select, plot (pass 2); the table is rewritten after each cell
    rows, rng = [], random.Random(SEED)
    for cell in cells:
        split = "dev" if cell in DEV_CELLS else "test"
        by_region = {}
        for key, cs in chosen.items():
            if key[0] == cell:
                for c in cs:
                    by_region.setdefault(c["region"], []).append(c)
        print(f"\n[rank] {cell}: {len(by_region)} tiles", flush=True)
        cell_rows = []
        for out in run_model(model, cell, list(by_region), mp, baseline_meta, cfg, device):
            for c in by_region[out["region"]]:
                cell_rows.append(score_candidate(c, out, output_assays))
        rows += cell_rows

        jobs = {}                                    # region -> [(row, selected, NN)]
        for key in sorted({(r["group"], r["category"]) for r in cell_rows}):
            ok = sorted([r for r in cell_rows if (r["group"], r["category"]) == key and r["n_fibers"] >= MIN_FIBERS],
                        key=lambda r: -r["score"])
            best, rest = ok[:PLOT_BEST[key[1]]], ok[PLOT_BEST[key[1]]:]
            for sel, picked in (("best", best), ("random", rng.sample(rest, min(PLOT_RANDOM[key[1]], len(rest))))):
                for j, r in enumerate(picked):
                    r["selected"] = sel
                    jobs.setdefault(r["region"], []).append((r, sel, j))

        print(f"[plot] {cell}: {sum(len(v) for v in jobs.values())} figures on {len(jobs)} tiles", flush=True)
        for out in run_model(model, cell, list(jobs), mp, baseline_meta, cfg, device):
            for r, sel, j in jobs[out["region"]]:
                fig = plot_locus(out, r, output_assays, split)
                if fig is None:
                    print(f"  {out['window']} {r['flagged']}: no fiber covers the flagged bin; skipped", flush=True)
                    continue
                sub = os.path.join(split, cell_label(cell), r["group"])
                os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
                chrom, s, e = out["window"]
                name = f"{sel}_{r['category']}_{j:02d}_{chrom}_{s}-{e}_{r['flagged']}-{r['partner']}.png"
                fig.savefig(os.path.join(out_dir, sub, name), dpi=DPI)
                plt.close(fig)
                r["figure"] = os.path.join(sub, name)
                print(f"  saved {r['figure']}  (score {r['score']:.2f}, frac {r['frac_recovered']:.2f}, "
                      f"r {r['r_model']:.2f} vs {r['r_baseline']:.2f})", flush=True)
        write_table(rows, os.path.join(out_dir, "ranking.tsv"))

    print(f"\nDone. {len(rows)} candidates ranked, figures and ranking.tsv in {out_dir}", flush=True)


def replot(model, output_assays, mp, baseline_meta, cfg, device, out_dir):
    """Re-plots the figures listed in REPLOT_FROM/ranking.tsv (same loci, names and folders), re-scoring each one."""
    by_label = {cell_label(c): c for c in DEV_CELLS + HELD_OUT}
    with open(os.path.join(REPO_ROOT, REPLOT_FROM, "ranking.tsv")) as f:
        old = [r for r in csv.DictReader(f, delimiter="\t") if r["figure"]]
    print(f"[replot] {len(old)} figures from {REPLOT_FROM}", flush=True)
    rows = []
    for cell in DEV_CELLS + HELD_OUT:
        jobs = {}
        for r in old:
            if by_label[r["cell"]] == cell:
                region = (r["chrom"], int(r["start"]), int(r["end"]))
                cand = dict(cell=cell, group=r["group"], category=r["category"], flagged=r["flagged"],
                            partner=r["partner"], region=region, bin=int(r["bin"]), d_screen=float(r["d_screen"]))
                jobs.setdefault(region, []).append((cand, r["selected"], r["figure"]))
        print(f"\n[plot] {cell}: {sum(len(v) for v in jobs.values())} figures on {len(jobs)} tiles", flush=True)
        for out in run_model(model, cell, list(jobs), mp, baseline_meta, cfg, device):
            for cand, sel, figure in jobs[out["region"]]:
                r = score_candidate(cand, out, output_assays)
                r["selected"] = sel
                fig = plot_locus(out, r, output_assays, "dev" if cell in DEV_CELLS else "test")
                if fig is None:
                    print(f"  {out['window']} {r['flagged']}: no fiber covers the flagged bin; skipped", flush=True)
                    continue
                os.makedirs(os.path.join(out_dir, os.path.dirname(figure)), exist_ok=True)
                fig.savefig(os.path.join(out_dir, figure), dpi=DPI)
                plt.close(fig)
                r["figure"] = figure
                rows.append(r)
                print(f"  saved {figure}  (MSE {r['mse_model']:.3f} vs {r['mse_baseline']:.3f})", flush=True)
        write_table(rows, os.path.join(out_dir, "ranking.tsv"))
    print(f"\nDone. {len(rows)} figures and ranking.tsv in {out_dir}", flush=True)


def write_table(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["cell"], r["group"], r["category"], -r["score"])):
            chrom, start, end = r["region"]
            row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            row.update(split="dev" if r["cell"] in DEV_CELLS else "test", cell=cell_label(r["cell"]), chrom=chrom,
                       start=start, end=end)
            w.writerow(row)

if __name__ == "__main__":
    main()
