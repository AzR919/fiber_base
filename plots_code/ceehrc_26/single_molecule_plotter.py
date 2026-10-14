"""
One-off script for the ceehrc_26 poster/talk.

Self-contained, large-font versions of the two plots we need. The plotting
code below is copied from vis_utils.py and reworked for poster use
(vis_utils.py itself is untouched):
  - the all-assay dashboard             (copy of plot_evaluator_record)
  - the single-molecule fiber stack     (copy of plot_single_column_fiber_stack),
    split into figures of ASSAYS_PER_FIGURE assay columns each so the images
    stay a manageable size.

Each figure states the cell type and locus exactly once (figure title); every
assay chart is titled with just its assay name, and legends are "GT" / "P".

For each index in SELECTED_INDICES (the <index> prefix of the files already
reviewed under ignore/ceehrc_26/<model_label>/), for each of the 3 models, this
re-runs inference for that locus and saves into plots/ceehrc_26/<label>/:
  <index>_<chrom>_<start>-<end>_dashboard_large.png
  <index>_<chrom>_<start>-<end>_single_molecule_<assayA>-<assayB>.png   (one per assay pair)
Existing files with other names in those directories are never touched.

Single-molecule figures show only fibers with summed m6a > M6A_THRESHOLD (first
NUM_FIBERS of them); the same fibers appear in every figure for a locus.

Edit the constants in the EDIT ME block (no CLI by design).
Usage: python plots_code/ceehrc_26/single_molecule_plotter.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_ROOT = os.path.join(REPO_ROOT, "plots", "ceehrc_26")

# ---- EDIT ME ----
# Indices to plot — the <index> prefix of files under ignore/ceehrc_26/<label>/.
SELECTED_INDICES = [147, 868]
NUM_FIBERS = 5           # fibers per single-molecule column
M6A_THRESHOLD = 200      # keep fibers with summed m6a strictly greater than this
ASSAYS_PER_FIGURE = 2    # assay columns per single-molecule figure
# -----------------

# ---- Font sizes (vis_utils defaults are ~10pt, unreadable once these big figures are scaled down) ----
FS_SUPTITLE = 38         # "<cell> — <locus>", once per figure
FS_TITLE = 34            # per-chart title (assay name)
FS_LABEL = 28
FS_FEATURE_LABEL = 30    # m6a / cpg / msp / nuc / fire_msp row labels
FS_TICK = 24
FS_LEGEND = 30
FS_CBAR = 24
FS_FIBER_TITLE = 26      # "Fiber #N"
FS_FIBER_YLABEL = 20     # per-fiber imputed-signal row y label
DPI = 150

# ---- Figure geometry ----
DASH_HEIGHT = 26         # dashboard height (inches); width is 10 per column
DASH_SPACER_RATIO = 0.55 # empty row between the two assays of a dashboard column (holds x label + colorbar + next title)
SM_COL_WIDTH = 16        # single-molecule width per assay column (inches)
SM_HEIGHT = 24           # single-molecule figure height (inches)
SM_BULK_RATIO = 1.0      # height of the bulk row relative to one fiber row (1.0 = same size)

plt.rcParams.update({
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.labelsize": FS_LABEL,
})

FEATURE_NAMES = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
FEATURE_COLORS = ["black", "purple", "blue", "green", "red"]
UNMETHYLATED_COLOR = "lightslategrey"


#--------------------------------------------------------------------------------------------------
# Plotting code copied from vis_utils.py (fonts enlarged, repeated text removed, spacing reworked)

def _render_input_channels(fig, gs_col, inp, input_flags):
    """
    Renders discrete and continuous input channels stacked across all N fibers.
    inp: (C, L, N) numpy array.
    """
    num_fibers = inp.shape[-1]
    active_indices = [j for j, flag in enumerate(input_flags) if flag]
    num_active = len(active_indices)

    k = 0
    input_axes = []
    for j in active_indices:
        ax = fig.add_subplot(gs_col[k, 0])
        input_axes.append(ax)
        is_single_bit = FEATURE_NAMES[j] in ["m6a", "cpg"]

        for i in range(num_fibers):
            fiber_feat = inp[k, :, i]  # (L,)

            if is_single_bit:
                unmeth_indices = np.where(fiber_feat < -0.5)[0]
                if len(unmeth_indices) > 0:
                    ax.scatter(
                        unmeth_indices,
                        np.full_like(unmeth_indices, -i),
                        marker='|',
                        color=UNMETHYLATED_COLOR,
                        s=15,
                        alpha=0.35,
                        linewidths=0.6,
                    )
                meth_indices = np.where(fiber_feat > 0.5)[0]
                if len(meth_indices) > 0:
                    ax.scatter(
                        meth_indices,
                        np.full_like(meth_indices, -i),
                        marker='|',
                        color=FEATURE_COLORS[j],
                        s=25,
                        alpha=0.85,
                        linewidths=1.0,
                    )
            else:
                masked = (fiber_feat > 0.5).astype(float)
                diff = np.diff(masked, prepend=0.0, append=0.0)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                for s, e in zip(starts, ends):
                    if e > s:
                        ax.axhspan(
                            -i - 0.35,
                            -i + 0.35,
                            xmin=s / len(fiber_feat),
                            xmax=e / len(fiber_feat),
                            color=FEATURE_COLORS[j],
                            alpha=0.5,
                            lw=0
                        )

        ax.set_ylabel(FEATURE_NAMES[j], fontsize=FS_FEATURE_LABEL, fontweight='bold')
        ax.set_ylim(-num_fibers - 0.5, 0.5)
        ax.set_xlim(0, inp.shape[1])

        if k < num_active - 1:
            ax.set_xticklabels([])
        k += 1

    if input_axes:
        input_axes[-1].set_xlabel("Genomic Position (bp)", fontsize=FS_LABEL)

    return input_axes


def _render_single_fiber_features(ax, fiber_inp, input_flags):
    """
    Renders discrete/continuous input channels for a single fiber on one Axes.
    fiber_inp: (C, L) numpy array.
    """
    active_indices = [j for j, flag in enumerate(input_flags) if flag]

    for k, j in enumerate(active_indices):
        feat_name = FEATURE_NAMES[j]
        color = FEATURE_COLORS[j]
        fiber_feat = fiber_inp[k]  # (L,)

        if feat_name in ["m6a", "cpg"]:
            unmeth_indices = np.where(fiber_feat < -0.5)[0]
            if len(unmeth_indices) > 0:
                ax.scatter(
                    unmeth_indices,
                    np.full_like(unmeth_indices, -k),
                    marker='|',
                    color=UNMETHYLATED_COLOR,
                    s=15,
                    alpha=0.35,
                    linewidths=0.6
                )
            meth_indices = np.where(fiber_feat > 0.5)[0]
            if len(meth_indices) > 0:
                ax.scatter(
                    meth_indices,
                    np.full_like(meth_indices, -k),
                    marker='|',
                    color=color,
                    s=25,
                    alpha=0.85,
                    linewidths=1.0
                )
        else:
            masked = (fiber_feat > 0.5).astype(float)
            diff = np.diff(masked, prepend=0.0, append=0.0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                if e > s:
                    ax.axhspan(
                        -k - 0.35,
                        -k + 0.35,
                        xmin=s / len(fiber_feat),
                        xmax=e / len(fiber_feat),
                        color=color,
                        alpha=0.5,
                        lw=0
                    )

    ax.set_ylim(-len(active_indices) + 0.5, 0.5)
    ax.set_yticks([])


def _render_bulk_comparison(ax, target, pred_bulk, assay_name):
    """Target (GT) vs predicted (P) bulk signal, titled with just the assay name.
    target, pred_bulk: (L,) numpy arrays."""
    ax.plot(target, color='dimgray', lw=2.5, label='GT')
    ax.plot(pred_bulk, color='darkorange', lw=2.5, label='P', alpha=0.85)
    ax.set_ylabel("Signal Intensity", fontsize=FS_LABEL)
    ax.legend(loc='upper right', frameon=False, fontsize=FS_LEGEND, handlelength=1.2)
    ax.set_title(assay_name, fontsize=FS_TITLE, fontweight='bold', pad=10)
    ax.set_xticklabels([])


def _render_fiber_heatmap(ax, out_fibers):
    """Predicted fiber heatmap. out_fibers: (L, N) numpy array.
    The colorbar is an inset placed below the axes, sized in inches so it doesn't
    shrink the heatmap or collide with the x label / next panel."""
    pred_matrix = out_fibers.T
    img = ax.imshow(pred_matrix, aspect='auto', cmap='magma',
                    interpolation='nearest', origin='upper',
                    extent=[0, pred_matrix.shape[1], -pred_matrix.shape[0], 0])

    h_in = ax.get_position().height * ax.figure.get_figheight()
    gap_in, thick_in = 0.75, 0.22   # distance below the axes (clears the x label), colorbar thickness
    cax = ax.inset_axes([0.25, -(gap_in + thick_in) / h_in, 0.5, thick_in / h_in])
    cbar = ax.figure.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('Accessibility Probability', fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR)

    ax.set_ylabel("Fibers (Imputed)", fontsize=FS_LABEL)
    ax.set_xlabel("Genomic Position (bp)", fontsize=FS_LABEL)
    return img


def _render_assay_pair(fig, col_gs, row_start, out_k, out_fibers_k, tar_k, assay_name):
    """
    Renders one assay in 2 consecutive rows of col_gs:
      row_start   -> bulk comparison (GT vs P)
      row_start+1 -> fiber heatmap for that assay
    out_k, tar_k: (L,)  out_fibers_k: (L, N)
    """
    ax_sig = fig.add_subplot(col_gs[row_start, 0])
    _render_bulk_comparison(ax_sig, tar_k, out_k, assay_name)
    ax_heat = fig.add_subplot(col_gs[row_start + 1, 0], sharex=ax_sig)
    _render_fiber_heatmap(ax_heat, out_fibers_k)


def plot_dashboard(fiber_features, processed_fibers, pred_bulk, target_bulk,
                   locus, cell_type, input_flags, output_assays):
    """
    All-assay dashboard: input features (col 0), then assays in the remaining columns,
    two per column (bulk comparison + fiber heatmap each). Cell type and locus appear
    once, in the figure title.
    fiber_features: (C, L, N), processed_fibers: (K, L, N), pred_bulk/target_bulk: (K, L).
    locus: (chrom, start, end). Returns matplotlib Figure.
    """
    K = len(output_assays)
    chr_name, start, end = locus

    num_data_cols = 1 + max(1, (K + 1) // 2)
    fig = plt.figure(figsize=(10 * num_data_cols, DASH_HEIGHT))
    # Adjust before creating axes so axes positions (used to size the colorbar insets) are final.
    fig.subplots_adjust(top=0.92, bottom=0.075, left=0.065, right=0.975)
    fig.suptitle(f"{cell_type} — {chr_name}:{start}-{end}", fontsize=FS_SUPTITLE, fontweight='bold', y=0.985)

    gs = gridspec.GridSpec(5, num_data_cols, figure=fig,
                           width_ratios=[1] * num_data_cols, wspace=0.3, hspace=0.35)

    input_axes = _render_input_channels(fig, gs, fiber_features, input_flags)
    if input_axes:
        input_axes[0].set_title("Input Fiber Stack", fontsize=FS_TITLE, fontweight='bold', pad=10)

    for pair_idx, k_start in enumerate(range(0, K, 2)):
        # rows: [assay A signal, assay A heatmap, spacer, assay B signal, assay B heatmap]
        col_gs = gridspec.GridSpecFromSubplotSpec(
            5, 1, subplot_spec=gs[:, 1 + pair_idx],
            height_ratios=[1, 1, DASH_SPACER_RATIO, 1, 1], hspace=0.1)
        for local_k, k in enumerate(range(k_start, min(k_start + 2, K))):
            _render_assay_pair(fig, col_gs, local_k * 3,
                               pred_bulk[k], processed_fibers[k], target_bulk[k],
                               output_assays[k])

    return fig


def select_fibers(fiber_features):
    """fiber_features: (C, L, N); channel 0 is m6a (input_flags is all-active for these
    models, so no earlier feature is skipped). Returns the first NUM_FIBERS fiber indices
    whose summed m6a is strictly greater than M6A_THRESHOLD."""
    m6a_sums = (fiber_features[0] > 0.5).sum(axis=0)  # (N,)
    return np.where(m6a_sums > M6A_THRESHOLD)[0][:NUM_FIBERS].tolist()


def plot_single_molecule_stack(true_bulks, pred_bulks, fiber_seq_inp, processed_fibers_list,
                               input_flags, bulk_names, selected_indices, locus, cell_type):
    """
    One column per assay (titled with just the assay name); each column is [bulk comparison]
    then, per selected fiber, [fiber-seq input row, imputed signal row]. The same fibers
    appear in every column. Cell type and locus appear once, in the figure title.
    true_bulks/pred_bulks: lists of K (L,) arrays; processed_fibers_list: list of K (L, N)
    arrays; fiber_seq_inp: (C, L, N); bulk_names: list of K strings. Returns Figure.
    """
    K = len(bulk_names)
    n = len(selected_indices)
    seq_len = fiber_seq_inp.shape[1]
    chr_name, s, e = locus

    fig = plt.figure(figsize=(SM_COL_WIDTH * K, SM_HEIGHT))
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.075, right=0.985)
    fig.suptitle(f"{cell_type} — {chr_name}:{s}-{e}", fontsize=FS_SUPTITLE, fontweight='bold', y=0.985)

    gs = gridspec.GridSpec(1 + n, K, figure=fig, hspace=0.6, wspace=0.2,
                           height_ratios=[SM_BULK_RATIO] + [1] * n)
    x_coords = np.arange(seq_len)

    for col in range(K):
        proc_c = processed_fibers_list[col]

        ax_bulk_row = fig.add_subplot(gs[0, col])
        ax_bulk_row.plot(true_bulks[col], color='dimgray', lw=2.5, label='GT')
        ax_bulk_row.plot(pred_bulks[col], color='darkorange', lw=2.5, label='P', alpha=0.85)
        ax_bulk_row.set_ylabel("Signal", fontsize=FS_LABEL, fontweight='bold')
        ax_bulk_row.legend(loc='upper right', frameon=False, fontsize=FS_LEGEND, handlelength=1.2)
        ax_bulk_row.set_xlim(0, seq_len)
        ax_bulk_row.set_title(bulk_names[col], fontsize=FS_TITLE, fontweight='bold', pad=10)

        global_max_y = max(1.0, float(proc_c[:, selected_indices].max()))

        for i, fiber_idx in enumerate(selected_indices):
            fiber_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[i + 1, col], hspace=0.0, height_ratios=[1, 1.2]
            )

            ax_inp = fig.add_subplot(fiber_gs[0, 0], sharex=ax_bulk_row)
            _render_single_fiber_features(ax_inp, fiber_seq_inp[:, :, fiber_idx], input_flags)
            ax_inp.set_title(f"Fiber #{fiber_idx}", fontsize=FS_FIBER_TITLE, fontweight='bold',
                             loc='left', pad=4)
            ax_inp.set_xticklabels([])

            ax_proc = fig.add_subplot(fiber_gs[1, 0], sharex=ax_bulk_row)
            proc_signal = proc_c[:, fiber_idx]
            ax_proc.plot(x_coords, proc_signal, color='firebrick', lw=2.0)
            ax_proc.fill_between(x_coords, 0, proc_signal, color='crimson', alpha=0.45)
            ax_proc.set_ylim(0, global_max_y * 1.05)
            ax_proc.set_ylabel("Imputed", fontsize=FS_FIBER_YLABEL, fontweight='bold')

            if i < n - 1:
                ax_proc.set_xticklabels([])
            else:
                ax_proc.set_xlabel("Genomic Position (bp)", fontsize=FS_LABEL, fontweight='bold')

    return fig


#--------------------------------------------------------------------------------------------------

def main():
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import plotter` works
    from plotter import DEVICE, DEVICE_TYPE, build_dataset, build_filtered_ccres, load_models
    from utils import unpack_batch

    print(f"Repo root: {REPO_ROOT}")
    print(f"Selected indices: {SELECTED_INDICES}")

    filtered_ccres = build_filtered_ccres()
    models = load_models()

    ref_model = next(iter(models.values()))
    output_assays = ref_model.init_args.get("output_assays", ["atac"])
    input_flags = ref_model.init_args["input_flags"]
    dna_type = ref_model.init_args["dna_type"]
    K = len(output_assays)

    dataset = build_dataset(output_assays, input_flags, dna_type)
    dataset.ccre_list = filtered_ccres
    dataset.num_sample_ccres = -1

    loader = DataLoader(dataset, batch_size=1, drop_last=False)

    selected_set = set(SELECTED_INDICES)
    remaining = set(SELECTED_INDICES)
    max_needed = max(SELECTED_INDICES) if SELECTED_INDICES else -1

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > max_needed:
                break
            if i not in selected_set:
                continue
            remaining.discard(i)

            chrom0 = batch["locus"][0][0]
            start0 = int(batch["locus"][1][0])
            end0 = int(batch["locus"][2][0])
            locus = (chrom0, start0, end0)
            stem = f"{i:05d}_{chrom0}_{start0}-{end0}"
            print(f"\n=== index {i}: {chrom0}:{start0}-{end0} ===", flush=True)

            ct = batch.get("cell_type", ["HepG2_200U"])
            cell_type = ct[0] if isinstance(ct, (list, tuple)) else ct
            cell_label = cell_type.replace("_200U", "")   # display name, matches the bar chart

            fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, DEVICE)
            fiber_np = fiber_features[0].float().cpu().numpy()   # (C, L, N)
            target_np = target_bulk[0].float().cpu().numpy()     # (K, L)

            selected_fibers = select_fibers(fiber_np)
            print(f"  {len(selected_fibers)} fibers with m6a > {M6A_THRESHOLD}: {selected_fibers}", flush=True)

            for label, model in models.items():
                dst_dir = os.path.join(PLOTS_ROOT, label)
                os.makedirs(dst_dir, exist_ok=True)

                with torch.amp.autocast(DEVICE_TYPE, dtype=torch.float16):
                    pred_bulk, processed_fibers = model(fiber_features, **forward_kwargs)
                pred_np = pred_bulk[0].float().cpu().numpy()          # (K, L)
                proc_np = processed_fibers[0].float().cpu().numpy()   # (K, L, N)

                fig = plot_dashboard(fiber_np, proc_np, pred_np, target_np,
                                     locus, cell_label, input_flags, output_assays)
                out_name = f"{stem}_dashboard_large.png"
                fig.savefig(os.path.join(dst_dir, out_name), dpi=DPI)
                plt.close(fig)
                print(f"  [{label}] saved {out_name}", flush=True)

                if not selected_fibers:
                    print(f"  [{label}] [WARN] no fibers passed m6a > {M6A_THRESHOLD}; skipping single-molecule plots")
                    continue

                for a in range(0, K, ASSAYS_PER_FIGURE):
                    ks = list(range(a, min(a + ASSAYS_PER_FIGURE, K)))
                    names = [output_assays[k] for k in ks]
                    fig = plot_single_molecule_stack(
                        true_bulks=[target_np[k] for k in ks],
                        pred_bulks=[pred_np[k] for k in ks],
                        fiber_seq_inp=fiber_np,
                        processed_fibers_list=[proc_np[k] for k in ks],
                        input_flags=input_flags,
                        bulk_names=names,
                        selected_indices=selected_fibers,
                        locus=locus,
                        cell_type=cell_label,
                    )
                    out_name = f"{stem}_single_molecule_{'-'.join(names)}.png"
                    fig.savefig(os.path.join(dst_dir, out_name), dpi=DPI)
                    plt.close(fig)
                    print(f"  [{label}] saved {out_name}", flush=True)

    if remaining:
        print(f"\n[WARN] indices not reached (out of range of the filtered set): {sorted(remaining)}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
