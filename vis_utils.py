"""
Visualization utilities for fiber-seq evaluation dashboards.
Base functions accept numpy arrays; _t wrapper functions accept torch tensors.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import logomaker


#--------------------------------------------------------------------------------------------------
# Constants

FEATURE_NAMES = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
FEATURE_COLORS = ["black", "purple", "blue", "green", "red"]
UNMETHYLATED_COLOR = "lightslategrey"

# Dashboard font sizes
FS_SUPTITLE = 20
FS_TITLE = 15
FS_LABEL = 13
FS_TICK = 12
FS_LEGEND = 12


#--------------------------------------------------------------------------------------------------
# Private numpy sub-renderers

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
                        label="Unmethylated (-1)" if (i == 0 and k == 0) else ""
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
                        label="Methylated (+1)" if (i == 0 and k == 0) else ""
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

        ax.set_ylabel(FEATURE_NAMES[j], fontsize=FS_LABEL + 1, fontweight='bold')
        ax.set_ylim(-num_fibers - 0.5, 0.5)
        ax.set_xlim(0, inp.shape[1])
        ax.tick_params(labelsize=FS_TICK)

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


def _window_pearson(a, b):
    """Pearson r between two (L,) profiles; NaN if either is flat."""
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _fmt_r(r):
    return "n/a" if r is None or np.isnan(r) else f"{r:.2f}"


def _shade_unscored(ax, score_mask):
    """Light grey bands over positions excluded from loss/metrics (too few fibers)."""
    edges = np.diff(np.concatenate([[0], (~score_mask).astype(np.int8), [0]]))
    for i, (s, e) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        ax.axvspan(s, e, color='lightgrey', alpha=0.5, lw=0, label='Not scored (few fibers)' if i == 0 else None)


def _render_bulk_comparison(ax, target, pred_bulk, assay_name, mode, has_target=True,
                            avg_loss=None, avg_pearson=None, score_mask=None):
    """
    Target vs predicted bulk signal for one assay. target, pred_bulk: (L,) numpy arrays.
    Title line 1: this window's loss and Pearson r over the scored positions; line 2 (if given): the epoch/eval
    averages. has_target=False: the cell type has no bulk data for this assay, so only the prediction is drawn.
    score_mask: (L,) bool, positions used for loss/metrics (None = all); the rest is shaded.
    """
    if score_mask is not None and not score_mask.all():
        _shade_unscored(ax, score_mask)
    if has_target:
        ax.plot(target, color='dimgray', lw=1.5, label='Target')
    ax.plot(pred_bulk, color='darkorange', lw=1.5, label='Predicted', alpha=0.85)
    ax.set_ylabel("Signal (asinh FC)", fontsize=FS_LABEL)
    ax.legend(loc='upper right', frameon=False, fontsize=FS_LEGEND)
    ax.tick_params(labelsize=FS_TICK, labelbottom=False)

    keep = np.ones(len(pred_bulk), dtype=bool) if score_mask is None else score_mask
    if has_target and keep.any():
        loss = float(np.mean((target[keep] - pred_bulk[keep]) ** 2))
        title = f"{assay_name}  |  loss {loss:.3f}  |  r {_fmt_r(_window_pearson(target[keep], pred_bulk[keep]))}"
    elif has_target:
        title = f"{assay_name}  |  no position scored (too few fibers)"
    else:
        title = f"{assay_name}  |  no ENCODE data (prediction only)"
    if avg_loss is not None or avg_pearson is not None:
        parts = []
        if avg_loss is not None:
            parts.append(f"loss {avg_loss:.3f}")
        if avg_pearson is not None:
            parts.append(f"r {_fmt_r(avg_pearson)}")
        title += f"\n{mode} avg: " + "  |  ".join(parts)
    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')


def _render_fiber_heatmap(ax, out_fibers):
    """
    Per-fiber predicted signal heatmap. out_fibers: (L, N) numpy array.
    The colorbar is an inset placed below the axes (sized in inches, in the gap before the next
    panel) so it neither shrinks the heatmap nor collides with its x label.
    """
    pred_matrix = out_fibers.T
    img = ax.imshow(pred_matrix, aspect='auto', cmap='magma',
                    interpolation='nearest', origin='upper',
                    extent=[0, pred_matrix.shape[1], -pred_matrix.shape[0], 0])
    h_in = ax.get_position().height * ax.figure.get_figheight()
    gap_in, thick_in = 0.75, 0.15
    cax = ax.inset_axes([0.3, -(gap_in + thick_in) / h_in, 0.4, thick_in / h_in])
    cbar = ax.figure.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('Per-fiber prediction', fontsize=FS_LABEL - 1)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    ax.set_ylabel("Fibers", fontsize=FS_LABEL)
    ax.set_xlabel("Genomic Position (bp)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    return img


def _render_assay_pair(fig, col_gs, row_start, out_k, out_fibers_k, tar_k, assay_name, mode,
                       has_target=True, avg_loss=None, avg_pearson=None, score_mask=None):
    """
    Renders one assay in 2 consecutive rows of col_gs:
      row_start   → bulk comparison (target vs pred)
      row_start+1 → per-fiber prediction heatmap
    out_k, tar_k: (L,)  out_fibers_k: (L, N)  score_mask: (L,) bool or None
    Returns (ax_sig, ax_heat).
    """
    ax_sig = fig.add_subplot(col_gs[row_start, 0])
    _render_bulk_comparison(ax_sig, tar_k, out_k, assay_name, mode, has_target, avg_loss, avg_pearson, score_mask)
    ax_heat = fig.add_subplot(col_gs[row_start + 1, 0], sharex=ax_sig)
    _render_fiber_heatmap(ax_heat, out_fibers_k)
    return ax_sig, ax_heat


def _filter_informative_fibers(inp, input_flags, min_m6a_sum=20, max_fibers=20):
    """
    Filters uninformative fibers by m6a signal count.
    inp: (C, L, N) numpy array. Returns list of valid fiber indices.
    """
    total_fibers = inp.shape[-1]
    active_features = [j for j, flag in enumerate(input_flags) if flag]

    m6a_channel_idx = 0
    for k, orig_j in enumerate(active_features):
        if FEATURE_NAMES[orig_j] == "m6a":
            m6a_channel_idx = k
            break

    valid_fiber_indices = []
    for f_idx in range(total_fibers):
        m6a_sum = np.sum(inp[m6a_channel_idx, :, f_idx] > 0.5)
        if m6a_sum >= min_m6a_sum:
            valid_fiber_indices.append(f_idx)
        if len(valid_fiber_indices) == max_fibers:
            break

    return valid_fiber_indices


#--------------------------------------------------------------------------------------------------
# Public numpy base functions

def plot_evaluation_dashboard(
    inp,
    input_flags,
    out,
    out_fibers,
    tar,
    locus,
    cell_type,
    output_assays,
    assay_avg_losses=None,
    mode="Train",
    assay_mask=None,
    assay_avg_pearson=None,
    score_mask=None,
):
    """
    Evaluation dashboard: input fiber stack (col 0) + K assays laid out in a roughly square grid of
    assay columns (K=4 → 2 columns of 2, K=8 → 3 columns of 3/3/2). Each assay is a bulk panel
    (target vs predicted, with this window's loss and Pearson r) over its per-fiber prediction heatmap.

    inp: (C, L, N), out: (K, L), out_fibers: (K, L, N), tar: (K, L) — numpy arrays.
    output_assays: list of K assay name strings.
    assay_avg_losses / assay_avg_pearson: dict {assay: float} of epoch/eval averages per assay, or None.
    assay_mask: (K,) bools, False where the cell type has no bulk data for that assay; None = all present.
    score_mask: (L,) bools, positions covered by enough fibers to be scored; None = all. The window's loss and r
    use only these positions, and the others are shaded.
    locus: (chrom, start, end). cell_type: str. Returns matplotlib Figure.
    """
    chr_name, start, end = locus
    K = len(output_assays)
    n_cols = max(1, math.ceil(math.sqrt(K)))
    per_col = math.ceil(K / n_cols)

    # Panel sizes match the old 2-assays-per-column dashboard; margins are fixed in inches so they hold
    # for any K (top: suptitle + 2-line panel title; bottom: last heatmap's x label and colorbar).
    fig_w, fig_h = 10 * (1 + n_cols), 9 * max(2, per_col)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"{mode} — {cell_type} — {chr_name}:{start:,}-{end:,}", fontsize=FS_SUPTITLE, fontweight='bold')
    gs = gridspec.GridSpec(1, 1 + n_cols, figure=fig, wspace=0.28,
                           left=0.8 / fig_w, right=1 - 0.4 / fig_w, top=1 - 1.6 / fig_h, bottom=1.5 / fig_h)

    in_gs = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, 0], hspace=0.25)
    input_axes = _render_input_channels(fig, in_gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title("Input fiber stack", fontsize=FS_TITLE, fontweight='bold')

    # Per assay: signal, heatmap, then a spacer for the heatmap's x label/colorbar and the next title.
    height_ratios = ([1, 1, 0.7] * per_col)[:-1]
    for c in range(n_cols):
        col_gs = gridspec.GridSpecFromSubplotSpec(len(height_ratios), 1, subplot_spec=gs[0, 1 + c],
                                                  height_ratios=height_ratios, hspace=0.12)
        for local_k, k in enumerate(range(c * per_col, min((c + 1) * per_col, K))):
            name = output_assays[k]
            _render_assay_pair(fig, col_gs, local_k * 3, out[k], out_fibers[k], tar[k], name, mode,
                               has_target=True if assay_mask is None else bool(assay_mask[k]),
                               avg_loss=assay_avg_losses.get(name) if assay_avg_losses else None,
                               avg_pearson=assay_avg_pearson.get(name) if assay_avg_pearson else None,
                               score_mask=score_mask)
    return fig


def plot_single_column_fiber_stack(
    true_bulk,
    pred_bulk,
    fiber_seq_inp,
    processed_fibers,
    input_flags,
    locus=None,
    filter_fn=None,
    start_idx=0,
    max_fibers=20,
    bulk_name="Bulk Signal",
    mode="Test",
):
    """
    Bulk signal and individual fiber predictions stacked vertically.

    Layout: [Bulk comparison] then per-fiber [Input features | Accessibility signal],
    in a single column. Pass lists instead of single values for true_bulk,
    pred_bulk, processed_fibers, and bulk_name to render multiple assays as
    side-by-side columns (same selected fibers, same fiber_seq_inp, shared
    across every column) instead of just one.

    true_bulk, pred_bulk: (L,) numpy array, or list of K (L,) arrays.
    fiber_seq_inp: (C, L, N) numpy array — shared across columns.
    processed_fibers: (L, N) numpy array, or list of K (L, N) arrays.
    bulk_name: str, or list of K strings — must match true_bulk/pred_bulk/processed_fibers.
    filter_fn: callable(fiber_idx, fiber_seq_inp) -> bool, or None.
    Returns (Figure, last_used_idx) or (None, last_used_idx).
    """
    is_multi_col = isinstance(true_bulk, (list, tuple))
    true_bulks = list(true_bulk) if is_multi_col else [true_bulk]
    pred_bulks = list(pred_bulk) if is_multi_col else [pred_bulk]
    processed_fibers_list = list(processed_fibers) if is_multi_col else [processed_fibers]
    bulk_names = list(bulk_name) if isinstance(bulk_name, (list, tuple)) else [bulk_name] * len(true_bulks)
    K = len(true_bulks)

    total_fibers = fiber_seq_inp.shape[-1]
    seq_len = fiber_seq_inp.shape[1]

    selected_indices = []
    curr_idx = start_idx
    while curr_idx < total_fibers and len(selected_indices) < max_fibers:
        if filter_fn is None or filter_fn(curr_idx, fiber_seq_inp):
            selected_indices.append(curr_idx)
        curr_idx += 1

    last_used_idx = curr_idx - 1 if len(selected_indices) > 0 else start_idx

    if len(selected_indices) == 0:
        print(f"[Plotter Warning] No fibers satisfied criteria starting from index {start_idx}.")
        return None, last_used_idx

    num_selected = len(selected_indices)
    fig = plt.figure(figsize=(16 * K, 2.2 * (1 + num_selected)))
    gs = gridspec.GridSpec(1 + num_selected, K, figure=fig, hspace=0.45, wspace=0.25)

    x_coords = np.arange(seq_len)

    for col in range(K):
        true_bulk_c, pred_bulk_c = true_bulks[col], pred_bulks[col]
        proc_c, name_c = processed_fibers_list[col], bulk_names[col]

        ax_bulk_row = fig.add_subplot(gs[0, col])
        ax_bulk_row.plot(true_bulk_c, color='dimgray', lw=1.5, label=name_c)
        ax_bulk_row.plot(pred_bulk_c, color='darkorange', lw=1.5, label='Predicted', alpha=0.85)
        ax_bulk_row.set_ylabel("Bulk Intensity", fontsize=10, fontweight='bold')
        ax_bulk_row.legend(loc='upper right', frameon=False)
        ax_bulk_row.set_xlim(0, seq_len)

        if K == 1:
            title_str = f"Bulk & Single Fiber Stack Overview ({mode})"
        else:
            title_str = f"{name_c} — Bulk & Single Fiber Stack ({mode})"
        if locus is not None:
            chr_name, s, e = locus
            title_str += f"\n{chr_name}:{s}-{e}"
        ax_bulk_row.set_title(title_str, fontsize=12, fontweight='bold', pad=10)

        global_max_y = max(1.0, float(proc_c[:, selected_indices].max()))

        for i, fiber_idx in enumerate(selected_indices):
            fiber_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[i + 1, col], hspace=0.0, height_ratios=[1, 1.2]
            )

            ax_inp = fig.add_subplot(fiber_gs[0, 0], sharex=ax_bulk_row)
            _render_single_fiber_features(ax_inp, fiber_seq_inp[:, :, fiber_idx], input_flags)
            ax_inp.set_title(f"Fiber #{fiber_idx}", fontsize=9, fontweight='bold', loc='left', pad=2)
            ax_inp.set_xticklabels([])

            ax_proc = fig.add_subplot(fiber_gs[1, 0], sharex=ax_bulk_row)
            proc_signal = proc_c[:, fiber_idx]
            ax_proc.plot(x_coords, proc_signal, color='firebrick', lw=1.2)
            ax_proc.fill_between(x_coords, 0, proc_signal, color='crimson', alpha=0.45)
            ax_proc.set_ylim(0, global_max_y * 1.05)
            ax_proc.set_ylabel("Accessibility", fontsize=8, fontweight='bold')

            if i < num_selected - 1:
                ax_proc.set_xticklabels([])
            else:
                ax_proc.set_xlabel("Genomic Position (bp)", fontsize=10, fontweight='bold')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.06, right=0.98)
    return fig, last_used_idx


def plot_loss(dir_path, losses, epoch, output_assays):
    """Saves a plot of epoch-wise training loss to dir_path."""
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, f"Epoch_{epoch}_loss.png")
    label = " + ".join(output_assays) if isinstance(output_assays, list) else str(output_assays)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, marker='o', color='tab:blue', lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Curve for {label}")
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evaluator_record(record, input_flags, assay_avg_losses=None, output_assays=None, mode="Test",
                          assay_avg_pearson=None):
    """
    Evaluation dashboard (see plot_evaluation_dashboard) from a single Evaluator locus record.
    record values: fiber_features (C,L,N), processed_fibers (K,L,N), pred_bulk (K,L),
    target_bulk (K,L), locus (chrom,start,end), optional cell_type, assay_mask (K,) and score_mask (L,).
    assay_avg_losses / assay_avg_pearson: dict {assay: float} of averages per assay, or None.
    Returns matplotlib Figure.
    """
    if output_assays is None:
        output_assays = ["atac"]
    return plot_evaluation_dashboard(
        record["fiber_features"], input_flags, record["pred_bulk"], record["processed_fibers"],
        record["target_bulk"], record["locus"], record.get("cell_type", "Unknown"), output_assays,
        assay_avg_losses, mode, record.get("assay_mask"), assay_avg_pearson, record.get("score_mask"))


def plot_dna_tensor_logo(dna_arr, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Sequence logo from a DNA array of shape (4, L, N). Averages over N axis.
    Returns matplotlib Figure.
    """
    ppm = dna_arr.mean(axis=2)  # (4, L)
    ppm_T = ppm.T               # (L, 4)
    df = pd.DataFrame(ppm_T, columns=['A', 'C', 'G', 'T'])

    if region_slice is not None:
        s, e = region_slice
        df = df.iloc[s:e].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 3))
    logo = logomaker.Logo(df, ax=ax, vpad=0.05)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True)
    ax.set_ylabel("Probability / Frequency")
    ax.set_xlabel("Position (bp)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


#--------------------------------------------------------------------------------------------------
# Public torch wrapper functions (_t suffix)

import torch


def plot_evaluation_dashboard_t(
    inp_t,
    input_flags,
    out_t,
    out_fibers_t,
    tar_t,
    locus,
    cell_type,
    output_assays,
    assay_avg_losses=None,
    mode="Train",
    assay_mask=None,
    assay_avg_pearson=None,
    score_mask=None,
):
    """
    Torch wrapper for plot_evaluation_dashboard.
    inp_t: (B, C, L, N), out_t: (B, K, L), out_fibers_t: (B, K, L, N), tar_t: (B, K, L).
    output_assays: list of K assay name strings.
    assay_avg_losses / assay_avg_pearson: dict {assay: float} of epoch averages per assay, or None.
    assay_mask: (B, K) tensor (1 = cell type has bulk data for the assay), or None = all present.
    score_mask: (B, L) bool tensor of scored positions (enough fibers), or None = all.
    locus: collated list-of-lists from DataLoader. cell_type: list[str].
    """
    mask = None if assay_mask is None else assay_mask[0].cpu().numpy().astype(bool)
    scored = None if score_mask is None else score_mask[0].cpu().numpy().astype(bool)
    inp = inp_t[0].cpu().detach().float().numpy()
    out = out_t[0].cpu().detach().float().numpy()           # (K, L)
    out_fibers = out_fibers_t[0].cpu().detach().float().numpy()  # (K, L, N)
    tar = tar_t[0].cpu().detach().float().numpy()           # (K, L)
    if isinstance(locus[0], (list, tuple)):
        locus_tup = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
    else:
        locus_tup = (locus[0], int(locus[1]), int(locus[2]))
    ct = cell_type[0] if isinstance(cell_type, (list, tuple)) else cell_type
    return plot_evaluation_dashboard(inp, input_flags, out, out_fibers, tar, locus_tup, ct, output_assays,
                                     assay_avg_losses, mode, mask, assay_avg_pearson, scored)


def plot_single_column_fiber_stack_t(
    true_bulk_t,
    pred_bulk_t,
    fiber_seq_inp_t,
    processed_fibers_t,
    input_flags,
    locus=None,
    filter_fn=None,
    start_idx=0,
    max_fibers=20,
    bulk_name="Bulk Signal",
    mode="Test",
):
    """
    Torch wrapper for plot_single_column_fiber_stack.
    true_bulk_t, pred_bulk_t: (L,) or (1, L) [or a list of these, one per assay column].
    fiber_seq_inp_t: (C,L,N) or (1,C,L,N) — shared across columns.
    processed_fibers_t: (L,N) or (1,L,N) [or a list of these, one per assay column].
    Pass lists (matching length, one per assay) for true_bulk_t/pred_bulk_t/
    processed_fibers_t/bulk_name to render multiple side-by-side columns.
    filter_fn receives (fiber_idx, fiber_seq_inp_np) where fiber_seq_inp_np is (C,L,N) numpy.
    """
    def _bulk_to_np(t):
        return t.squeeze().cpu().detach().float().numpy()

    def _fibers_to_np(t):
        if t.dim() == 3:
            t = t[0]
        return t.cpu().detach().float().numpy()

    if isinstance(true_bulk_t, (list, tuple)):
        true_bulk = [_bulk_to_np(t) for t in true_bulk_t]
        pred_bulk = [_bulk_to_np(t) for t in pred_bulk_t]
        processed_fibers = [_fibers_to_np(t) for t in processed_fibers_t]
    else:
        true_bulk = _bulk_to_np(true_bulk_t)
        pred_bulk = _bulk_to_np(pred_bulk_t)
        processed_fibers = _fibers_to_np(processed_fibers_t)

    if fiber_seq_inp_t.dim() == 4:
        fiber_seq_inp_t = fiber_seq_inp_t[0]
    fiber_seq_inp = fiber_seq_inp_t.cpu().detach().float().numpy()

    return plot_single_column_fiber_stack(
        true_bulk, pred_bulk, fiber_seq_inp, processed_fibers,
        input_flags, locus=locus, filter_fn=filter_fn,
        start_idx=start_idx, max_fibers=max_fibers,
        bulk_name=bulk_name, mode=mode,
    )


def plot_evaluator_record_t(record_t, input_flags, assay_avg_losses=None, output_assays=None, mode="Test",
                            assay_avg_pearson=None):
    """
    Torch wrapper for plot_evaluator_record.
    record_t: dict from Evaluator with batched CPU tensors and collated locus.
    Squeezes batch dim [0] and converts tensors to numpy.
    output_assays: list of K assay name strings.
    assay_avg_losses / assay_avg_pearson: dict {assay: float} of averages per assay, or None.
    """
    record = {}
    squeeze_keys = {"fiber_features", "processed_fibers", "pred_bulk", "target_bulk"}
    for k, v in record_t.items():
        if k in squeeze_keys and isinstance(v, torch.Tensor):
            record[k] = v[0].cpu().detach().float().numpy()
        elif k in ("assay_mask", "score_mask") and isinstance(v, torch.Tensor):
            record[k] = v[0].cpu().numpy().astype(bool)
        elif k == "locus":
            locus = v
            record[k] = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
        elif k == "cell_type":
            record[k] = v[0] if isinstance(v, (list, tuple)) else v
        else:
            record[k] = v
    return plot_evaluator_record(record, input_flags, assay_avg_losses, output_assays, mode, assay_avg_pearson)


def plot_dna_tensor_logo_t(dna_tensor_t, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Torch wrapper for plot_dna_tensor_logo.
    dna_tensor_t: (4, L, N) or (B, 4, L, N) tensor.
    """
    if dna_tensor_t.dim() == 4:
        dna_tensor_t = dna_tensor_t[0]
    dna_arr = dna_tensor_t.cpu().detach().float().numpy()
    return plot_dna_tensor_logo(dna_arr, region_slice, title)
