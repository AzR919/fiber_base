"""
Visualization utilities for fiber-seq evaluation dashboards.
Base functions accept numpy arrays; _t wrapper functions accept torch tensors.
"""

import os
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

        ax.set_ylabel(FEATURE_NAMES[j], fontsize=11, fontweight='bold')
        ax.set_ylim(-num_fibers - 0.5, 0.5)
        ax.set_xlim(0, inp.shape[1])

        if k < num_active - 1:
            ax.set_xticklabels([])
        k += 1

    if input_axes:
        input_axes[-1].set_xlabel("Genomic Position (bp)")

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


def _render_bulk_comparison(ax, target, pred_bulk, chr_info, instance_loss, mode, bulk_name, avg_loss=None, cell_type=None):
    """Sub-renderer for target vs predicted bulk signal. target, pred_bulk: (L,) numpy arrays."""
    ax.plot(target, color='dimgray', lw=1.5, label=bulk_name)
    ax.plot(pred_bulk, color='darkorange', lw=1.5, label='Predicted', alpha=0.8)
    ax.set_ylabel("Signal Intensity")
    ax.legend(loc='upper right', frameon=False)

    title = f"{bulk_name} — {cell_type} | Instance Loss: {instance_loss:.4f}"
    if avg_loss is not None:
        title += f"\nEpoch Avg Loss ({mode}): {avg_loss:.4f} | {chr_info}"
    else:
        title += f"\n{chr_info}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticklabels([])


def _render_fiber_heatmap(ax, out_fibers):
    """Sub-renderer for predicted fiber accessibility heatmap. out_fibers: (L, N) numpy array."""
    pred_matrix = out_fibers.T
    img = ax.imshow(pred_matrix, aspect='auto', cmap='magma',
                    interpolation='nearest', origin='upper',
                    extent=[0, pred_matrix.shape[1], -pred_matrix.shape[0], 0])
    plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.10, fraction=0.04, label='Accessibility Probability')
    ax.set_ylabel("Fibers (Imputed)")
    ax.set_xlabel("Genomic Position (bp)")
    return img


def _render_assay_pair(fig, col_gs, row_start, out_k, out_fibers_k, tar_k,
                        assay_name, chr_info, instance_loss, mode, avg_loss, cell_type):
    """
    Renders one assay in 2 sub-rows of col_gs:
      row_start   → bulk comparison (target vs pred)
      row_start+1 → fiber heatmap for that assay
    out_k, tar_k: (L,)  out_fibers_k: (L, N)
    Returns (ax_sig, ax_heat).
    """
    ax_sig = fig.add_subplot(col_gs[row_start, 0])
    _render_bulk_comparison(ax_sig, tar_k, out_k, chr_info, instance_loss, mode,
                             assay_name, avg_loss, cell_type)
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
):
    """
    3-column evaluation dashboard: input features (col 0), assays 1-2 (col 1), assays 3-4 (col 2).

    inp: (C, L, N), out: (K, L), out_fibers: (K, L, N), tar: (K, L) — numpy arrays.
    output_assays: list of K assay name strings.
    assay_avg_losses: dict {assay: float} of epoch-average losses per assay, or None.
    locus: (chrom, start, end). cell_type: str. Returns matplotlib Figure.
    """
    chr_name, start, end = locus
    chr_info = f"{chr_name}:{start}-{end}"
    K = len(output_assays)

    num_data_cols = 1 + max(1, (K + 1) // 2)  # col0 + 1 or 2 assay columns
    fig_width = 10 * num_data_cols
    fig = plt.figure(figsize=(fig_width, 18))
    gs = gridspec.GridSpec(5, num_data_cols, figure=fig,
                           width_ratios=[1] * num_data_cols, wspace=0.3, hspace=0.3)

    input_axes = _render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title(f"Input Features, {cell_type}\n{chr_info}", fontsize=13, fontweight='bold')

    for pair_idx, k_start in enumerate(range(0, K, 2)):
        col_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1 + pair_idx], hspace=0.4)
        for local_k, k in enumerate(range(k_start, min(k_start + 2, K))):
            instance_loss = float(np.mean((tar[k] - out[k]) ** 2))
            avg_loss_k = assay_avg_losses.get(output_assays[k]) if assay_avg_losses else None
            _render_assay_pair(fig, col_gs, local_k * 2,
                               out[k], out_fibers[k], tar[k],
                               output_assays[k], chr_info, instance_loss, mode, avg_loss_k, cell_type)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.96)
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
    Bulk signal and individual fiber predictions stacked vertically in a single column.

    Layout: [Bulk comparison] then per-fiber [Input features | Accessibility signal].
    true_bulk, pred_bulk: (L,) numpy arrays.
    fiber_seq_inp: (C, L, N) numpy array.
    processed_fibers: (L, N) numpy array.
    filter_fn: callable(fiber_idx, fiber_seq_inp) -> bool, or None.
    Returns (Figure, last_used_idx) or (None, last_used_idx).
    """
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
    fig = plt.figure(figsize=(16, 2.2 * (1 + num_selected)))
    gs = gridspec.GridSpec(1 + num_selected, 1, figure=fig, hspace=0.45)

    ax_bulk_row = fig.add_subplot(gs[0, 0])
    ax_bulk_row.plot(true_bulk, color='dimgray', lw=1.5, label=bulk_name)
    ax_bulk_row.plot(pred_bulk, color='darkorange', lw=1.5, label='Predicted', alpha=0.85)
    ax_bulk_row.set_ylabel("Bulk Intensity", fontsize=10, fontweight='bold')
    ax_bulk_row.legend(loc='upper right', frameon=False)
    ax_bulk_row.set_xlim(0, seq_len)

    title_str = f"Bulk & Single Fiber Stack Overview ({mode})"
    if locus is not None:
        chr_name, s, e = locus
        title_str += f"\n{chr_name}:{s}-{e}"
    ax_bulk_row.set_title(title_str, fontsize=12, fontweight='bold', pad=10)

    x_coords = np.arange(seq_len)
    global_max_y = max(1.0, float(processed_fibers[:, selected_indices].max()))

    for i, fiber_idx in enumerate(selected_indices):
        fiber_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[i + 1, 0], hspace=0.0, height_ratios=[1, 1.2]
        )

        ax_inp = fig.add_subplot(fiber_gs[0, 0], sharex=ax_bulk_row)
        _render_single_fiber_features(ax_inp, fiber_seq_inp[:, :, fiber_idx], input_flags)
        ax_inp.set_title(f"Fiber #{fiber_idx}", fontsize=9, fontweight='bold', loc='left', pad=2)
        ax_inp.set_xticklabels([])

        ax_proc = fig.add_subplot(fiber_gs[1, 0], sharex=ax_bulk_row)
        proc_signal = processed_fibers[:, fiber_idx]
        ax_proc.plot(x_coords, proc_signal, color='firebrick', lw=1.2)
        ax_proc.fill_between(x_coords, 0, proc_signal, color='crimson', alpha=0.45)
        ax_proc.set_ylim(0, global_max_y * 1.05)
        ax_proc.set_ylabel("Accessibility", fontsize=8, fontweight='bold')

        if i < num_selected - 1:
            ax_proc.set_xticklabels([])
        else:
            ax_proc.set_xlabel("Genomic Position (bp)", fontsize=10, fontweight='bold')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.92)
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


def plot_evaluator_record(record, input_flags, assay_avg_losses=None, output_assays=None, mode="Test"):
    """
    3-column evaluation dashboard from a single Evaluator locus record.
    record values: fiber_features (C,L,N), processed_fibers (K,L,N), pred_bulk (K,L),
    target_bulk (K,L), locus (chrom,start,end).
    output_assays: list of K assay name strings.
    assay_avg_losses: dict {assay: float} of epoch-average losses per assay, or None.
    Returns matplotlib Figure.
    """
    if output_assays is None:
        output_assays = ["atac"]
    K = len(output_assays)

    chr_name, start, end = record["locus"]
    chr_info = f"{chr_name}:{start}-{end}"

    inp = record["fiber_features"]                # (C, L, N)
    processed_fibers = record["processed_fibers"] # (K, L, N)
    pred_bulk = record["pred_bulk"]               # (K, L)
    target_bulk = record["target_bulk"]           # (K, L)

    cell_type_label = record.get("cell_type", "Unknown")

    num_data_cols = 1 + max(1, (K + 1) // 2)
    fig_width = 10 * num_data_cols
    fig = plt.figure(figsize=(fig_width, 18))
    gs = gridspec.GridSpec(5, num_data_cols, figure=fig,
                           width_ratios=[1] * num_data_cols, wspace=0.3, hspace=0.35)

    input_axes = _render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title(f"Input Fiber Stack\n{chr_info}", fontsize=12, fontweight='bold')

    for pair_idx, k_start in enumerate(range(0, K, 2)):
        col_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1 + pair_idx], hspace=0.45)
        for local_k, k in enumerate(range(k_start, min(k_start + 2, K))):
            inst_k = float(np.mean((target_bulk[k] - pred_bulk[k]) ** 2))
            avg_loss_k = assay_avg_losses.get(output_assays[k]) if assay_avg_losses else None
            _render_assay_pair(fig, col_gs, local_k * 2,
                               pred_bulk[k], processed_fibers[k], target_bulk[k],
                               output_assays[k], chr_info, inst_k, mode, avg_loss_k, cell_type_label)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.96)
    return fig


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
):
    """
    Torch wrapper for plot_evaluation_dashboard.
    inp_t: (B, C, L, N), out_t: (B, K, L), out_fibers_t: (B, K, L, N), tar_t: (B, K, L).
    output_assays: list of K assay name strings.
    assay_avg_losses: dict {assay: float} of epoch-average losses per assay, or None.
    locus: collated list-of-lists from DataLoader. cell_type: list[str].
    """
    inp = inp_t[0].cpu().detach().float().numpy()
    out = out_t[0].cpu().detach().float().numpy()           # (K, L)
    out_fibers = out_fibers_t[0].cpu().detach().float().numpy()  # (K, L, N)
    tar = tar_t[0].cpu().detach().float().numpy()           # (K, L)
    if isinstance(locus[0], (list, tuple)):
        locus_tup = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
    else:
        locus_tup = (locus[0], int(locus[1]), int(locus[2]))
    ct = cell_type[0] if isinstance(cell_type, (list, tuple)) else cell_type
    return plot_evaluation_dashboard(inp, input_flags, out, out_fibers, tar, locus_tup, ct, output_assays, assay_avg_losses, mode)


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
    true_bulk_t, pred_bulk_t: (L,) or (1, L). fiber_seq_inp_t: (C,L,N) or (1,C,L,N).
    processed_fibers_t: (L,N) or (1,L,N).
    filter_fn receives (fiber_idx, fiber_seq_inp_np) where fiber_seq_inp_np is (C,L,N) numpy.
    """
    true_bulk = true_bulk_t.squeeze().cpu().detach().float().numpy()
    pred_bulk = pred_bulk_t.squeeze().cpu().detach().float().numpy()

    if fiber_seq_inp_t.dim() == 4:
        fiber_seq_inp_t = fiber_seq_inp_t[0]
    fiber_seq_inp = fiber_seq_inp_t.cpu().detach().float().numpy()

    if processed_fibers_t.dim() == 3:
        processed_fibers_t = processed_fibers_t[0]
    processed_fibers = processed_fibers_t.cpu().detach().float().numpy()

    return plot_single_column_fiber_stack(
        true_bulk, pred_bulk, fiber_seq_inp, processed_fibers,
        input_flags, locus=locus, filter_fn=filter_fn,
        start_idx=start_idx, max_fibers=max_fibers,
        bulk_name=bulk_name, mode=mode,
    )


def plot_evaluator_record_t(record_t, input_flags, assay_avg_losses=None, output_assays=None, mode="Test"):
    """
    Torch wrapper for plot_evaluator_record.
    record_t: dict from Evaluator with batched CPU tensors and collated locus.
    Squeezes batch dim [0] and converts tensors to numpy.
    output_assays: list of K assay name strings.
    assay_avg_losses: dict {assay: float} of epoch-average losses per assay, or None.
    """
    record = {}
    squeeze_keys = {"fiber_features", "processed_fibers", "pred_bulk", "target_bulk"}
    for k, v in record_t.items():
        if k in squeeze_keys and isinstance(v, torch.Tensor):
            record[k] = v[0].cpu().detach().float().numpy()
        elif k == "locus":
            locus = v
            record[k] = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
        elif k == "cell_type":
            record[k] = v[0] if isinstance(v, (list, tuple)) else v
        else:
            record[k] = v
    return plot_evaluator_record(record, input_flags, assay_avg_losses, output_assays, mode)


def plot_dna_tensor_logo_t(dna_tensor_t, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Torch wrapper for plot_dna_tensor_logo.
    dna_tensor_t: (4, L, N) or (B, 4, L, N) tensor.
    """
    if dna_tensor_t.dim() == 4:
        dna_tensor_t = dna_tensor_t[0]
    dna_arr = dna_tensor_t.cpu().detach().float().numpy()
    return plot_dna_tensor_logo(dna_arr, region_slice, title)
