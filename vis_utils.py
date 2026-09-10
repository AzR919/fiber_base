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

    title = f"Imputation Results {cell_type} ({mode} Loss: {instance_loss:.6f})\n{chr_info}"
    if avg_loss is not None:
        title += f" (Epoch Avg {mode} Loss: {avg_loss:.6f})"
    ax.set_title(title, fontsize=13, fontweight='bold')
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
    bulk_name,
    avg_loss=None,
    mode="Train",
):
    """
    Unified evaluation dashboard: input features (left) and predicted outputs (right).

    inp: (C, L, N), out: (L,), out_fibers: (L, N), tar: (L,) — numpy arrays.
    locus: (chrom, start, end). cell_type: str. Returns matplotlib Figure.
    """
    chr_name, start, end = locus
    chr_info = f"{chr_name}:{start}-{end}"
    num_input_features = sum(input_flags)

    grid_rows = max(2, num_input_features)
    fig_height = max(10, 2.5 * grid_rows)
    fig = plt.figure(figsize=(20, fig_height))
    gs = gridspec.GridSpec(grid_rows, 2, figure=fig, width_ratios=[1, 1], wspace=0.25, hspace=0.3)

    input_axes = _render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title(f"Input Features, {cell_type}\n{chr_info}", fontsize=13, fontweight='bold')

    ax_bulk = fig.add_subplot(gs[0:1, 1])
    ax_heat = fig.add_subplot(gs[1:3, 1], sharex=ax_bulk)

    instance_loss = float(np.mean((tar - out) ** 2))
    _render_bulk_comparison(ax_bulk, tar, out, chr_info, instance_loss, mode, bulk_name, avg_loss, cell_type)
    _render_fiber_heatmap(ax_heat, out_fibers)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
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


def plot_loss(dir_path, losses, epoch, bulk_name):
    """Saves a plot of epoch-wise training loss to dir_path."""
    os.makedirs(dir_path, exist_ok=True)
    save_path = os.path.join(dir_path, f"Epoch_{epoch}_loss.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, marker='o', color='tab:blue', lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss Curve for {bulk_name}")
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_evaluator_record(record, input_flags, loss=0.0, ct_losses={}, bulk_name="H3K27ac", mode="Test"):
    """
    Multi-panel deconvolution dashboard from a single Evaluator locus record.
    record values: fiber_features (C,L,N), processed_fibers (L,N), pred_bulk (L,),
    target_bulk (L,), pred/target_cell_type_bulks {ct: (L,)}, locus (chrom,start,end).
    Returns matplotlib Figure.
    """
    chr_name, start, end = record["locus"]
    chr_info = f"{chr_name}:{start}-{end}"

    inp = record["fiber_features"]                # (C, L, N)
    processed_fibers = record["processed_fibers"] # (L, N)
    pred_bulk = record["pred_bulk"]               # (L,)
    target_bulk = record["target_bulk"]           # (L,)
    ct_preds = record["pred_cell_type_bulks"]     # {ct: (L,)}
    ct_targets = record["target_cell_type_bulks"] # {ct: (L,)}
    instance_loss = record["loss"]
    ct_instance_loss = record["cell_type_losses"]

    cell_types = list(ct_preds.keys())[:2]
    num_active_inputs = sum(input_flags)

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(max(4, num_active_inputs), 2, figure=fig, width_ratios=[1, 1], wspace=0.25, hspace=0.35)

    input_axes = _render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title(f"Input Fiber Stack\n{chr_info}", fontsize=12, fontweight='bold')

    rhs_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1], hspace=0.45)

    ax_comp = fig.add_subplot(rhs_gs[0, 0])
    _render_bulk_comparison(ax_comp, target_bulk, pred_bulk, chr_info, instance_loss, mode, bulk_name, loss,
                            cell_type="Mixed" if len(cell_types) > 1 else cell_types[0])

    ax_heat = fig.add_subplot(rhs_gs[1, 0], sharex=ax_comp)
    _render_fiber_heatmap(ax_heat, processed_fibers)

    if len(cell_types) >= 2:
        ct_a = cell_types[0]
        ax_cta = fig.add_subplot(rhs_gs[2, 0], sharex=ax_comp)
        _render_bulk_comparison(ax_cta, ct_targets[ct_a], ct_preds[ct_a], "", ct_instance_loss[ct_a], mode,
                                bulk_name, ct_losses[ct_a]["loss"], cell_type=ct_a)

        ct_b = cell_types[1]
        ax_ctb = fig.add_subplot(rhs_gs[3, 0], sharex=ax_comp)
        _render_bulk_comparison(ax_ctb, ct_targets[ct_b], ct_preds[ct_b], "", ct_instance_loss[ct_a], mode,
                                bulk_name, ct_losses[ct_b]["loss"], cell_type=ct_b)
        ax_ctb.set_xlabel("Genomic Position (bp)", fontsize=11)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
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
    bulk_name,
    avg_loss=None,
    mode="Train",
):
    """
    Torch wrapper for plot_evaluation_dashboard.
    inp_t: (B, C, L, N), out_t: (B, L), out_fibers_t: (B, L, N), tar_t: (B, L).
    locus: collated list-of-lists from DataLoader. cell_type: list[str].
    """
    inp = inp_t[0].cpu().detach().float().numpy()
    out = out_t[0].cpu().detach().float().numpy()
    out_fibers = out_fibers_t[0].cpu().detach().float().numpy()
    tar = tar_t[0].cpu().detach().float().numpy()
    # locus can be a plain (chrom, start, end) tuple or DataLoader-collated ([chroms], tensor, tensor)
    if isinstance(locus[0], (list, tuple)):
        locus_tup = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
    else:
        locus_tup = (locus[0], int(locus[1]), int(locus[2]))
    ct = cell_type[0] if isinstance(cell_type, (list, tuple)) else cell_type
    return plot_evaluation_dashboard(inp, input_flags, out, out_fibers, tar, locus_tup, ct, bulk_name, avg_loss, mode)


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


def plot_evaluator_record_t(record_t, input_flags, loss=0.0, ct_losses={}, bulk_name="H3K27ac", mode="Test"):
    """
    Torch wrapper for plot_evaluator_record.
    record_t: dict from Evaluator with batched CPU tensors and collated locus.
    Squeezes batch dim [0] and converts tensors to numpy.
    """
    record = {}
    squeeze_keys = {"fiber_features", "processed_fibers", "pred_bulk", "target_bulk"}
    for k, v in record_t.items():
        if k in squeeze_keys and isinstance(v, torch.Tensor):
            record[k] = v[0].cpu().detach().float().numpy()
        elif k in ("pred_cell_type_bulks", "target_cell_type_bulks") and isinstance(v, dict):
            record[k] = {
                ct: (t[0].cpu().detach().float().numpy() if isinstance(t, torch.Tensor) else t)
                for ct, t in v.items()
            }
        elif k == "locus":
            locus = v
            record[k] = (locus[0][0], int(locus[1][0]), int(locus[2][0]))
        else:
            record[k] = v
    return plot_evaluator_record(record, input_flags, loss, ct_losses, bulk_name, mode)


def plot_dna_tensor_logo_t(dna_tensor_t, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Torch wrapper for plot_dna_tensor_logo.
    dna_tensor_t: (4, L, N) or (B, 4, L, N) tensor.
    """
    if dna_tensor_t.dim() == 4:
        dna_tensor_t = dna_tensor_t[0]
    dna_arr = dna_tensor_t.cpu().detach().float().numpy()
    return plot_dna_tensor_logo(dna_arr, region_slice, title)
