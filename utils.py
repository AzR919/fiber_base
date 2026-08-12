"""
Common utility functions for experiment tracking, seeding, training, and visualization.
"""

import os
import sys
import random
import shutil
import datetime
import contextlib
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from torchinfo import summary as torchinfo_summary

#--------------------------------------------------------------------------------------------------
# Reproducibility & Environment Setup

def save_slurm_script(res_dir, slurm_script_path):
    """
    Copies the active Slurm batch submission script into the output result directory.
    """
    try:
        destination = os.path.join(res_dir, "submitted_sbatch_script.sh")
        shutil.copy(slurm_script_path, destination)
        print(f"[Slurm Tracker] Successfully copied submission script to: {destination}")
    except:
        print(f"[Slurm Tracker] Failed to copy submission script [{slurm_script_path}] to: {destination}")

def set_seed(seed: int = 919):
    """
    Sets seeds for Python, NumPy, and PyTorch across CPU and GPU.
    Enforces deterministic CUDA operations for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Worker init function for PyTorch DataLoader to ensure deterministic multi-processing sampling.
    Pass as: DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#--------------------------------------------------------------------------------------------------
# File Utilities & Experiment Tracking

def get_config_names_str(args) -> str:
    """
    Extracts filenames (without paths or extensions) from provided config arguments
    and joins them with underscores.
    """
    config_keys = ["data_config", "model_config", "train_config", "eval_config"]
    config_names = [args.name_prefix] if args.name_prefix is not None else []

    for key in config_keys:
        cfg_path = getattr(args, key, None)
        if cfg_path:
            # Extract filename without extension (e.g., 'path/to/data_config.yaml' -> 'data_config')
            base_name = os.path.splitext(os.path.basename(cfg_path))[0]
            config_names.append(base_name)

    if args.name_suffix is not None:
        config_names.append(args.name_suffix)

    # Join extracted config names (e.g., "data_config_model_config_train_config")
    return "_".join(config_names)


def create_save_str(args) -> str:
    """Generates a structured, unique run identifier including timestamp and concatenated config names."""
    now = datetime.datetime.now().strftime("%y-%m-%d_T%H-%M-%S")
    configs_str = get_config_names_str(args)

    # Build component list, filtering out empty strings
    components = [now]
    if configs_str:
        components.append(configs_str)

    return "_".join(components)


class AverageMeter:
    """Computes and stores the running average and current value of metrics during training."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> int:
    """Returns total count of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (16, 5, 2048, 200)):
    """
    Pretty-prints model architecture summary using torchinfo if available,
    otherwise falls back to parameter count and string representation.
    """
    print("\n" + "=" * 60)
    print(f" MODEL SUMMARY: {model.__class__.__name__}")
    print("=" * 60)
    print(f" Total Trainable Parameters: {count_parameters(model):,}\n")

    try:
        summary_str = torchinfo_summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            row_settings=["var_names"],
            verbose=0
        )
        print(summary_str)
    except Exception as e:
        print(f"Notice: torchinfo summary could not run on sample input shape {input_size}. ({e})")
        print(model)
    print("=" * 60 + "\n")


@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr at the OS level."""
    devnull = os.open(os.devnull, os.O_RDWR)
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        os.close(save_stdout)
        os.close(save_stderr)
        os.close(devnull)


#--------------------------------------------------------------------------------------------------
# Modular Plotting Components

FEATURE_NAMES = ["m6a", "cpg", "msp", "nuc", "fire_msp"]
FEATURE_COLORS = ["black", "purple", "blue", "green", "red"]

def render_input_channels(fig, gs_column, inp, input_flags):
    """Sub-renderer for discrete and continuous dynamic input channels (Left Column)."""
    num_fibers = inp.shape[-1]
    active_indices = [j for j, flag in enumerate(input_flags) if flag]
    num_active = len(active_indices)

    k = 0
    input_axes = []
    for j in active_indices:
        ax = fig.add_subplot(gs_column[k, 0])
        input_axes.append(ax)
        is_single_bit = FEATURE_NAMES[j] in ["m6a", "cpg"]

        for i in range(num_fibers):
            fiber_feat = inp[0, k, :, i].cpu().detach()

            if is_single_bit:
                indices = torch.where(fiber_feat > 0.5)[0].numpy()
                if len(indices) > 0:
                    ax.scatter(indices, np.full_like(indices, -i),
                               marker='|', color=FEATURE_COLORS[j], s=25, alpha=0.7, linewidths=0.9)
            else:
                masked = (fiber_feat > 0.5).float()
                diff = torch.diff(masked, prepend=torch.tensor([0.0]), append=torch.tensor([0.0]))
                starts = torch.where(diff == 1)[0]
                ends = torch.where(diff == -1)[0]

                for s, e in zip(starts, ends):
                    if e > s:
                        ax.axhspan(-i - 0.35, -i + 0.35,
                                   xmin=(s / len(fiber_feat)).item(), xmax=(e / len(fiber_feat)).item(),
                                   color=FEATURE_COLORS[j], alpha=0.5, lw=0)

        ax.set_ylabel(FEATURE_NAMES[j], fontsize=11, fontweight='bold')
        ax.set_ylim(-num_fibers - 0.5, 0.5)
        ax.set_xlim(0, inp.shape[2])

        if k < num_active - 1:
            ax.set_xticklabels([])
        k += 1

    if input_axes:
        input_axes[-1].set_xlabel("Genomic Position (bp)")

    return input_axes


def render_bulk_comparison(ax, target, pred_bulk, chr_info, instance_loss, mode, bulk_name, avg_loss=None):
    """Sub-renderer for target vs predicted bulk signal comparison."""
    ax.plot(target.cpu().numpy(), color='dimgray', lw=1.5, label=bulk_name)
    ax.plot(pred_bulk.cpu().detach().numpy(), color='darkorange', lw=1.5, label='Predicted', alpha=0.8)
    ax.set_ylabel("Signal Intensity")
    ax.legend(loc='upper right', frameon=False)

    title = f"Imputation Results ({mode} Loss: {instance_loss:.6f})\n{chr_info}"
    if avg_loss is not None:
        title += f" (Epoch Avg {mode} Loss: {avg_loss:.6f})"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticklabels([])

def render_bulk_signal_panel(ax, target, pred, bulk_name, title_prefix="", color_pred='darkorange', color_tar='dimgray'):
    """
    Sub-renderer for comparing target bulk signal vs predicted bulk signal on RHS.
    Computes MSE loss and Pearson R correlation on the fly, and uses configurable prediction colors.

    Args:
        ax (matplotlib.axes.Axes): Axis object to draw on.
        target (torch.Tensor or np.ndarray): Ground truth bulk signal tensor/array.
        pred (torch.Tensor or np.ndarray): Predicted bulk signal tensor/array.
        bulk_name (str): Histone modification or bulk signal name (e.g., 'H3K27ac', 'H3K4me3').
        title_prefix (str): Prefix title for the panel (e.g., 'Composite Mixed Signal', 'Deconvoluted Bulk: GM12878').
        color_pred (str): Line color for the prediction trace.
        color_tar (str): Line color for the target trace.
    """
    # 1. Flatten inputs to 1D numpy arrays
    t_np = target.detach().cpu().numpy().flatten() if isinstance(target, torch.Tensor) else np.asarray(target).flatten()
    p_np = pred.detach().cpu().numpy().flatten() if isinstance(pred, torch.Tensor) else np.asarray(pred).flatten()

    # 2. Compute performance metrics
    loss = float(np.mean((t_np - p_np) ** 2))

    # 3. Plot target and prediction traces
    ax.plot(t_np, color=color_tar, lw=1.5, label=f"Target ({bulk_name})")
    ax.plot(p_np, color=color_pred, lw=1.5, label="Predicted", alpha=0.85)

    # 4. Format labels, legend, and title
    ax.set_ylabel("Signal", fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    title_str = f"{title_prefix} [{bulk_name}]" if title_prefix else f"[{bulk_name}]"
    ax.set_title(f"{title_str} (MSE: {loss:.5f})", fontsize=11, fontweight='bold')
    ax.set_xticklabels([])


def render_fiber_heatmap(ax, out_fibers):
    """Sub-renderer for predicted fiber accessibility heatmap."""
    pred_matrix = out_fibers[0].cpu().detach().numpy().T
    img = ax.imshow(pred_matrix, aspect='auto', cmap='magma',
                         interpolation='nearest', origin='upper',
                         extent=[0, pred_matrix.shape[1], -pred_matrix.shape[0], 0])

    plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.10, fraction=0.04, label='Accessibility Probability')
    ax.set_ylabel("Fibers (Imputed)")
    ax.set_xlabel("Genomic Position (bp)")
    return img


def filter_informative_fibers(inp, input_flags, min_m6a_sum=20, max_fibers=20):
    """Filters out uninformative fibers based on m6a signal presence."""
    total_fibers = inp.shape[-1]
    active_features = [j for j, flag in enumerate(input_flags) if flag]

    m6a_channel_idx = 0
    for k, orig_j in enumerate(active_features):
        if FEATURE_NAMES[orig_j] == "m6a":
            m6a_channel_idx = k
            break

    valid_fiber_indices = []
    for f_idx in range(total_fibers):
        m6a_sum = torch.sum(inp[0, m6a_channel_idx, :, f_idx] > 0.5).item()
        if m6a_sum >= min_m6a_sum:
            valid_fiber_indices.append(f_idx)
        if len(valid_fiber_indices) == max_fibers:
            break

    return valid_fiber_indices


#--------------------------------------------------------------------------------------------------
# Top-Level Visualization Dashboards (Return Figure Objects)

def plot_evaluation_dashboard(inp, input_flags, out, out_fibers, tar, locus, cell_type, bulk_name, avg_loss=None, mode="Train"):
    """
    Constructs a unified evaluation figure displaying inputs (left) and predicted outputs (right).
    Returns Matplotlib Figure object for saving or W&B logging.
    """
    chr_name, start, end = locus[0][0], locus[1][0], locus[2][0]
    chr_info = f"{chr_name}:{start}-{end}"
    cell_info = cell_type[0]
    num_input_features = sum(input_flags)

    grid_rows = max(2, num_input_features)
    fig_height = max(10, 2.5 * grid_rows)
    fig = plt.figure(figsize=(20, fig_height))
    gs = gridspec.GridSpec(grid_rows, 2, figure=fig, width_ratios=[1, 1], wspace=0.25, hspace=0.3)

    # Left Column: Inputs
    input_axes = render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        input_axes[0].set_title(f"Input Features, {cell_info}\n{chr_info}", fontsize=13, fontweight='bold')

    # Right Column: Bulk + Heatmap
    ax_bulk = fig.add_subplot(gs[0:1, 1])
    ax_heat = fig.add_subplot(gs[1:3, 1], sharex=ax_bulk)

    tar_sig = tar[0].cpu().detach().numpy()
    out_sig = out[0].cpu().detach().numpy()
    instance_loss = float(np.mean((tar_sig - out_sig) ** 2))

    render_bulk_comparison(ax_bulk, tar[0], out[0], chr_info, instance_loss, mode, bulk_name, avg_loss)
    render_fiber_heatmap(ax_heat, out_fibers)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
    return fig


def plot_single_fibers_dashboard(inp, input_flags, out_fibers, locus, mode="Train"):
    """
    Constructs an ultra-compact ribbon layout comparing input channels vs continuous predicted accessibility.
    Returns Matplotlib Figure object or None if no informative fibers are found.
    """
    chr_name, start, end = locus[0][0], locus[1][0], locus[2][0]
    sequence_length = inp.shape[2]

    active_features = [(j, FEATURE_NAMES[j], FEATURE_COLORS[j]) for j, flag in enumerate(input_flags) if flag]
    num_active_features = len(active_features)

    valid_indices = filter_informative_fibers(inp, input_flags, min_m6a_sum=20, max_fibers=20)
    num_fibers_to_plot = len(valid_indices)

    if num_fibers_to_plot == 0:
        print("Warning: No fibers passed the sum(m6a) >= 20 filter step. Skipping plot generation.")
        return None

    global_max_val = max(1.0, float(max(out_fibers[0, :, f_idx].max().item() for f_idx in valid_indices)))

    fig = plt.figure(figsize=(24, 0.75 * num_fibers_to_plot))
    outer_gs = gridspec.GridSpec(num_fibers_to_plot, 2, figure=fig, width_ratios=[1, 1], hspace=0.4, wspace=0.12)

    for display_idx, fiber_idx in enumerate(valid_indices):
        # Left side: Input feature ribbons
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            num_active_features, 1,
            subplot_spec=outer_gs[display_idx, 0],
            hspace=0.0
        )

        for k, (orig_j, feat_name, color) in enumerate(active_features):
            ax_feat = fig.add_subplot(inner_gs[k, 0])
            fiber_feat = inp[0, k, :, fiber_idx].cpu().detach().numpy()
            ribbon_data = np.atleast_2d(fiber_feat > 0.5).astype(float)
            cmap_discrete = mcolors.ListedColormap(['white', color])

            ax_feat.imshow(ribbon_data, aspect='auto', cmap=cmap_discrete, interpolation='nearest', vmin=0, vmax=1)
            ax_feat.set_xlim(0, sequence_length - 1)
            ax_feat.set_xticks([])
            ax_feat.set_yticks([])
            ax_feat.set_ylabel(feat_name, fontsize=7, rotation=0, labelpad=15, va='center', fontweight='bold')

            if display_idx == 0 and k == 0:
                ax_feat.set_title(f"Input Feature Ribbons\n{chr_name}:{start}-{end}", fontsize=12, fontweight='bold', pad=15)

            if display_idx == num_fibers_to_plot - 1 and k == num_active_features - 1:
                ax_feat.set_xticks(np.linspace(0, sequence_length - 1, 5, dtype=int))
                ax_feat.set_xlabel("Genomic Position (bp)", fontsize=10)

        # Right side: Continuous accessibility ribbon
        ax_out = fig.add_subplot(outer_gs[display_idx, 1])
        pred_signal = out_fibers[0, :, fiber_idx].cpu().detach().numpy()
        output_ribbon = np.atleast_2d(pred_signal)

        img = ax_out.imshow(output_ribbon, aspect='auto', cmap='magma',
                            interpolation='nearest', vmin=0.0, vmax=global_max_val)

        ax_out.set_xlim(0, sequence_length - 1)
        ax_out.set_yticks([])
        ax_out.set_ylabel(f"Fib {fiber_idx}", fontsize=9, fontweight='bold', rotation=270, labelpad=18)
        ax_out.yaxis.set_label_position("right")

        if display_idx < num_fibers_to_plot - 1:
            ax_out.set_xticks([])
        else:
            ax_out.set_xticks(np.linspace(0, sequence_length - 1, 5, dtype=int))
            ax_out.set_xlabel("Genomic Position (bp)", fontsize=10)

        if display_idx == 0:
            ax_out.set_title(f"Imputed Continuous Ribbons ({mode} Profile - Scaled 0-{global_max_val:.2f})", fontsize=12, fontweight='bold', pad=15)

    cbar_ax = fig.add_axes([0.55, 0.02, 0.35, 0.015])
    cbar = plt.colorbar(img, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Accessibility Value (0 - {global_max_val:.2f} Dynamic Max Spectrum)', fontsize=9, fontweight='bold')

    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.93)
    return fig


def plot_loss(dir_path, losses, epoch, bulk_name):
    """Saves a plot of epoch-wise training loss."""
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


def plot_evaluator_record(record, input_flags, bulk_name="H3K27ac", mode="Val"):
    """
    Plots a multi-panel deconvolution dashboard from a single Evaluator locus record.

    Layout Architecture:
      - Left Column : Dynamic input channels stacked vertically with a boundary line separating cell types.
      - Right Column: Stacked 1D signals and single-cell heatmap:
            1. Composite Mixed Bulk (Target vs Prediction)
            2. Imputed Single-Cell Fiber Accessibility Heatmap
            3. Cell Type A Bulk Signal (Target vs Deconvoluted Prediction)
            4. Cell Type B Bulk Signal (Target vs Deconvoluted Prediction)

    Args:
        record (dict): A dictionary element from Evaluator.evaluate()['locus_records'].
                       Expected keys: 'locus', 'inputs', 'processed_fibers',
                       'pred_composite_bulk', 'target_composite_bulk',
                       'pred_cell_type_bulks', 'target_cell_type_bulks', 'cell_type_masks'.
        input_flags (list of int): 5-bit list indicating active feature channels (e.g., [1, 1, 1, 1, 1]).
        bulk_name (str): Histone modification target label (e.g., 'H3K27ac', 'H3K4me3').
        mode (str): Evaluation split label ('Val' or 'Test').

    Returns:
        matplotlib.figure.Figure: The complete figure object for saving or logging.
    """
    # 1. Unpack Evaluator Record Data
    locus = record["locus"]
    chr_name, start, end = locus[0][0], locus[1][0], locus[2][0]
    chr_info = f"{chr_name}:{start}-{end}"

    inp = record["inputs"]                          # Shape: [1, C, L, N]
    processed_fibers = record["processed_fibers"]  # Shape: [1, L, N]
    pred_composite = record["pred_composite_bulk"]  # Shape: [1, L]
    target_composite = record["target_composite_bulk"] # Shape: [1, L]
    ct_preds = record["pred_cell_type_bulks"]        # Dict: cell_type -> [1, L]
    ct_targets = record["target_cell_type_bulks"]    # Dict: cell_type -> [1, L]
    ct_masks = record["cell_type_masks"]            # Dict: cell_type -> [N]

    cell_types = sorted(list(ct_preds.keys()))
    num_active_inputs = sum(input_flags)

    # 2. Determine Fiber Boundary Index for Visual Separation
    split_idx = None
    if len(cell_types) >= 1:
        mask_ct0 = ct_masks[cell_types[0]]
        mask_ct0 = mask_ct0[0] if mask_ct0.dim() > 1 else mask_ct0
        split_idx = int(mask_ct0.sum().item())

    # 3. Figure Layout Creation
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(max(4, num_active_inputs), 2, figure=fig, width_ratios=[1, 1], wspace=0.25, hspace=0.35)

    # 4. Left Column: Render Input Feature Channels
    input_axes = render_input_channels(fig, gs, inp, input_flags)
    if input_axes:
        ct_label = f"{cell_types[0]} (Top) / {cell_types[1]} (Bottom)" if len(cell_types) == 2 else "Mixed Stack"
        input_axes[0].set_title(f"Input Fiber Stack [{ct_label}]\n{chr_info}", fontsize=12, fontweight='bold')

    # 5. Right Column Sub-Grid: Composite, Heatmap, and Deconvoluted Profiles
    rhs_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 1], hspace=0.45)

    # RHS Row 0: Composite Mixed Signal
    ax_comp = fig.add_subplot(rhs_gs[0, 0])
    render_bulk_signal_panel(
        ax_comp, target_composite, pred_composite,
        bulk_name=bulk_name, title_prefix=f"{mode} Composite Mixed Signal",
        color_pred="darkorange"
    )

    # RHS Row 1: Predicted Fiber Continuous Heatmap
    ax_heat = fig.add_subplot(rhs_gs[1, 0], sharex=ax_comp)
    render_fiber_heatmap(ax_heat, processed_fibers)

    # RHS Row 2: Cell Type A Bulk Signal Deconvolution
    if len(cell_types) >= 1:
        ct_a = cell_types[0]
        ax_cta = fig.add_subplot(rhs_gs[2, 0], sharex=ax_comp)
        render_bulk_signal_panel(
            ax_cta, ct_targets[ct_a], ct_preds[ct_a],
            bulk_name=f"{bulk_name} ({ct_a})", title_prefix=f"Deconvoluted Bulk: {ct_a}",
            color_pred="crimson"
        )

    # RHS Row 3: Cell Type B Bulk Signal Deconvolution
    if len(cell_types) >= 2:
        ct_b = cell_types[1]
        ax_ctb = fig.add_subplot(rhs_gs[3, 0], sharex=ax_comp)
        render_bulk_signal_panel(
            ax_ctb, ct_targets[ct_b], ct_preds[ct_b],
            bulk_name=f"{bulk_name} ({ct_b})", title_prefix=f"Deconvoluted Bulk: {ct_b}",
            color_pred="royalblue"
        )
        ax_ctb.set_xlabel("Genomic Position (bp)", fontsize=11)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
    return fig


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    set_seed(919)
    print("Testing utils module...")

    # Test Meter
    meter = AverageMeter()
    meter.update(2.5, n=2)
    meter.update(5.0, n=1)
    print(f"AverageMeter average: {meter.avg:.2f} (Expected: 3.33)")

    # Test Dummy Model for Pretty Print Summary
    dummy_model = nn.Sequential(
        nn.Conv1d(5, 32, kernel_size=15, padding=7),
        nn.GELU(),
        nn.Conv1d(32, 1, kernel_size=1)
    )
    print_model_summary(dummy_model, input_size=(16, 5, 2048, 200))

if __name__ == "__main__":
    tester()
