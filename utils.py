"""
Common utility functions

"""

import os
import sys
import datetime

import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager

#--------------------------------------------------------------------------------------------------
# Helpers

def create_save_str(args):

    now = datetime.datetime.now()
    now = now.strftime("%y-%m-%d_T%H-%M-%S")

    save_str = f"{now}_{args.name_suffix}"

    return save_str

def set_seed(seed):

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)

    # If using CUDA, set seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_sample(dir, inp, out, tar, locus, extra, plot_sum=False):

    chr, start, end = locus[0][0], locus[1][0], locus[2][0]

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}.png")

    if plot_sum:
        # ================== 3. Plot ==================
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1, 4]})
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 4]})

    # Top: DNase-seq
    ax1.plot(tar[0].cpu(), color='black', alpha=0.7, label='Target')
    ax1.plot(out[0].cpu().detach(), color='orange', alpha=0.7, label='Model Output')
    ax1.set_ylabel("Signal")
    ax1.set_title(f"Model Prediction {chr}:{start}-{end}")
    ax1.legend()

    if plot_sum:
        # Mid: sum m6as
        pseudo_bulk_m6a = torch.sum(inp, dim=-1, keepdim=True)
        ax2.plot(pseudo_bulk_m6a[0].cpu(), color='steelblue', alpha=0.7, label='Target')
        ax2.set_ylabel("Total")
        ax2.set_title("sum m6as")
        ax2.legend()

    for i in range(inp.shape[2]):
        # MSP chunks: stretches of consecutive positions where fiber > 0.5
        # Find runs of consecutive 1s
        fiber = inp[0, :, i].cpu()
        masked = (fiber > 0.5).float()
        diff = torch.diff(masked, prepend=torch.tensor([0.0]), append=torch.tensor([0.0]))
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e > s:  # Valid stretch
                # Shade the MSP region (accessible chunk)
                ax3.axhspan(-i - 0.3, -i + 0.3, xmin=(s/len(fiber)).item(), xmax=(e/len(fiber)).item(),
                            color='blue', alpha=0.3, zorder=1)

        # Similar plot for nucleosome regions
        masked = (fiber < -0.5).float()
        diff = torch.diff(masked, prepend=torch.tensor([0.0]), append=torch.tensor([0.0]))
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e > s:  # Valid stretch
                # Shade the MSP region (accessible chunk)
                ax3.axhspan(-i - 0.3, -i + 0.3, xmin=(s/len(fiber)).item(), xmax=(e/len(fiber)).item(),
                            color='green', alpha=0.3, zorder=1)

    ax3.set_ylim(-inp.shape[2] - 0.5, 0.5)
    ax3.set_ylabel("Input Fibers")
    ax3.set_xlabel("Genomic Position")
    ax3.set_xlim(0, inp.shape[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def plot_sample_out_fibers(dir, inp, out, out_fibers, tar, locus, extra, plot_sum=False):
    """
    out_fibers: Predicted assay per fiber (B, L, N)
    """
    chr, start, end = locus[0][0], locus[1][0], locus[2][0]
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}.png")

    # Adjusted height ratios to include the new per-fiber prediction plot
    # ratios: Target, (Pseudo-bulk), Input Fibers, Output Fibers
    if plot_sum:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [1, 1, 4, 4]})
    else:
        fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [1, 4, 4]})

    # 1. Top: Bulk Assay (Target vs Model Bulk Output)
    ax1.plot(tar[0].cpu(), color='black', alpha=0.7, label='Target Bulk')
    ax1.plot(out[0].cpu().detach(), color='orange', alpha=0.7, label='Model Bulk Output')
    ax1.set_ylabel("Signal")
    ax1.set_title(f"Model Prediction {chr}:{start}-{end}")
    ax1.legend()

    if plot_sum:
        # 2. Mid: Simple sum of m6as (Input Signal Density)
        pseudo_bulk_m6a = torch.sum(inp, dim=-1)
        ax2.plot(pseudo_bulk_m6a[0].cpu(), color='steelblue', alpha=0.7)
        ax2.set_ylabel("Sum m6A")

    # Helper function to plot binary "runs" to avoid duplicating code
    def plot_stretches(ax, data_matrix, row_idx, threshold, color, invert=False):
        fiber = data_matrix[0, :, row_idx].cpu().detach()
        if invert:
            masked = (fiber < -threshold).float()
        else:
            masked = (fiber > threshold).float()

        diff = torch.diff(masked, prepend=torch.tensor([0.0]), append=torch.tensor([0.0]))
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e > s:
                ax.axhspan(-row_idx - 0.3, -row_idx + 0.3,
                           xmin=(s/len(fiber)).item(), xmax=(e/len(fiber)).item(),
                           color=color, alpha=0.3, zorder=1)

    # 3. Input Fibers Plot
    for i in range(inp.shape[-1]):
        plot_stretches(ax3, inp[2], i, 0.5, 'blue')     # msp
        plot_stretches(ax3, inp[3], i, 0.5, 'green')    # nuc
        plot_stretches(ax3, inp[4], i, 0.5, 'red')      # fire_msp

    ax3.set_ylabel("Input Fibers")
    ax3.set_ylim(-inp.shape[-1] - 0.5, 0.5)

    # 4. Output Predicted Fibers (Heatmap)
    # out_fibers is (B, L, N). We take the first batch and transpose to (N, L)
    # Transpose so that each row is a fiber
    pred_matrix = out_fibers[0].cpu().detach().numpy().T

    # We use 'Oranges' colormap to match your bulk output color
    img = ax4.imshow(pred_matrix, aspect='auto',
                     interpolation='nearest', origin='upper',
                     extent=[0, pred_matrix.shape[1], -pred_matrix.shape[0], 0])

    # Add a small colorbar for the heatmap
    plt.colorbar(img, ax=ax4, fraction=0.02, pad=0.01, label='Imputed Heatmap')

    ax4.set_ylabel("Predicted Assay Fibers")
    ax4.set_xlabel("Genomic Position (bp)")
    ax4.set_xlim(0, inp.shape[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(dir, losses, extra):

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}_loss.png")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss plot")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

#--------------------------------------------------------------------------------------------------
# output redirection on system level

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr at the OS level."""
    # Open devnull
    devnull = os.open(os.devnull, os.O_RDWR)

    # Save the actual stdout/stderr file descriptors to restore them later
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)

    try:
        # Flush Python's buffers first
        sys.stdout.flush()
        sys.stderr.flush()

        # Duplicate devnull onto stdout (1) and stderr (2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        yield
    finally:
        # Flush again before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore the original file descriptors
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)

        # Close the temporary file descriptors
        os.close(save_stdout)
        os.close(save_stderr)
        os.close(devnull)

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
