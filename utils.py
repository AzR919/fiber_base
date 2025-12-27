"""
Common utility functions

"""

import os
import sys

import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager

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

def plot_sample(dir, inp, out, tar, extra):

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Epoch_{extra}.png")

    # ================== 3. Plot ==================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1, 4]})

    # Top: DNase-seq
    ax1.plot(tar[0].cpu(), color='black', alpha=0.7, label='Target')
    ax1.plot(out[0].cpu().detach(), color='orange', alpha=0.7, label='Model Output')
    ax1.set_ylabel("Signal")
    ax1.set_title("Model Prediction")
    ax1.legend()

    # Mid: sum m6as
    pseudo_bulk_m6a = torch.sum(inp, dim=-1, keepdim=True)
    ax2.plot(pseudo_bulk_m6a[0].cpu(), color='steelblue', alpha=0.7, label='Target')
    ax2.set_ylabel("Total")
    ax2.set_title("sum m6as")
    ax2.legend()

    for i in range(inp.shape[2]):  # Plot first 10 fibers
        fiber = inp[0, :, i].cpu()
        m6a_positions = torch.where(fiber > 0.5)[0]
        ax3.hlines(-i, 0, len(fiber)-1, color='black', lw=0.6)
        if len(m6a_positions) > 0:
            ax3.scatter(m6a_positions, [-i]*len(m6a_positions),
                    color='red', s=15, zorder=5)

    ax3.set_ylim(-inp.shape[2] - 0.5, 0.5)
    ax3.set_ylabel("Input Fibers")
    ax3.set_xlabel("Genomic Position")
    ax3.set_xlim(0, inp.shape[1])

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
