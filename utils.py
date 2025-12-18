"""
Common utility functions

"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt


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

def plot_sample(inp, out, tar, extra):

    # ================== 3. Plot ==================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'height_ratios': [1, 4]})

    # Top: DNase-seq
    ax1.plot(tar[0].cpu(), color='steelblue', alpha=0.7, label='Target')
    ax1.plot(out[0].cpu().detach(), color='orange', alpha=0.7, label='Model Output')
    ax1.set_ylabel("Signal")
    ax1.set_title("Model Prediction")
    ax1.legend()

    for i in range(inp.shape[2]):  # Plot first 10 fibers
        fiber = inp[0, :, i].cpu()
        m6a_positions = torch.where(fiber > 0.5)[0]
        ax2.hlines(-i, 0, len(fiber)-1, color='black', lw=0.6)
        if len(m6a_positions) > 0:
            ax2.scatter(m6a_positions, [-i]*len(m6a_positions), 
                    color='red', s=15, zorder=5)

    ax2.set_ylim(-inp.shape[2] - 0.5, 0.5)
    ax2.set_ylabel("Input Fibers")
    ax2.set_xlabel("Genomic Position")
    ax2.set_xlim(0, inp.shape[1])

    plt.tight_layout()
    plt.savefig(f"../results/model_io_plot_{extra}.png", dpi=300, bbox_inches='tight')
    plt.show()
