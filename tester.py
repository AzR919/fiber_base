"""
File for random testing.
Should not be in the final version
"""

import os
import wandb
import pyft
import pysam
import torch
import pyBigWig

from args import get_args

from data_utils import *
from models import *
from utils import *

import os
import torch
import numpy as np


import torch
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

def plot_dna_tensor_logo(dna_tensor, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Plots a sequence logo from a DNA tensor of shape (4, L, N).

    Parameters:
    - dna_tensor: torch.Tensor or np.ndarray of shape (4, L, N)
    - region_slice: tuple/slice, optional sub-region (start, end) to zoom in
                    e.g., (2400, 2500) for a 100 bp window in the middle.
    - title: Title for the generated matplotlib plot.
    """
    if isinstance(dna_tensor, torch.Tensor):
        dna_tensor = dna_tensor.detach().cpu().numpy()

    # 1. Average across the fibers (N dimension) -> shape: (4, L)
    # This yields the nucleotide frequency/probability at each position
    ppm = dna_tensor.mean(axis=2)

    # 2. Transpose to shape (L, 4) for tabular format
    ppm_T = ppm.T  # Shape: (L, 4)

    # 3. Convert to Pandas DataFrame expected by logomaker
    # Standard channel order convention: [A, C, G, T]
    df = pd.DataFrame(ppm_T, columns=['A', 'C', 'G', 'T'])

    # 4. If sequence length L is large (e.g., 5000 bp), slice to a readable sub-window
    if region_slice is not None:
        start, end = region_slice
        df = df.iloc[start:end].reset_index(drop=True)

    # 5. Create the sequence logo
    fig, ax = plt.subplots(figsize=(14, 3))

    # logomaker handles color schemes and character heights automatically
    logo = logomaker.Logo(
        df,
        ax=ax,
        # color_scheme='dna', # Standard colors: A=Green, C=Blue, G=Yellow/Orange, T=Red
        vpad=0.05
    )

    # Style the plot
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True)
    ax.set_ylabel("Probability / Frequency")
    ax.set_xlabel("Position (bp)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig("./test")
    return

def tester_0():

    base_dir = "./ignore/"
    epoch = 20
    # run_type = "avg"
    run_type = "sum"

    # output_name = f"{base_dir}output_{run_type}_{epoch}.npz"
    # pred_fibers_name = f"{base_dir}pred_fibers_{run_type}_{epoch}.npz"
    # target_name = f"{base_dir}target_{run_type}_{epoch}.npz"
    output_name = f"{base_dir}output_{epoch}.npz"
    pred_fibers_name = f"{base_dir}pred_fibers_{epoch}.npz"
    target_name = f"{base_dir}target_{epoch}.npz"

    output_np = np.load(output_name)["arr_0"]
    pred_fibers_np = np.load(pred_fibers_name)["arr_0"]
    target_np = np.load(target_name)["arr_0"]

    pass


if __name__=="__main__":
    tester_0()
    pass
